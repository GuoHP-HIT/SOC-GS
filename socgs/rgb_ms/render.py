"""Same-view cross-spectral rendering and evaluation (RGB + MS experiment).

This module implements the "cross-spectral rendering from the same viewpoint"
used for the quantitative evaluation of the paper:

* the RGB checkpoint poses come from COLMAP (``datasets/<scene>/val.json``);
* the multispectral poses are the ones optimised during training, stored as
  ``logs/<scene>/ms_iter_<iter>_r_.npy`` / ``ms_iter_<iter>_t_.npy`` (the
  latest snapshot is used automatically unless ``pose_iteration`` is given);
* every validation view is rasterised once per modality with the
  ``val_all`` stage (RGB + MS channels in one pass) and the per-channel images
  are written to ``<output>/rgb``, ``<output>/ms`` together with the
  ground-truth images in ``<output>/GT`` (file names follow the ``TEST_ID``
  view labels, e.g. ``0008.png``);
* per-modality PSNR/SSIM (MATLAB-compatible, see ``socgs.common.metrics``)
  and the average render time / GPU memory are printed.

The identical logic for the trimodal experiment lives in
``socgs.rgb_ir_ms.render``.
"""

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import taichi as ti
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from ..common.Camera import CameraInfo
from ..common.metrics import calculate_psnr, calculate_ssim
from ..common.utils import SE3_to_quaternion_and_translation_torch, quaternion_to_rotation_matrix_torch
from .GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from .ImagePoseDataset import ImagePoseDataset

#: view labels used for the validation images of every scene (see README)
DEFAULT_TEST_IDS = ['8', '10', '14', '20', '22']

GPU_ID = 0
_nvml_available = None


def _nvml():
    """Lazy pynvml initialisation (used for the GPU memory measurement)."""
    global _nvml_available
    if _nvml_available is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _nvml_available = pynvml
        except Exception:  # noqa: BLE001 - memory reporting is optional
            _nvml_available = False
    return _nvml_available


def get_gpu_mem():
    """Currently used GPU memory in MB (0 if nvidia-ml is unavailable)."""
    pynvml = _nvml()
    if not pynvml:
        return 0.0
    handler = pynvml.nvmlDeviceGetHandleByIndex(GPU_ID)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    return round(meminfo.used / 1024 / 1024, 2)


# --------------------------------------------------------------------- #
# learned-pose files
# --------------------------------------------------------------------- #
def list_pose_iterations(log_dir: str, modal: str) -> List[int]:
    """Return the training iterations for which ``<modal>_iter_*_r_.npy`` exists."""
    iterations = []
    pattern = re.compile(rf"^{modal}_iter_(\d+)_r_\.npy$")
    if os.path.isdir(log_dir):
        for name in os.listdir(log_dir):
            match = pattern.match(name)
            if match:
                iterations.append(int(match.group(1)))
    return sorted(iterations)


def load_learned_poses(log_dir: str, modal: str, iteration: Optional[int] = None):
    """Load the optimised ``modal`` poses of the latest (or given) iteration.

    Returns ``(q, t)`` numpy arrays of shape (num_cameras, 1, 4) / (N, 1, 3).
    """
    iterations = list_pose_iterations(log_dir, modal)
    if not iterations:
        raise FileNotFoundError(
            f"no '{modal}_iter_*_r_.npy' pose files found under {log_dir!r} "
            f"(train the scene first, see README)")
    selected = iteration if iteration is not None else iterations[-1]
    if selected not in iterations:
        raise FileNotFoundError(
            f"pose iteration {selected} not found under {log_dir!r}; "
            f"available iterations: {iterations}")
    q_path = os.path.join(log_dir, f"{modal}_iter_{selected:06d}_r_.npy")
    t_path = os.path.join(log_dir, f"{modal}_iter_{selected:06d}_t_.npy")
    return np.load(q_path), np.load(t_path)


def learned_poses_to_cameras(q_np, t_np, camera_ids) -> torch.Tensor:
    """Build the (V, 4, 4) camera matrices of the learned poses for given ids."""
    num_cameras = len(camera_ids)
    cameras = torch.zeros((num_cameras, 4, 4))
    for idx, camera_id in enumerate(camera_ids):
        q = torch.tensor(q_np[camera_id])
        t = torch.tensor(t_np[camera_id])
        r = quaternion_to_rotation_matrix_torch(q)
        cameras[idx, :3, :3] = r
        cameras[idx, :3, 3] = t
        cameras[idx, 3, 3] = 1.0
    return cameras


# --------------------------------------------------------------------- #
# renderer
# --------------------------------------------------------------------- #
class GaussianPointRenderer:
    """Rasterise a trained scene from RGB and/or multispectral viewpoints."""

    @dataclass
    class GaussianPointRendererConfig:
        parquet_path: str
        cameras: torch.Tensor  # (V, 4, 4) RGB camera poses
        cameras_ms: torch.Tensor  # (V, 4, 4) learned MS camera poses
        device: str = "cuda"
        camera_id: int = 0
        image_height: int = 3008
        image_width: int = 4112
        image_height_ms: int = 254
        image_width_ms: int = 510
        camera_intrinsics: Optional[torch.Tensor] = None  # RGB intrinsics
        camera_intrinsics_ms: Optional[torch.Tensor] = None  # MS intrinsics

        def __post_init__(self):
            if self.camera_intrinsics is None:
                self.camera_intrinsics = torch.tensor(
                    [[4753.35413581782, 0.0, 4112 / 2],
                     [0.0, 4720.84268099054, 3008 / 2],
                     [0.0, 0.0, 1.0]])
            if self.camera_intrinsics_ms is None:
                self.camera_intrinsics_ms = torch.tensor(
                    [[706.025317785572, 0.0, 510 / 2],
                     [0.0, 707.873167295786, 254 / 2],
                     [0.0, 0.0, 1.0]])

    @dataclass
    class ExtraSceneInfo:
        start_offset: int
        end_offset: int
        center: torch.Tensor
        visible: bool

    def __init__(self, config: GaussianPointRendererConfig) -> None:
        from ..common.GaussianPointCloudScene import (
            GaussianPointCloudScene,
            PointCloudSceneConfig,
        )
        self.config = config
        self.ToPIL = transforms.ToPILImage()
        # the rasteriser requires image sizes divisible by the tile size (16)
        self.config.image_height -= self.config.image_height % 16
        self.config.image_width -= self.config.image_width % 16
        self.config.image_height_ms -= self.config.image_height_ms % 16
        self.config.image_width_ms -= self.config.image_width_ms % 16

        scene = GaussianPointCloudScene.from_trained_parquet(
            config.parquet_path,
            config=PointCloudSceneConfig(max_num_points_ratio=None, add_sphere=False))
        self.scene = self._merge_scenes([scene])
        self.scene = self.scene.to(self.config.device)
        self.cameras = self.config.cameras.to(self.config.device)
        self.cameras_ms = self.config.cameras_ms.to(self.config.device)
        self.camera_info = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics.to(self.config.device),
            camera_width=self.config.image_width,
            camera_height=self.config.image_height,
            camera_intrinsics_multispectral=self.config.camera_intrinsics_ms.to(self.config.device),
            camera_height_multispectral=self.config.image_height_ms,
            camera_width_multispectral=self.config.image_width_ms,
            camera_id=self.config.camera_id,
        )
        self.camera_info_ms = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics_ms.to(self.config.device),
            camera_width=self.config.image_width_ms,
            camera_height=self.config.image_height_ms,
            camera_intrinsics_multispectral=self.config.camera_intrinsics_ms.to(self.config.device),
            camera_height_multispectral=self.config.image_height_ms,
            camera_width_multispectral=self.config.image_width_ms,
            camera_id=self.config.camera_id,
        )
        self.rasteriser = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=0.4,
                far_plane=2000.,
                depth_to_sort_key_scale=10.))

    def _merge_scenes(self, scene_list):
        from ..common.GaussianPointCloudScene import (
            GaussianPointCloudScene,
            PointCloudSceneConfig,
        )
        merged_point_cloud = torch.cat([scene.point_cloud for scene in scene_list], dim=0)
        merged_point_cloud_features = \
            torch.cat([scene.point_cloud_features for scene in scene_list], dim=0)
        num_of_points_list = [scene.point_cloud.shape[0] for scene in scene_list]
        start_offset_list = [0] + np.cumsum(num_of_points_list).tolist()[:-1]
        end_offset_list = np.cumsum(num_of_points_list).tolist()
        self.extra_scene_info_dict = {
            idx: self.ExtraSceneInfo(
                start_offset=start_offset,
                end_offset=end_offset,
                center=scene_list[idx].point_cloud.mean(dim=0),
                visible=True,
            ) for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list))
        }
        point_object_id = \
            torch.zeros((merged_point_cloud.shape[0],), dtype=torch.int32, device=self.config.device)
        for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list)):
            point_object_id[start_offset:end_offset] = idx
        merged_scene = GaussianPointCloudScene(
            point_cloud=merged_point_cloud,
            point_cloud_features=merged_point_cloud_features,
            point_object_id=point_object_id,
            config=PointCloudSceneConfig(max_num_points_ratio=None))
        return merged_scene

    def _render_view(self, camera, camera_info, stage='val_all'):
        """Rasterise one view; returns the (H, W, 4) RGB+MS image (val_all)."""
        q, t = SE3_to_quaternion_and_translation_torch(camera)
        with torch.no_grad():
            rasterized_image, rasterized_depth, pixel_valid_point_count = self.rasteriser(
                GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                    point_cloud=self.scene.point_cloud,
                    point_cloud_features=self.scene.point_cloud_features,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    point_object_id=self.scene.point_object_id,
                    camera_info=camera_info,
                    q_pointcloud_camera=q,
                    t_pointcloud_camera=t,
                    q_pointcloud_camera_multispectral=q,
                    t_pointcloud_camera_multispectral=t,
                    color_max_sh_band=3,
                ),
                current_train_stage=stage,
            )
        return rasterized_image

    @staticmethod
    def _split_channels(rasterized_image):
        """Split the val_all output into rgb (3) and ms (gray, repeated 3x)."""
        rasterized_image_rgb = rasterized_image[:, :, :3]
        rasterized_image_ms = rasterized_image[:, :, 3:4].repeat(1, 1, 3)
        rasterized_image_rgb = torch.clamp(rasterized_image_rgb, min=0, max=1).permute(2, 0, 1)
        rasterized_image_ms = torch.clamp(rasterized_image_ms, min=0, max=1).permute(2, 0, 1)
        return rasterized_image_rgb, rasterized_image_ms

    def run(self, output_prefix, camera_modality='rgb', test_ids=None):
        """Render every view and save rgb/ms images (and GT where provided).

        Returns the dict of rendered image tensors per modality.
        """
        save_dir_rgb = os.path.join(output_prefix, 'rgb')
        save_dir_ms = os.path.join(output_prefix, 'ms')
        os.makedirs(save_dir_rgb, exist_ok=True)
        os.makedirs(save_dir_ms, exist_ok=True)
        test_ids = test_ids or DEFAULT_TEST_IDS
        render_time = []
        render_mem_occupancy = []

        num_cameras = self.cameras.shape[0]
        camera_sets = {'rgb': (self.cameras, self.camera_info, save_dir_rgb),
                       'ms': (self.cameras_ms, self.camera_info_ms, save_dir_ms)}
        cameras, camera_info, save_dir = camera_sets[camera_modality]

        for i in tqdm(range(num_cameras)):
            c = cameras[i, :, :].unsqueeze(0)
            start_time = time.time()
            rasterized_image = self._render_view(c, camera_info)
            end_time = time.time()
            if i > 0:  # the first view is a warm-up frame
                render_time.append(end_time - start_time)
                render_mem_occupancy.append(get_gpu_mem())

            image_rgb, image_ms = self._split_channels(rasterized_image)
            frame_name = f'{test_ids[i].zfill(4)}.png'
            self.ToPIL(image_rgb).save(os.path.join(save_dir_rgb, frame_name))
            self.ToPIL(image_ms).save(os.path.join(save_dir_ms, frame_name))

        avg_time = sum(render_time) / len(render_time)
        avg_mem = sum(render_mem_occupancy) / len(render_mem_occupancy)
        print(f"Average Render Time: {avg_time}s")
        print(f"Average CUDA Memory Occupancy: {avg_mem}MB")
        return {'rgb': save_dir_rgb, 'ms': save_dir_ms}


def _dump_gt(val_loader, modality, output_gt_path, test_ids):
    """Save the ground-truth views of ``modality`` (autoscaled like training)."""
    os.makedirs(output_gt_path, exist_ok=True)
    for idx, val_data in enumerate(tqdm(val_loader, desc="saving GT")):
        image_gt, _, _, image_gt_ms, _, _, _, _ = val_data
        if modality == 'rgb':
            image = image_gt
        elif modality == 'ms':
            image = image_gt_ms
        else:
            raise ValueError(f"unknown modality {modality}")
        image = torchvision.transforms.functional.to_pil_image(image)
        image.save(os.path.join(output_gt_path, f'{test_ids[idx].zfill(4)}.png'))


def compute_image_metrics(output_path, modalities=('rgb', 'ms'), test_ids=None):
    """Average PSNR/SSIM of the rendered images vs GT (cv2/MATLAB compatible)."""
    test_ids = test_ids or DEFAULT_TEST_IDS
    metrics = {}
    for modality in modalities:
        pred_dir = os.path.join(output_path, modality)
        gt_dir = os.path.join(output_path, 'GT')
        psnr_list, ssim_list = [], []
        for view_id in test_ids:
            pred = cv2.imread(os.path.join(pred_dir, f'{view_id.zfill(4)}.png'))
            gt = cv2.imread(os.path.join(gt_dir, f'{view_id.zfill(4)}.png'))
            if pred is None or gt is None:
                raise FileNotFoundError(
                    f"cannot read prediction/GT pair for {view_id} "
                    f"(pred_dir={pred_dir}, gt_dir={gt_dir})")
            psnr_list.append(calculate_psnr(gt * 1.0, pred * 1.0))
            ssim_list.append(calculate_ssim(gt * 1.0, pred * 1.0))
        metrics[modality] = (float(np.mean(psnr_list)), float(np.mean(ssim_list)))
        print(f"{modality}: PSNR {metrics[modality][0]:.6f} dB, "
              f"SSIM {metrics[modality][1]:.6f}")
    return metrics
