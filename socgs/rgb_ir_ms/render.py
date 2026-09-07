"""Same-view cross-spectral rendering and evaluation (RGB + IR + MS experiment).

Trimodal counterpart of ``socgs.rgb_ms.render``: every validation view is
rasterised with the ``val_all`` stage, whose output carries all five channels
(R, G, B, MS, IR) in a single pass, and the per-channel images are written to
``<output>/rgb``, ``<output>/ms`` and ``<output>/ir`` next to the
ground-truth images in ``<output>/GT``.

* RGB poses come from COLMAP (``datasets/<scene>/val.json``);
* the multispectral/infrared poses are the ones optimised during training
  (``logs/<scene>/<ms|ir>_iter_<iter>_r_/_t_.npy``; the latest snapshot is
  used automatically unless ``pose_iteration`` is given).

File names follow the ``TEST_ID`` view labels (e.g. ``0008.png``) of the
original release.
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from ..common.Camera import CameraInfo
from ..common.checkpoints import (
    DEFAULT_TEST_IDS,
    get_gpu_mem,
    load_learned_poses,
)
from ..common.metrics import calculate_psnr, calculate_ssim
from ..common.utils import SE3_to_quaternion_and_translation_torch, quaternion_to_rotation_matrix_torch
from .GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from .ImagePoseDataset import ImagePoseDataset

#: modalities available in the trimodal experiment
MODALITIES = ("rgb", "ms", "ir")


class GaussianPointRenderer:
    """Rasterise a trained scene from RGB, MS and/or IR viewpoints."""

    @dataclass
    class GaussianPointRendererConfig:
        parquet_path: str
        cameras: torch.Tensor  # (V, 4, 4) RGB camera poses
        cameras_ms: torch.Tensor  # (V, 4, 4) MS poses
        cameras_ir: torch.Tensor  # (V, 4, 4) IR poses
        device: str = "cuda"
        camera_id: int = 0
        image_height: int = 3008
        image_width: int = 4112
        image_height_ms: int = 254
        image_width_ms: int = 510
        image_height_ir: int = 1023
        image_width_ir: int = 1023
        camera_intrinsics: Optional[torch.Tensor] = None  # RGB intrinsics
        camera_intrinsics_ms: Optional[torch.Tensor] = None  # MS intrinsics
        camera_intrinsics_ir: Optional[torch.Tensor] = None  # IR intrinsics

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
            if self.camera_intrinsics_ir is None:
                self.camera_intrinsics_ir = torch.tensor(
                    [[503.40783691, 0.0, 1023 / 2],
                     [0.0, 504.63809204, 1023 / 2],
                     [0.0, 0.0, 1.0]])

    def __init__(self, config: GaussianPointRendererConfig) -> None:
        from ..common.GaussianPointCloudScene import (
            GaussianPointCloudScene,
            PointCloudSceneConfig,
        )
        self.config = config
        self.ToPIL = transforms.ToPILImage()
        # the rasteriser requires image sizes divisible by the tile size (16)
        for attr in ("image_width", "image_height", "image_width_ms",
                     "image_height_ms", "image_width_ir", "image_height_ir"):
            value = getattr(self.config, attr)
            setattr(self.config, attr, value - value % 16)

        scene = GaussianPointCloudScene.from_trained_parquet(
            config.parquet_path,
            config=PointCloudSceneConfig(max_num_points_ratio=None, add_sphere=False))
        self.scene = scene.to(self.config.device)
        self.cameras = self.config.cameras.to(self.config.device)
        self.cameras_ms = self.config.cameras_ms.to(self.config.device)
        self.cameras_ir = self.config.cameras_ir.to(self.config.device)

        self.camera_info = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics.to(self.config.device),
            camera_width=self.config.image_width,
            camera_height=self.config.image_height,
            camera_intrinsics_multispectral=self.config.camera_intrinsics_ms.to(self.config.device),
            camera_height_multispectral=self.config.image_height_ms,
            camera_width_multispectral=self.config.image_width_ms,
            camera_intrinsics_infrared=self.config.camera_intrinsics_ir.to(self.config.device),
            camera_height_infrared=self.config.image_height_ir,
            camera_width_infrared=self.config.image_width_ir,
            camera_id=self.config.camera_id,
        )
        self.camera_info_ms = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics_ms.to(self.config.device),
            camera_width=self.config.image_width_ms,
            camera_height=self.config.image_height_ms,
            camera_intrinsics_multispectral=self.config.camera_intrinsics_ms.to(self.config.device),
            camera_height_multispectral=self.config.image_height_ms,
            camera_width_multispectral=self.config.image_width_ms,
            camera_intrinsics_infrared=self.config.camera_intrinsics_ir.to(self.config.device),
            camera_height_infrared=self.config.image_height_ir,
            camera_width_infrared=self.config.image_width_ir,
            camera_id=self.config.camera_id,
        )
        self.camera_info_ir = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics_ir.to(self.config.device),
            camera_width=self.config.image_width_ir,
            camera_height=self.config.image_height_ir,
            camera_intrinsics_multispectral=self.config.camera_intrinsics_ms.to(self.config.device),
            camera_height_multispectral=self.config.image_height_ms,
            camera_width_multispectral=self.config.image_width_ms,
            camera_intrinsics_infrared=self.config.camera_intrinsics_ir.to(self.config.device),
            camera_height_infrared=self.config.image_height_ir,
            camera_width_infrared=self.config.image_width_ir,
            camera_id=self.config.camera_id,
        )
        self.rasteriser = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=0.4,
                far_plane=2000.,
                depth_to_sort_key_scale=10.))

    def _render_view(self, camera, camera_info):
        """Rasterise one view; returns the (H, W, 5) RGB+MS+IR image (val_all)."""
        q, t = SE3_to_quaternion_and_translation_torch(camera)
        with torch.no_grad():
            rasterized_image, _, _ = self.rasteriser(
                GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                    point_cloud=self.scene.point_cloud,
                    point_cloud_features=self.scene.point_cloud_features,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    point_object_id=self.scene.point_object_id,
                    camera_info=camera_info,
                    q_pointcloud_camera=q,
                    t_pointcloud_camera=t,
                    color_max_sh_band=3,
                ),
                current_train_stage='val_all',
            )
        return rasterized_image

    @staticmethod
    def _split_channels(rasterized_image):
        """Split the val_all output into rgb, ms (x3) and ir (x3) images."""
        rgb = torch.clamp(rasterized_image[:, :, :3], min=0, max=1).permute(2, 0, 1)
        ms = torch.clamp(rasterized_image[:, :, 3:4], min=0, max=1).repeat(1, 1, 3).permute(2, 0, 1)
        ir = torch.clamp(rasterized_image[:, :, 4:], min=0, max=1).repeat(1, 1, 3).permute(2, 0, 1)
        return rgb, ms, ir

    def run(self, output_prefix, camera_modality='rgb', file_names=None):
        """Render every view of the chosen camera set; save all channels."""
        save_dirs = {name: os.path.join(output_prefix, name)
                     for name in ('rgb', 'ms', 'ir')}
        for directory in save_dirs.values():
            os.makedirs(directory, exist_ok=True)
        render_time = []
        render_mem_occupancy = []

        camera_sets = {'rgb': (self.cameras, self.camera_info),
                       'ms': (self.cameras_ms, self.camera_info_ms),
                       'ir': (self.cameras_ir, self.camera_info_ir)}
        cameras, camera_info = camera_sets[camera_modality]
        num_cameras = cameras.shape[0]
        file_names = file_names or [f'{DEFAULT_TEST_IDS[i].zfill(4)}.png'
                                    for i in range(num_cameras)]

        for i in tqdm(range(num_cameras)):
            start_time = time.time()
            rasterized_image = self._render_view(cameras[i].unsqueeze(0), camera_info)
            end_time = time.time()
            if i > 0:  # the first view is a warm-up frame
                render_time.append(end_time - start_time)
                render_mem_occupancy.append(get_gpu_mem())

            image_rgb, image_ms, image_ir = self._split_channels(rasterized_image)
            self.ToPIL(image_rgb).save(os.path.join(save_dirs['rgb'], file_names[i]))
            self.ToPIL(image_ms).save(os.path.join(save_dirs['ms'], file_names[i]))
            self.ToPIL(image_ir).save(os.path.join(save_dirs['ir'], file_names[i]))

        print(f"Average Render Time: {sum(render_time) / len(render_time)}s")
        print(f"Average CUDA Memory Occupancy: "
              f"{sum(render_mem_occupancy) / len(render_mem_occupancy)}MB")


# --------------------------------------------------------------------- #
# same-view evaluation orchestration (used by the entry scripts)
# --------------------------------------------------------------------- #
def _frame_names(num_frames: int, val_ids: Optional[List[str]], frame_mode: str) -> List[str]:
    """Output file names: TEST_ID labels (original release) or frame_%03d."""
    if frame_mode == 'test_ids':
        ids = list(val_ids) if val_ids else DEFAULT_TEST_IDS
        return [f'{ids[i].zfill(4)}.png' for i in range(num_frames)]
    return [f'frame_{i:03}.png' for i in range(num_frames)]


def render_same_view_evaluation(*, scene, render_view, log_dir, output_prefix,
                                poses_dir, parquet_path=None, pose_iteration=None,
                                val_ids=None, taichi_memory_gb=4.0, frame_mode='test_ids',
                                gt_prefix=None):
    """Same-view cross-spectral rendering/evaluation of a trained scene.

    Parameters follow the original CLI of
    ``cross_spectral_render_w_same_view.py``: ``log_dir`` holds
    ``<scene>/best_scene.parquet`` + ``<ms|ir>_iter_*`` pose snapshots, and
    ``poses_dir`` holds ``<scene>/val.json``. ``render_view`` selects the
    camera set ('rgb' COLMAP poses, or learned 'ms'/'ir' poses).
    """
    import taichi as ti

    if render_view not in MODALITIES:
        raise ValueError(f"render_view must be one of {MODALITIES}, got {render_view!r}")

    if parquet_path is None:
        parquet_path = os.path.join(log_dir, scene, 'best_scene.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"checkpoint not found: {parquet_path} (train the scene first, "
            f"or pass --parquet_path)")
    rgb_poses_path = os.path.join(poses_dir, scene, 'val.json')
    if not os.path.exists(rgb_poses_path):
        raise FileNotFoundError(f"validation json not found: {rgb_poses_path}")

    # output layout: <output_prefix>/<scene>/<view>_view/{rgb,ms,ir,GT} (legacy
    # behaviour); gt_prefix redirects the ground-truth images when given
    output_path = os.path.join(output_prefix, scene, render_view + '_view')
    output_gt_path = os.path.join(output_path, 'GT') if gt_prefix is None else gt_prefix

    # optimised MS/IR poses (only needed when rendering the MS/IR views); the
    # latest snapshot is used unless pose_iteration is given
    pose_dir = os.path.join(log_dir, scene)
    needed = ('ms', 'ir') if render_view in ('ms', 'ir') else ()
    learned_poses = {modal: (load_learned_poses(pose_dir, modal, iteration=pose_iteration)
                             if modal in needed else None) for modal in ('ms', 'ir')}

    val_dataset = ImagePoseDataset(dataset_json_path=rgb_poses_path)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=4)
    num_frames = len(val_data_loader)
    file_names = _frame_names(num_frames, val_ids, frame_mode)

    # camera matrices of the validation views (RGB + learned MS/IR)
    cameras = {modal: torch.zeros((num_frames, 4, 4)) for modal in ('rgb', 'ms', 'ir')}
    camera_ids = []
    camera_info = camera_info_ms = camera_info_ir = None
    for idx, val_data in enumerate(val_data_loader):
        (_, q_rgb, t_rgb, _, _, _, _, _, _, camera_info, camera_info_ms, camera_info_ir) = val_data
        cameras['rgb'][idx, :3, :3] = quaternion_to_rotation_matrix_torch(q_rgb)
        cameras['rgb'][idx, :3, 3] = t_rgb
        cameras['rgb'][idx, 3, 3] = 1.0
        camera_ids.append(camera_info.camera_id)
        for modal in ('ms', 'ir'):
            learned = learned_poses.get(modal)
            if learned is None:
                continue
            q_modal = torch.tensor(learned[0][camera_info.camera_id])
            cameras[modal][idx, :3, :3] = quaternion_to_rotation_matrix_torch(q_modal)
            cameras[modal][idx, :3, 3] = torch.tensor(learned[1][camera_info.camera_id])
            cameras[modal][idx, 3, 3] = 1.0

    # ground-truth views of the evaluated modality
    gt_channel = {'rgb': 0, 'ms': 1, 'ir': 2}[render_view]
    os.makedirs(output_gt_path, exist_ok=True)
    for idx, val_data in enumerate(tqdm(val_data_loader, desc="saving GT")):
        images = (val_data[0], val_data[3], val_data[6])  # rgb / ms / ir tensors
        torchvision.transforms.functional.to_pil_image(images[gt_channel]).save(
            os.path.join(output_gt_path, file_names[idx]))

    config = GaussianPointRenderer.GaussianPointRendererConfig(
        parquet_path=parquet_path,
        cameras=cameras['rgb'],
        cameras_ms=cameras['ms'],
        cameras_ir=cameras['ir'],
        camera_id=camera_ids[0] if camera_ids else 0,
    )
    # override the camera meta data with the (autoscaled) dataset values
    config.image_width = camera_info.camera_width
    config.image_height = camera_info.camera_height
    config.camera_intrinsics = camera_info.camera_intrinsics
    config.camera_intrinsics_ms = camera_info_ms.camera_intrinsics
    config.camera_intrinsics_ir = camera_info_ir.camera_intrinsics
    config.image_width_ms = camera_info_ms.camera_width
    config.image_height_ms = camera_info_ms.camera_height
    config.image_width_ir = camera_info_ir.camera_width
    config.image_height_ir = camera_info_ir.camera_height

    ti.init(arch=ti.cuda, device_memory_GB=taichi_memory_gb, kernel_profiler=True)

    renderer = GaussianPointRenderer(config)
    renderer.run(output_path, camera_modality=render_view, file_names=file_names)

    # metrics of the evaluated view (pred vs. GT, MATLAB-compatible PSNR/SSIM)
    if os.path.isdir(output_gt_path):
        pred_dir = os.path.join(output_path, render_view)
        psnr_list, ssim_list = [], []
        for name in file_names:
            pred = cv2.imread(os.path.join(pred_dir, name))
            gt = cv2.imread(os.path.join(output_gt_path, name))
            if pred is None or gt is None:
                raise FileNotFoundError(
                    f"cannot read prediction/GT pair {name} (dir {output_path})")
            psnr_list.append(calculate_psnr(gt * 1.0, pred * 1.0))
            ssim_list.append(calculate_ssim(gt * 1.0, pred * 1.0))
        print(f"{render_view}: PSNR {float(np.mean(psnr_list)):.6f} dB, "
              f"SSIM {float(np.mean(ssim_list)):.6f}")
    else:
        print("GT directory not present -- skipped PSNR/SSIM.")
    return output_path
