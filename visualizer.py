"""Interactive viewer for trained SOC-GS scenes (RGB | MS | IR panels).

WASD/mouse to move around, ``q/e`` to rotate, ``0``-``9`` to select the scene
when several parquet files are merged, ``h/p`` to hide/restore the selected
scene's points. The number of panels follows the checkpoint: bimodal scenes
(72 features) show RGB + MS, trimodal scenes (88 features) show
RGB + MS + IR.

Usage:
    python visualizer.py --parquet_path_list logs/rgb_ms/orange/best_scene.parquet \
        --camera_modality RGB
"""

import argparse

import numpy as np
import taichi as ti
import torch
from dataclasses import dataclass
from typing import List, Tuple

from socgs.common.Camera import CameraInfo
from socgs.common.GaussianPointCloudScene import GaussianPointCloudScene
from socgs.common.utils import (
    quaternion_conjugate_torch,
    quaternion_multiply_torch,
    quaternion_rotate_torch,
    SE3_to_quaternion_and_translation_torch,
)
from socgs.registry import experiment_from_parquet, import_experiment

RENDER_RGB_HEIGHT, RENDER_RGB_WIDTH = [1024, 1392]
RENDER_MS_HEIGHT, RENDER_MS_WIDTH = [240, 496]
RENDER_IR_HEIGHT, RENDER_IR_WIDTH = [1008, 1008]

INTRINSICS_RGB = torch.tensor(np.array([[1609.1, 0, 696],
                                        [0, 1607.1, 512],
                                        [0, 0, 1]]))
INTRINSICS_IR = torch.tensor(np.array([[503.40783691, 0, 1023/2],
                                       [0, 504.63809204, 1023/2],
                                       [0, 0, 1]]))
INTRINSICS_MS = torch.tensor(np.array([[706.025317785572, 0, 510/2],
                                       [0, 707.873167295786, 254/2],
                                       [0, 0, 1]]))


@ti.kernel
def torchImage2tiImage(field: ti.template(), data: ti.types.ndarray()):
    for row, col in ti.ndrange(data.shape[0], data.shape[1]):
        field[col, data.shape[0] - row - 1] = \
            ti.math.vec3(data[row, col, 0], data[row, col, 1], data[row, col, 2])


class GaussianPointVisualizer:
    @dataclass
    class GaussianPointVisualizerConfig:
        device: str = "cuda"
        image_height: int = 1024
        image_width: int = 1392
        camera_intrinsics: torch.Tensor = None
        initial_T_pointcloud_camera: torch.Tensor = None
        parquet_path_list: List[str] = None
        step_size: float = 0.1
        mouse_sensitivity: float = 3

    @dataclass
    class GaussianPointVisualizerState:
        next_t_pointcloud_camera: torch.Tensor
        next_q_pointcloud_camera: torch.Tensor
        selected_scene: int = 0
        last_mouse_pos: Tuple[float, float] = None

    @dataclass
    class ExtraSceneInfo:
        start_offset: int
        end_offset: int
        center: torch.Tensor
        visible: bool

    def __init__(self, config) -> None:
        self.config = config
        self.config.image_height -= self.config.image_height % 16
        self.config.image_width -= self.config.image_width % 16

        parquet_path = self.config.parquet_path_list
        if isinstance(parquet_path, (list, tuple)):
            assert len(parquet_path) == 1, \
                "this visualizer renders one checkpoint at a time"
            parquet_path = parquet_path[0]
        print(f"Loading {parquet_path}")
        scene = GaussianPointCloudScene.from_trained_parquet(
            parquet_path, config=GaussianPointCloudScene.PointCloudSceneConfig(
                max_num_points_ratio=None, add_sphere=False))
        self.scene = self._merge_scenes([scene])
        self.scene = self.scene.to(self.config.device)
        with torch.no_grad():
            self.scene.point_cloud[torch.isnan(self.scene.point_cloud)] = 0
            self.scene.point_cloud_features[torch.isnan(self.scene.point_cloud_features)] = 0

        # number of spectral panels follows the checkpoint (bimodal 4ch / trimodal 5ch)
        self.experiment = experiment_from_parquet(parquet_path)
        self.panels = ['rgb', 'ms', 'ir'] if self.experiment == 'rgb_ir_ms' else ['rgb', 'ms']

        initial_T_pointcloud_camera = self.config.initial_T_pointcloud_camera.to(
            self.config.device).unsqueeze(0)
        initial_q, initial_t = SE3_to_quaternion_and_translation_torch(
            initial_T_pointcloud_camera)
        self.state = self.GaussianPointVisualizerState(
            next_q_pointcloud_camera=initial_q,
            next_t_pointcloud_camera=initial_t,
            selected_scene=0,
            last_mouse_pos=None,
        )

        self.gui = ti.GUI(
            "SOC-GS Gaussian Point Visualizer",
            (self.config.image_width * len(self.panels), self.config.image_height),
            fast_gui=True)

        intrinsics = self.config.camera_intrinsics.to(self.config.device)
        self.camera_info = CameraInfo(
            camera_intrinsics=intrinsics,
            camera_width=self.config.image_width,
            camera_height=self.config.image_height,
            camera_intrinsics_multispectral=intrinsics,
            camera_height_multispectral=self.config.image_height,
            camera_width_multispectral=self.config.image_width,
            camera_intrinsics_infrared=intrinsics,
            camera_height_infrared=self.config.image_height,
            camera_width_infrared=self.config.image_width,
            camera_id=0,
        )

        rasterisation_module = import_experiment(self.experiment, 'GaussianPointCloudRasterisation')
        self.rasteriser = rasterisation_module.GaussianPointCloudRasterisation(
            config=rasterisation_module.GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=0.4,
                far_plane=2000.,
                depth_to_sort_key_scale=10.))
        self.rasterisation_input_cls = \
            rasterisation_module.GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput

        self.image_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(
            len(self.panels) * self.config.image_width, self.config.image_height))

    def _rasterise_current_view(self):
        """Rasterise val_all with the current camera and slice the panels."""
        input_data = self.rasterisation_input_cls(
            point_cloud=self.scene.point_cloud,
            point_cloud_features=self.scene.point_cloud_features,
            point_invalid_mask=self.scene.point_invalid_mask,
            point_object_id=self.scene.point_object_id,
            camera_info=self.camera_info,
            q_pointcloud_camera=self.state.next_q_pointcloud_camera,
            t_pointcloud_camera=self.state.next_t_pointcloud_camera,
            color_max_sh_band=3,
        )
        if self.experiment == 'rgb_ms':  # bimodal input carries the MS pose fields
            input_data.q_pointcloud_camera_multispectral = self.state.next_q_pointcloud_camera
            input_data.t_pointcloud_camera_multispectral = self.state.next_t_pointcloud_camera
        with torch.no_grad():
            rasterized_image, _, _ = self.rasteriser(
                input_data, current_train_stage='val_all')
        panels = []
        panels.append(rasterized_image[:, :, :3])
        panels.append(rasterized_image[:, :, 3:4].repeat(1, 1, 3))
        if self.experiment == 'rgb_ir_ms':
            panels.append(rasterized_image[:, :, 4:].repeat(1, 1, 3))
        whole_image = torch.cat(panels, dim=1)
        torchImage2tiImage(self.image_buffer, whole_image)

    def start(self):
        while self.gui.running:
            events = self.gui.get_events(self.gui.PRESS)
            start_offset = 0
            end_offset = self.scene.point_cloud.shape[0]
            selected_objects = torch.arange(
                len(self.extra_scene_info_dict), device=self.config.device)
            object_selected = self.state.selected_scene != 0
            move_factor = -1 if object_selected else 1
            if object_selected:
                info = self.extra_scene_info_dict[self.state.selected_scene - 1]
                start_offset = info.start_offset
                end_offset = info.end_offset
                selected_objects = self.state.selected_scene - 1

            for event in events:
                if event.key >= "0" and event.key <= "9":
                    scene_index = int(event.key)
                    if scene_index <= len(self.extra_scene_info_dict):
                        self.state.selected_scene = scene_index
                elif event.key == "w":
                    delta = torch.zeros_like(self.state.next_t_pointcloud_camera)
                    delta[selected_objects, 2] = self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(v=delta,
                                                    q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "s":
                    delta = torch.zeros_like(self.state.next_t_pointcloud_camera)
                    delta[selected_objects, 2] = -self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(v=delta,
                                                    q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "a":
                    delta = torch.zeros_like(self.state.next_t_pointcloud_camera)
                    delta[selected_objects, 0] = -self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(v=delta,
                                                    q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "d":
                    delta = torch.zeros_like(self.state.next_t_pointcloud_camera)
                    delta[selected_objects, 0] = self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(v=delta,
                                                    q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "-":
                    delta = torch.zeros_like(self.state.next_t_pointcloud_camera)
                    delta[selected_objects, 1] = self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(v=delta,
                                                    q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "=":
                    delta = torch.zeros_like(self.state.next_t_pointcloud_camera)
                    delta[selected_objects, 1] = -self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(v=delta,
                                                    q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "q":
                    delta_q = torch.zeros_like(self.state.next_q_pointcloud_camera)
                    delta_q[..., 3] = 1.
                    delta_q[selected_objects, 3] = np.cos(-self.config.step_size / 2 * move_factor)
                    delta_q[selected_objects, 1] = np.sin(-self.config.step_size / 2 * move_factor)
                    delta_q /= torch.norm(delta_q, dim=-1, keepdim=True)
                    self.state.next_q_pointcloud_camera = quaternion_multiply_torch(
                        self.state.next_q_pointcloud_camera, delta_q)
                    self.state.next_q_pointcloud_camera /= torch.norm(
                        self.state.next_q_pointcloud_camera, dim=-1, keepdim=True)
                elif event.key == "e":
                    delta_q = torch.zeros_like(self.state.next_q_pointcloud_camera)
                    delta_q[..., 3] = 1.
                    delta_q[selected_objects, 3] = np.cos(self.config.step_size / 2 * move_factor)
                    delta_q[selected_objects, 1] = np.sin(self.config.step_size / 2 * move_factor)
                    delta_q /= torch.norm(delta_q, dim=-1, keepdim=True)
                    self.state.next_q_pointcloud_camera = quaternion_multiply_torch(
                        self.state.next_q_pointcloud_camera, delta_q)
                    self.state.next_q_pointcloud_camera /= torch.norm(
                        self.state.next_q_pointcloud_camera, dim=-1, keepdim=True)
                elif event.key == "h":
                    self.scene.point_invalid_mask[start_offset:end_offset] = 1
                elif event.key == "p":
                    self.scene.point_invalid_mask[start_offset:end_offset] = 0

            mouse_pos = self.gui.get_cursor_pos()
            if self.gui.is_pressed(self.gui.LMB):
                if self.state.last_mouse_pos is None:
                    self.state.last_mouse_pos = mouse_pos
                else:
                    dy, dx = mouse_pos[0] - self.state.last_mouse_pos[0], \
                        mouse_pos[1] - self.state.last_mouse_pos[1]
                    angle_x = dx * self.config.mouse_sensitivity
                    angle_y = dy * self.config.mouse_sensitivity
                    if object_selected:
                        pointcloud_object_center = \
                            self.extra_scene_info_dict[self.state.selected_scene - 1].center
                        pointcloud_object_center = pointcloud_object_center.to(
                            self.state.next_t_pointcloud_camera.device).unsqueeze(0)
                        pointcloud_camera_to_center = pointcloud_object_center - \
                            self.state.next_t_pointcloud_camera[selected_objects]
                        camera_camera_to_center = quaternion_rotate_torch(
                            q=quaternion_conjugate_torch(
                                self.state.next_q_pointcloud_camera[selected_objects]),
                            v=pointcloud_camera_to_center)

                    delta_q_y = torch.zeros_like(self.state.next_q_pointcloud_camera)
                    delta_q_y[..., 3] = 1.
                    delta_q_y[selected_objects, 3] = np.cos(angle_y / 2)
                    delta_q_y[selected_objects, 1] = np.sin(angle_y / 2)
                    delta_q_y /= torch.norm(delta_q_y, dim=-1, keepdim=True)
                    self.state.next_q_pointcloud_camera = quaternion_multiply_torch(
                        self.state.next_q_pointcloud_camera, delta_q_y)
                    self.state.next_q_pointcloud_camera /= torch.norm(
                        self.state.next_q_pointcloud_camera, dim=-1, keepdim=True)

                    delta_q_x = torch.zeros_like(self.state.next_q_pointcloud_camera)
                    delta_q_x[..., 3] = 1.
                    delta_q_x[selected_objects, 3] = np.cos(angle_x / 2)
                    delta_q_x[selected_objects, 0] = np.sin(angle_x / 2)
                    delta_q_x /= torch.norm(delta_q_x, dim=-1, keepdim=True)
                    self.state.next_q_pointcloud_camera = quaternion_multiply_torch(
                        self.state.next_q_pointcloud_camera, delta_q_x)
                    self.state.next_q_pointcloud_camera /= torch.norm(
                        self.state.next_q_pointcloud_camera, dim=-1, keepdim=True)

                    if object_selected:
                        object_center_new = quaternion_rotate_torch(
                            q=self.state.next_q_pointcloud_camera[selected_objects],
                            v=camera_camera_to_center)
                        self.state.next_t_pointcloud_camera[selected_objects] = \
                            pointcloud_object_center - object_center_new

                    self.state.last_mouse_pos = mouse_pos
            else:
                self.state.last_mouse_pos = None

            self._rasterise_current_view()
            self.gui.set_image(self.image_buffer)
            self.gui.show()
        self.gui.close()

    def _merge_scenes(self, scene_list):
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
        point_object_id = torch.zeros(
            (merged_point_cloud.shape[0],), dtype=torch.int32, device=self.config.device)
        for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list)):
            point_object_id[start_offset:end_offset] = idx
        merged_scene = GaussianPointCloudScene(
            point_cloud=merged_point_cloud,
            point_cloud_features=merged_point_cloud_features,
            point_object_id=point_object_id,
            config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
        return merged_scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path_list", type=str, nargs="+",
                        default=['./logs/rgb_ms/orange/best_scene.parquet'])
    parser.add_argument("--camera_modality", type=str, default='RGB',
                        choices=['RGB', 'MS', 'IR'],
                        help="initial camera (intrinsics/image size) of the viewer")
    args = parser.parse_args()
    ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)

    config = GaussianPointVisualizer.GaussianPointVisualizerConfig(
        parquet_path_list=args.parquet_path_list)
    config.initial_T_pointcloud_camera = torch.tensor(
        [[0.9980490981, -0.0050870339, 0.062226360, -2.7901366379],
         [0.0055578843, 0.9999572038, -0.0073959841, 2.5695686101],
         [-0.0621860734, 0.0077274022, 0.9980346585, 0.3444592766],
         [0.0, 0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    if args.camera_modality == 'RGB':
        config.image_height, config.image_width = RENDER_RGB_HEIGHT, RENDER_RGB_WIDTH
        config.camera_intrinsics = torch.tensor(INTRINSICS_RGB, device='cuda', dtype=torch.float32)
    elif args.camera_modality == 'MS':
        config.image_height, config.image_width = RENDER_MS_HEIGHT, RENDER_MS_WIDTH
        config.camera_intrinsics = torch.tensor(INTRINSICS_MS, device='cuda', dtype=torch.float32)
    elif args.camera_modality == 'IR':
        config.image_height, config.image_width = RENDER_IR_HEIGHT, RENDER_IR_WIDTH
        config.camera_intrinsics = torch.tensor(INTRINSICS_IR, device='cuda', dtype=torch.float32)
    visualizer = GaussianPointVisualizer(config)
    visualizer.start()
