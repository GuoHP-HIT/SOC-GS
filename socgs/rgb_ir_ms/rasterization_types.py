"""Configuration and typed inputs for RGB-IR-MS rasterization."""

from dataclasses import dataclass

import torch
from dataclass_wizard import YAMLWizard

from ..common.Camera import CameraInfo


@dataclass
class GaussianPointCloudRasterisationConfig(YAMLWizard):
    near_plane: float = 0.4
    far_plane: float = 2000.
    depth_to_sort_key_scale: float = 10.
    rgb_only: bool = False
    grad_color_factor = 5.
    grad_high_order_color_factor = 1.
    grad_s_factor = 0.5
    grad_q_factor = 1.
    grad_alpha_factor = 20.
    pose_factor = 1.


@dataclass
class GaussianPointCloudRasterisationInput:
    point_cloud: torch.Tensor
    point_cloud_features: torch.Tensor
    point_object_id: torch.Tensor
    point_invalid_mask: torch.Tensor
    camera_info: CameraInfo
    q_pointcloud_camera: torch.Tensor
    t_pointcloud_camera: torch.Tensor
    color_max_sh_band: int = 2


@dataclass
class BackwardValidPointHookInput:
    point_id_in_camera_list: torch.Tensor
    grad_point_in_camera: torch.Tensor
    grad_pointfeatures_in_camera: torch.Tensor
    grad_viewspace: torch.Tensor
    magnitude_grad_viewspace: torch.Tensor
    magnitude_grad_viewspace_on_image: torch.Tensor
    num_overlap_tiles: torch.Tensor
    num_affected_pixels: torch.Tensor
    point_depth: torch.Tensor
    point_uv_in_camera: torch.Tensor
    hook_modality: str
