"""Camera data structures shared by all SOC-GS experiments.

``CameraInfo`` describes one camera: its intrinsics and image size for the RGB
view plus - for the cross-spectral cameras - the corresponding multispectral
(and infrared, in the trimodal experiment) intrinsics/image sizes. All cameras
are assumed to be co-located (same viewpoint), which is what the
same-view cross-spectral rendering relies on.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraInfo:
    camera_intrinsics: torch.Tensor  # 3x3 matrix of the RGB camera
    camera_height: int  # height of the RGB image
    camera_width: int  # width of the RGB image
    camera_intrinsics_multispectral: torch.Tensor  # 3x3 matrix of the MS camera
    camera_height_multispectral: int  # height of the MS image
    camera_width_multispectral: int  # width of the MS image
    # infrared entries are only used by the RGB+IR+MS (trimodal) experiment
    camera_intrinsics_infrared: Optional[torch.Tensor] = None  # 3x3 matrix for IR
    camera_height_infrared: Optional[int] = None  # height of the IR image
    camera_width_infrared: Optional[int] = None  # width of the IR image
    camera_id: int = 0  # camera id


@dataclass
class CameraView:
    camera_view_id: int  # camera view id
    # 4x4 SE(3) matrix, transforms points from the camera frame to the pointcloud frame
    T_pointcloud_camera: torch.Tensor
    camera_id: int  # camera id of the camera that took this view
    image_id: int  # image id of the image that was taken by this camera
    # timestamp of the image that was taken by this camera, if available, otherwise None.
    # Unit: microseconds
    timestamp: Optional[int] = None


class CameraDatabase:
    def __init__(self):
        self.camera_info_dict = {}
        self.camera_view_dict = {}

    def add_camera_info(self, camera_info: CameraInfo):
        self.camera_info_dict[camera_info.camera_id] = camera_info

    def get_camera_info(self, camera_id: int) -> CameraInfo:
        return self.camera_info_dict[camera_id]

    def add_camera_view(self, camera_view: CameraView):
        self.camera_view_dict[camera_view.camera_view_id] = camera_view

    def get_camera_view_and_info(self, camera_view_id: int) -> CameraView:
        return self.camera_view_dict[camera_view_id], self.camera_info_dict[self.camera_view_dict[camera_view_id].camera_id]
