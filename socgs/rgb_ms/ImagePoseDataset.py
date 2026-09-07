"""Image + pose dataset for the bimodal (RGB + multispectral) experiment.

The dataset json lists the *RGB* views (COLMAP poses). The multispectral image
of the same viewpoint is loaded from the sibling ``ms_imgs`` directory of the
RGB image (identical file names), which is possible because the RGB and MS
cameras of the capture rig are co-located. Each sample returns:

    (image, q_pointcloud_camera, t_pointcloud_camera,
     image_multispectral, q_pointcloud_camera_multispectral,
     t_pointcloud_camera_multispectral, camera_info, camera_info_ms)

where the MS pose placeholders are initialised as a copy of the RGB pose (the
per-view MS pose is optimised during training, see
``socgs.common.pose.LearnPose``) and ``camera_info_ms`` carries the
multispectral intrinsics/image size.
"""

import os.path
from typing import Any

import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from ..common.Camera import CameraInfo
from ..common.utils import SE3_to_quaternion_and_translation_torch
from .GaussianPointCloudRasterisation import TILE_WIDTH, TILE_HEIGHT

#: images are downscaled only when both sides exceed this bound
MAX_RESOLUTION_TRAIN = 1600

#: fixed multispectral intrinsics of the capture rig (registered MS camera)
INTRINSICS_MS = torch.tensor(np.array([[706.025317785572, 0, 510/2],
                                       [0, 707.873167295786, 254/2],
                                       [0, 0, 1]]))
#: native MS image size used to rescale the intrinsics after cropping
BASE_MS_HEIGHT, BASE_MS_WIDTH = np.int64([254, 510])


class ImagePoseDataset(torch.utils.data.Dataset):
    """A dataset that contains images, poses and camera intrinsics."""

    def __init__(self, dataset_json_path: str):
        super().__init__()
        required_columns = ["image_path", "T_pointcloud_camera",
                            "camera_intrinsics", "camera_height", "camera_width", "camera_id"]
        self.df = pd.read_json(dataset_json_path, orient="records")
        for column in required_columns:
            assert column in self.df.columns, f"column {column} is not in the dataset"

    def __len__(self):
        return len(self.df)

    def _pandas_field_to_tensor(self, field: Any) -> torch.Tensor:
        if isinstance(field, np.ndarray):
            return torch.from_numpy(field)
        elif isinstance(field, list):
            return torch.tensor(field)
        elif isinstance(field, torch.Tensor):
            return field

    @staticmethod
    def _autoscale_image_and_camera_info(image, image_multispectral, camera_info: CameraInfo):
        """Optionally downscale the images and return adjusted camera infos."""
        if camera_info.camera_height <= MAX_RESOLUTION_TRAIN and camera_info.camera_width <= MAX_RESOLUTION_TRAIN:
            image, camera_width, camera_height, camera_intrinsics = \
                image, camera_info.camera_width, camera_info.camera_height, camera_info.camera_intrinsics
        else:
            image, camera_width, camera_height, camera_intrinsics = \
                ImagePoseDataset._resize_image(image, camera_info.camera_width,
                                               camera_info.camera_height, camera_info.camera_intrinsics)

        if camera_info.camera_height_multispectral <= MAX_RESOLUTION_TRAIN and \
                camera_info.camera_width_multispectral <= MAX_RESOLUTION_TRAIN:
            image_multispectral, camera_width_multispectral, camera_height_multispectral, \
                camera_intrinsics_multispectral = \
                image_multispectral, camera_info.camera_width_multispectral, \
                camera_info.camera_height_multispectral, camera_info.camera_intrinsics_multispectral
        else:
            image_multispectral, camera_width_multispectral, camera_height_multispectral, \
                camera_intrinsics_multispectral = \
                ImagePoseDataset._resize_image(image_multispectral, camera_info.camera_width_multispectral,
                                               camera_info.camera_height_multispectral,
                                               camera_info.camera_intrinsics_multispectral)

        resized_camera_info = CameraInfo(
            camera_intrinsics=camera_intrinsics.to(torch.float32),
            camera_height=camera_height,
            camera_width=camera_width,
            camera_intrinsics_multispectral=camera_intrinsics_multispectral.to(torch.float32),
            camera_height_multispectral=camera_height_multispectral,
            camera_width_multispectral=camera_width_multispectral,
            camera_id=camera_info.camera_id)
        return image, image_multispectral, resized_camera_info

    @staticmethod
    def _calculate_target_image_size(original_width, original_height):
        """Round the image size down to a multiple of the rasterisation tile."""
        target_width = original_width - original_width % TILE_WIDTH
        target_height = original_height - original_height % TILE_HEIGHT
        return target_width, target_height

    @staticmethod
    def _resize_image(image: torch.Tensor, info_camera_width: int, info_camera_height: int,
                      info_intrinsics: torch.Tensor):
        image = transforms.functional.resize(image, size=3008, max_size=4112, antialias=True)
        _, camera_height, camera_width = image.shape
        camera_width, camera_height = \
            ImagePoseDataset._calculate_target_image_size(camera_width, camera_height)
        scale_x = camera_width / info_camera_width
        scale_y = camera_height / info_camera_height
        image = image[:3, :camera_height, :camera_width].contiguous()
        camera_intrinsics = info_intrinsics.clone()
        camera_intrinsics[0, 0] *= scale_x
        camera_intrinsics[1, 1] *= scale_y
        camera_intrinsics[0, 2] *= scale_x
        camera_intrinsics[1, 2] *= scale_y
        return image, camera_width, camera_height, camera_intrinsics

    @staticmethod
    def _fix_intrinsics(camera_intrinsics: torch.Tensor, scale_x: float, scale_y: float):
        camera_intrinsics[0, :] *= scale_x
        camera_intrinsics[1, :] *= scale_y
        return camera_intrinsics

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]["image_path"]
        source_path = os.path.split(os.path.split(image_path)[0])[0]
        image_path_multispectral = os.path.join(
            source_path, 'ms_imgs', os.path.split(image_path)[-1])
        T_pointcloud_camera = self._pandas_field_to_tensor(
            self.df.iloc[idx]["T_pointcloud_camera"])
        q_pointcloud_camera, t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
            T_pointcloud_camera.unsqueeze(0))
        # allocate memory for the estimated MS pose (initialised as the RGB pose)
        q_pointcloud_camera_multispectral = q_pointcloud_camera.clone().detach()
        t_pointcloud_camera_multispectral = t_pointcloud_camera.clone().detach()

        camera_intrinsics = self._pandas_field_to_tensor(self.df.iloc[idx]["camera_intrinsics"])
        camera_intrinsics_multispectral = INTRINSICS_MS
        base_camera_height = self.df.iloc[idx]["camera_height"]
        base_camera_width = self.df.iloc[idx]["camera_width"]
        base_camera_height_multispectral, base_camera_width_multispectral = BASE_MS_HEIGHT, BASE_MS_WIDTH
        camera_id = self.df.iloc[idx]["camera_id"]

        image = PIL.Image.open(image_path)
        image = torchvision.transforms.functional.to_tensor(image)
        image_multispectral = PIL.Image.open(image_path_multispectral)
        image_multispectral = torchvision.transforms.functional.to_tensor(image_multispectral)

        # use the real image size instead of the COLMAP camera width/height
        camera_height = image.shape[1]
        camera_width = image.shape[2]
        camera_height_multispectral = image_multispectral.shape[1]
        camera_width_multispectral = image_multispectral.shape[2]

        # fix the intrinsics after cropping (relative to the recorded size)
        camera_intrinsics = ImagePoseDataset._fix_intrinsics(
            camera_intrinsics, camera_width / base_camera_width, camera_height / base_camera_height)
        camera_intrinsics_multispectral = ImagePoseDataset._fix_intrinsics(
            camera_intrinsics_multispectral,
            camera_width_multispectral / base_camera_width_multispectral,
            camera_height_multispectral / base_camera_height_multispectral)

        # image width/height must be divisible by the tile size: crop the images
        camera_width, camera_height = \
            ImagePoseDataset._calculate_target_image_size(camera_width, camera_height)
        camera_width_multispectral, camera_height_multispectral = \
            ImagePoseDataset._calculate_target_image_size(
                camera_width_multispectral, camera_height_multispectral)
        image = image[:3, :camera_height, :camera_width].contiguous()
        image_multispectral = image_multispectral[:3, :camera_height_multispectral,
                                                  :camera_width_multispectral].contiguous()
        camera_info = CameraInfo(
            camera_intrinsics=camera_intrinsics,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_intrinsics_multispectral=camera_intrinsics_multispectral,
            camera_height_multispectral=camera_height_multispectral,
            camera_width_multispectral=camera_width_multispectral,
            camera_id=camera_id,
        )
        image, image_multispectral, camera_info = \
            ImagePoseDataset._autoscale_image_and_camera_info(image, image_multispectral, camera_info)
        camera_info_ms = CameraInfo(
            camera_intrinsics=camera_info.camera_intrinsics_multispectral,
            camera_height=camera_info.camera_height_multispectral,
            camera_width=camera_info.camera_width_multispectral,
            camera_intrinsics_multispectral=camera_info.camera_intrinsics_multispectral,
            camera_height_multispectral=camera_info.camera_height_multispectral,
            camera_width_multispectral=camera_info.camera_width_multispectral,
            camera_id=camera_info.camera_id,
        )
        return image, q_pointcloud_camera, t_pointcloud_camera, image_multispectral, \
               q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
               camera_info, camera_info_ms
