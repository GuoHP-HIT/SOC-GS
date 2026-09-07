import os.path

import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from ..common.Camera import CameraInfo
from typing import Any
from ..common.utils import SE3_to_quaternion_and_translation_torch
from .GaussianPointCloudRasterisation import TILE_WIDTH, TILE_HEIGHT

# TILE_WIDTH = 16
# TILE_HEIGHT = 16
# MAX_RESOLUTION_TRAIN=4800
MAX_RESOLUTION_TRAIN = 1600  # original taichi gaussian splatting setting, which according to the images size incorrect
INTRINSICS_RGB = torch.tensor(np.array([[4753.35413581782, 0, 4112/2],
                                        [0, 4720.84268099054, 3008/2],
                                        [0, 0, 1]]))
INTRINSICS_IR = torch.tensor(np.array([[503.40783691, 0, 1023/2],
                                      [0, 504.63809204, 1023/2],
                                      [0, 0, 1]]))
INTRINSICS_MS = torch.tensor(np.array([[706.025317785572, 0, 510/2],
                                      [0, 707.873167295786, 254/2],
                                      [0, 0, 1]]))

class ImagePoseDataset(torch.utils.data.Dataset):
    """
    A dataset that contains images and poses, and camera intrinsics.
    """

    def __init__(self, dataset_json_path: str, dataset_json_ms_path: str):
        super().__init__()
        required_columns = ["image_path", "T_pointcloud_camera",
                            "camera_intrinsics", "camera_height", "camera_width", "camera_id"]
        self.df = pd.read_json(dataset_json_path, orient="records")
        # self.df_ms = pd.read_json(dataset_json_ms_path, orient="records")
        for column in required_columns:
            assert column in self.df.columns, f"column {column} is not in the dataset"
            # assert column in self.df_ms.columns, f"column {column} is not in the dataset"

    def __len__(self):
        # return 1 # for debugging
        return len(self.df)

    def _pandas_field_to_tensor(self, field: Any) -> torch.Tensor:
        if isinstance(field, np.ndarray):
            return torch.from_numpy(field)
        elif isinstance(field, list):
            return torch.tensor(field)
        elif isinstance(field, torch.Tensor):
            return field

    @staticmethod
    def _autoscale_image_and_camera_info(image: torch.Tensor, image_multispectral: torch.Tensor, camera_info: CameraInfo):
        if camera_info.camera_height <= MAX_RESOLUTION_TRAIN and camera_info.camera_width <= MAX_RESOLUTION_TRAIN:
            image, camera_width, camera_height, camera_intrinsics = \
                image, camera_info.camera_width, camera_info.camera_height, camera_info.camera_intrinsics
        else:
            image, camera_width, camera_height, camera_intrinsics = \
                ImagePoseDataset._resize_image(image, camera_info.camera_width, camera_info.camera_height, camera_info.camera_intrinsics)

        if camera_info.camera_height_multispectral <= MAX_RESOLUTION_TRAIN and camera_info.camera_width_multispectral <= MAX_RESOLUTION_TRAIN:
            image_multispectral, camera_width_multispectral, camera_height_multispectral, camera_intrinsics_multispectral = \
                image_multispectral, camera_info.camera_width_multispectral, camera_info.camera_height_multispectral, camera_info.camera_intrinsics_multispectral
        else:
            image_multispectral, camera_width_multispectral, camera_height_multispectral, camera_intrinsics_multispectral = \
                ImagePoseDataset._resize_image(image_multispectral, camera_info.camera_width_multispectral, camera_info.camera_height_multispectral,
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
    def _calculete_target_image_size(original_width, original_height):
        target_width = original_width - original_width % TILE_WIDTH
        target_height = original_height - original_height % TILE_HEIGHT
        return target_width, target_height

    @staticmethod
    def _resize_image(image: torch.Tensor, info_camera_width: int, info_camera_height: int,
                      info_intrinsics: torch.Tensor):
        # image = transforms.functional.resize(image, size=1024, max_size=1600, antialias=True)
        image = transforms.functional.resize(image, size=3008, max_size=4112, antialias=True)
        # image = transforms.functional.resize(image, size=512, max_size=800, antialias=True)
        _, camera_height, camera_width = image.shape
        camera_width = camera_width - camera_width % TILE_WIDTH
        camera_height = camera_height - camera_height % TILE_HEIGHT
        scale_x = camera_width / info_camera_width
        scale_y = camera_height / info_camera_height
        image = image[:3, :camera_height, :camera_width].contiguous()
        camera_intrinsics = info_intrinsics
        camera_intrinsics = camera_intrinsics.clone()
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
        image_path_multispectral = os.path.join(source_path, 'ms_imgs', os.path.split(image_path)[-1])
        # image_path_multispectral = self.df_ms.iloc[idx]["image_path"]
        T_pointcloud_camera = self._pandas_field_to_tensor(
            self.df.iloc[idx]["T_pointcloud_camera"])
        q_pointcloud_camera, t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
            T_pointcloud_camera.unsqueeze(0))
        # allocate memory for estimated pose of MS
        q_pointcloud_camera_multispectral = q_pointcloud_camera.clone().detach()
        t_pointcloud_camera_multispectral = t_pointcloud_camera.clone().detach()
        # camera_intrinsics = INTRINSICS_RGB
        camera_intrinsics = self._pandas_field_to_tensor(
                self.df.iloc[idx]["camera_intrinsics"])
        # camera_intrinsics_multispectral = self._pandas_field_to_tensor(
        #     self.df_ms.iloc[idx]["camera_intrinsics"])
        camera_intrinsics_multispectral = INTRINSICS_MS
        # fx_scale = camera_intrinsics[0, 0] / INTRINSICS_RGB[0, 0]
        # fy_scale = camera_intrinsics[1, 1] / INTRINSICS_RGB[1, 1]
        # camera_intrinsics_multispectral[0, 0] *= (fx_scale * 0.95)
        # camera_intrinsics_multispectral[1, 1] *= (fy_scale * 0.95)
        base_camera_height = self.df.iloc[idx]["camera_height"]
        base_camera_width = self.df.iloc[idx]["camera_width"]
        # base_camera_height_multispectral = self.df_ms.iloc[idx]["camera_height"]
        # base_camera_width_multispectral = self.df_ms.iloc[idx]["camera_width"]
        base_camera_height_multispectral, base_camera_width_multispectral = np.int64([254, 510])
        camera_id = self.df.iloc[idx]["camera_id"]
        image = PIL.Image.open(image_path)
        image = torchvision.transforms.functional.to_tensor(image)
        image_multispectral = PIL.Image.open(image_path_multispectral)
        image_multispectral = torchvision.transforms.functional.to_tensor(image_multispectral)

        # use real image size instead of camera_height and camera_width from colmap
        # camera_height = base_camera_height
        # camera_width = base_camera_width
        camera_height = image.shape[1]
        camera_width = image.shape[2]
        # camera_height_multispectral = base_camera_height_multispectral
        # camera_width_multispectral = base_camera_width_multispectral
        camera_height_multispectral = image_multispectral.shape[1]
        camera_width_multispectral = image_multispectral.shape[2]


        # fix intrinsics of crop
        camera_intrinsics = ImagePoseDataset._fix_intrinsics(camera_intrinsics,
                                                             camera_width / base_camera_width,
                                                             camera_height / base_camera_height)
        camera_intrinsics_multispectral = ImagePoseDataset._fix_intrinsics(camera_intrinsics_multispectral,
                                                             camera_width_multispectral / base_camera_width_multispectral,
                                                             camera_height_multispectral / base_camera_height_multispectral)
        # we want image width and height to be always divisible by 16
        # so we crop the image
        camera_width, camera_height = \
            ImagePoseDataset._calculete_target_image_size(camera_width, camera_height)
        camera_width_multispectral, camera_height_multispectral = \
            ImagePoseDataset._calculete_target_image_size(camera_width_multispectral, camera_height_multispectral)
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
        image, image_multispectral, camera_info = ImagePoseDataset._autoscale_image_and_camera_info(image, image_multispectral, camera_info)
        camera_info_ms = CameraInfo(
            camera_intrinsics=camera_info.camera_intrinsics_multispectral,
            camera_height=camera_info.camera_height_multispectral,
            camera_width=camera_info.camera_width_multispectral,
            camera_intrinsics_multispectral=camera_info.camera_intrinsics_multispectral,
            camera_height_multispectral=camera_info.camera_height_multispectral,
            camera_width_multispectral=camera_info.camera_width_multispectral,
            camera_id=camera_info.camera_id
        )
        return image, q_pointcloud_camera, t_pointcloud_camera, image_multispectral, \
               q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, camera_info, camera_info_ms


if __name__ == "__main__":
    dataset_json_path = './datasets/penguin_split/rgb//train.json'
    dataset_json_path_ms = './datasets/penguin_split/ms/train.json'
    dataloader = ImagePoseDataset(dataset_json_path, dataset_json_path_ms)