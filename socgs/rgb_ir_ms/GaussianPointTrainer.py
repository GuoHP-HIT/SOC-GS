# %%
from ..common.GaussianPointCloudScene import GaussianPointCloudScene
from ..common.pose import LearnPose
from .ImagePoseDataset import ImagePoseDataset
from .GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from .GaussianPointAdaptiveController import GaussianPointAdaptiveController
from ..common.LossFunction import LossFunction
import torch
from dataclass_wizard import YAMLWizard
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from pytorch_msssim import ssim
from tqdm import tqdm
import taichi as ti
import os
from collections import deque
import numpy as np
from typing import Optional
import cv2

# from pyecharts.charts import Scatter3D
# scatter3D = Scatter3D()

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

class GaussianPointCloudTrainer:
    @dataclass
    # this repo test for RGB and multispectral cross-spectral render, so the extra modality refers to multispectral
    class TrainConfig(YAMLWizard):
        train_dataset_json_path: str = ""
        val_dataset_json_path: str = ""
        train_dataset_json_path_ms: str = ""
        full_train_dataset_json_path: str = ""
        val_dataset_json_path_ms: str = ""
        pointcloud_parquet_path: str = ""
        val_image_save_path: str = ""
        num_iterations: int = 40000
        warmup_single_modality_iterations: int = 5000
        fine_bundle_adjustment_iteration: int = 10000
        ending_iterations: int = 500
        val_interval: int = 100000
        feature_learning_rate: float = 1e-3
        position_learning_rate: float = 1e-5
        pose_learning_rate: float = 1e-3
        position_learning_rate_decay_rate: float = 0.97
        pose_learning_rate_decay_rate: float = 0.97
        pose_learning_rate_decay_interval: int = 300
        position_learning_rate_decay_interval: int = 300
        increase_color_max_sh_band_interval: int = 1000.
        log_loss_interval: int = 10
        log_metrics_interval: int = 100
        scatter3D_plot_frequency: int = 200000
        print_metrics_to_console: bool = True
        log_image_interval: int = 1000
        enable_taichi_kernel_profiler: bool = False
        log_taichi_kernel_profile_interval: int = 1000
        log_validation_image: bool = True
        initial_downsample_factor: int = 1
        half_downsample_factor_interval: int = 250
        summary_writer_log_dir: str = "logs"
        output_model_dir: Optional[str] = None
        rasterisation_config: GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig()
        adaptive_controller_config: GaussianPointAdaptiveController.GaussianPointAdaptiveControllerConfig = GaussianPointAdaptiveController.GaussianPointAdaptiveControllerConfig()
        gaussian_point_cloud_scene_config: GaussianPointCloudScene.PointCloudSceneConfig = GaussianPointCloudScene.PointCloudSceneConfig()
        loss_function_config: LossFunction.LossFunctionConfig = LossFunction.LossFunctionConfig()

    def __init__(self, config: TrainConfig):
        self.config = config
        # create the log directory if it doesn't exist
        os.makedirs(self.config.summary_writer_log_dir, exist_ok=True)
        if self.config.output_model_dir is None:
            self.config.output_model_dir = self.config.summary_writer_log_dir
            os.makedirs(self.config.output_model_dir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=self.config.summary_writer_log_dir)

        # load train & test dataset
        # (train_dataset contains 25 RGB images and only use for RGB rasterisation optimization;
        # cs_train_dataset contains 30 IR and MS images used for initialize BA and joint optimization
        # because of the poses of MS and IR images are unknown, we need to optimize its pose)
        # (image, q_pointcloud_camera, t_pointcloud_camera, \
        # image_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
        # image_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
        # camera_info, camera_info_ms, camera_info_ir)
        self.train_dataset = ImagePoseDataset(
            dataset_json_path=self.config.train_dataset_json_path)
        self.cs_train_dataset = ImagePoseDataset(
            dataset_json_path=self.config.full_train_dataset_json_path)
        self.val_dataset = ImagePoseDataset(
            dataset_json_path=self.config.val_dataset_json_path)

        # load parquet point cloud and move to GPU
        self.scene = GaussianPointCloudScene.from_parquet(
            self.config.pointcloud_parquet_path, config=self.config.gaussian_point_cloud_scene_config)
        self.scene = self.scene.cuda()

        # load adaptive gaussian controller and rasterisation
        self.adaptive_controller = GaussianPointAdaptiveController(
            config=self.config.adaptive_controller_config,
            maintained_parameters=GaussianPointAdaptiveController.GaussianPointAdaptiveControllerMaintainedParameters(
                pointcloud=self.scene.point_cloud,
                pointcloud_features=self.scene.point_cloud_features,
                point_invalid_mask=self.scene.point_invalid_mask,
                point_object_id=self.scene.point_object_id,
            ))
        self.rasterisation = GaussianPointCloudRasterisation(
            config=self.config.rasterisation_config,
            backward_valid_point_hook=self.adaptive_controller.update,
        )

        self.loss_function = LossFunction(
            config=self.config.loss_function_config
        )

        self.toPIL = transforms.ToPILImage()
        os.makedirs(self.config.val_image_save_path, exist_ok=True)

        self.best_psnr_score = 0.
        self.best_ssim_score = 0.
        self.best_psnr_score_ms = 0.
        self.best_ssim_score_ms = 0.
        self.best_psnr_score_ir = 0.
        self.best_ssim_score_ir = 0.
        self.best_psnr_score_store_parquet = 0.

    def train(self):
        ti.init(arch=ti.cuda, device_memory_GB=0.1, debug=True,
                kernel_profiler=self.config.enable_taichi_kernel_profiler, )  # we don't use taichi fields, so we don't need to allocate memory, but taichi requires the memory to be allocated > 0
        train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=None, shuffle=True, pin_memory=True, num_workers=4)
        cs_train_data_loader = torch.utils.data.DataLoader(
            self.cs_train_dataset, batch_size=None, shuffle=True, pin_memory=True, num_workers=4)

        val_data_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=4)
        train_data_loader_iter = cycle(train_data_loader)
        cs_train_data_loader_iter = cycle(cs_train_data_loader)

        optimizer = torch.optim.Adam(
            [self.scene.point_cloud_features], lr=self.config.feature_learning_rate, betas=(0.9, 0.999))
        position_optimizer = torch.optim.Adam(
            [self.scene.point_cloud], lr=self.config.position_learning_rate, betas=(0.9, 0.999))

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=position_optimizer, gamma=self.config.position_learning_rate_decay_rate)

        downsample_factor = self.config.initial_downsample_factor

        recent_losses = deque(maxlen=100)

        # all modalities to optimize
        full_modality_pool = ['RGB', 'MS', 'IR']
        SH_initialize_pool = ['MS', 'IR']
        modality_cycle_pool = cycle(full_modality_pool)
        SH_cycle_pool = cycle(SH_initialize_pool)

        previous_problematic_iteration = -1000
        scatter_data_3D = []
        for iteration in tqdm(range(self.config.num_iterations)):
            '''
            Step1: single modality warmup
            in single modality warmup iterations, the multispectral image do not take part in the training process
            so that the 3dgs could initialized  by the rgb modality
            at the end of single modality warmup iterations, set pointcloud_feature[56:72] as mean of sh-r, sh-g, sh-b
            '''
            if iteration <= self.config.warmup_single_modality_iterations:
                # send data to dataloader
                # image_gt, q_pointcloud_camera, t_pointcloud_camera, _, _, _, _, _, _, \
                # camera_info, camera_info_ms, camera_info_ir = next(train_data_loader_iter)
                image_gt, q_pointcloud_camera, t_pointcloud_camera, \
                image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
                image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
                camera_info, camera_info_ms, camera_info_ir = next(train_data_loader_iter)

                assert camera_info.camera_id == camera_info_ms.camera_id == camera_info_ir.camera_id

                # send data to cuda
                image_gt = image_gt.cuda()
                q_pointcloud_camera = q_pointcloud_camera.cuda()
                t_pointcloud_camera = t_pointcloud_camera.cuda()
                camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
                camera_info.camera_width = int(camera_info.camera_width)
                camera_info.camera_height = int(camera_info.camera_height)

                # Rasterisation RGB modality
                optimizer.zero_grad()
                position_optimizer.zero_grad()
                gaussian_point_cloud_rasterisation_input = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                    point_cloud=self.scene.point_cloud,
                    point_cloud_features=self.scene.point_cloud_features,
                    point_object_id=self.scene.point_object_id,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    camera_info=camera_info,
                    q_pointcloud_camera=q_pointcloud_camera,
                    t_pointcloud_camera=t_pointcloud_camera,
                    color_max_sh_band=iteration // self.config.increase_color_max_sh_band_interval,
                )
                rasterized_image, rasterized_depth, pixel_valid_point_count = self.rasterisation(
                    gaussian_point_cloud_rasterisation_input,
                    current_train_stage='s1',
                )
                image_pred = rasterized_image
                image_depth = rasterized_depth
                # clip to [0, 1]
                image_pred = torch.clamp(image_pred, min=0, max=1)
                # hxwx3->3xhxw
                image_pred = image_pred.permute(2, 0, 1)

                loss, l1_loss, ssim_loss = self.loss_function(
                    image_pred,
                    image_gt,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    pointcloud_features=self.scene.point_cloud_features)
                loss.backward()
                optimizer.step()
                position_optimizer.step()
                recent_losses.append(loss.item())
                self.adaptive_controller.refinement()

            '''
            Step2: cross-spectral modality fine-tune
            in this step, using FOV as the standard, select the modality with the largest FOV as the main modality,
            and other modalities as auxiliary modalities. The main modality participates in the optimization process
            of 3DGS' mean, covariance, scale and alpha, while the auxiliary modality only updates the corresponding
            spherical harmonic based on the optimization results of the main modality. At the same time, LoFTR is 
            introduced to register fixed 3DGS rendering results to optimize cross-spectral poses.  
            '''

            # ->Step2 Part0: Using pretrained RGB 3DGS to initialize MS color sphere(proved useless)
            if iteration == self.config.warmup_single_modality_iterations:
                # initial MS and IR Gaussians
                optimizer.zero_grad()
                position_optimizer.zero_grad()
                print('Set the mean of RGB-SH as the initial value of MS-SH and IR-SH')
                # render initial image
                with torch.no_grad():
                    cs_initial = self.scene.point_cloud_features[:, 8 + 16:24 + 16]
                    self.scene.point_cloud_features[:, 56:56 + 16] = cs_initial
                    self.scene.point_cloud_features[:, 72:72 + 16] = cs_initial
                del cs_initial

                # Initialize MS pose by RGB rasterisation
                # initialize pose buffer(use SO3)
                initial_ms_pose_r = np.zeros([len(cs_train_data_loader), 3])
                initial_ms_pose_t = np.zeros([len(cs_train_data_loader), 3])

                # pose estimate initialization
                with torch.no_grad():
                    scene_view_num = len(cs_train_data_loader)
                    initial_count = []
                    while len(initial_count) < scene_view_num:
                        image_gt, q_pointcloud_camera, t_pointcloud_camera, \
                            image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
                            image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
                            camera_info, camera_info_ms, camera_info_ir = next(cs_train_data_loader_iter)
                        assert camera_info.camera_id == camera_info_ms.camera_id == camera_info_ir.camera_id
                        if str(camera_info.camera_id) in initial_count:
                            pass
                        else:
                            print(f"Initializing MS Image id:{camera_info.camera_id}")
                            initial_count.append(str(camera_info.camera_id))
                            # send data to cuda
                            image_gt = image_gt.cuda()
                            image_gt_multispectral = image_gt_multispectral.cuda()
                            q_pointcloud_camera = q_pointcloud_camera.cuda()
                            t_pointcloud_camera = t_pointcloud_camera.cuda()
                            q_pointcloud_camera_multispectral = q_pointcloud_camera_multispectral.cuda()
                            t_pointcloud_camera_multispectral = t_pointcloud_camera_multispectral.cuda()
                            # Obviously there is redundancy here, and optimization will be carried out in the future
                            camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
                            camera_info.camera_intrinsics_multispectral = camera_info.camera_intrinsics_multispectral.cuda()
                            camera_info.camera_width = int(camera_info.camera_width)
                            camera_info.camera_height = int(camera_info.camera_height)
                            camera_info.camera_width_multispectral = int(camera_info.camera_width_multispectral)
                            camera_info.camera_height_multispectral = int(camera_info.camera_height_multispectral)
                            camera_info_ms.camera_intrinsics = camera_info_ms.camera_intrinsics.cuda()
                            camera_info_ms.camera_intrinsics_multispectral = camera_info_ms.camera_intrinsics_multispectral.cuda()
                            camera_info_ms.camera_width = int(camera_info_ms.camera_width)
                            camera_info_ms.camera_height = int(camera_info_ms.camera_height)
                            camera_info_ms.camera_width_multispectral = int(camera_info_ms.camera_width_multispectral)
                            camera_info_ms.camera_height_multispectral = int(camera_info_ms.camera_height_multispectral)
                            # use estimated MS pose to render RGB images
                            gaussian_point_cloud_rasterisation_input_ms = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                                point_cloud=self.scene.point_cloud,
                                point_cloud_features=self.scene.point_cloud_features,
                                point_object_id=self.scene.point_object_id,
                                point_invalid_mask=self.scene.point_invalid_mask,
                                camera_info=camera_info_ms,
                                q_pointcloud_camera=q_pointcloud_camera,
                                t_pointcloud_camera=t_pointcloud_camera,
                                color_max_sh_band=3,
                            )
                            rasterized_image, keypoints_in_pred, keypoints_in_gt, q_pointcloud_camera_ms, t_pointcloud_camera_ms, vector_rotation, vector_translation = \
                                self.rasterisation(gaussian_point_cloud_rasterisation_input_ms,
                                                   current_train_stage='pose_initialize',
                                                   reference_ms_image=image_gt_multispectral)
                            # save the images with keypoints
                            image_pred = torch.clamp(rasterized_image, min=0, max=1)
                            image_pred = image_pred.permute(2, 0, 1)
                            image_pred_name = 'before_BA_initialize_ms_id_' + str(camera_info.camera_id) + '.png'
                            rgb_image_pred_path = os.path.join(self.config.val_image_save_path, image_pred_name)
                            rgb_image_pred = self.toPIL(image_pred)
                            rgb_image_pred.save(rgb_image_pred_path)
                            image_gt_name = 'keypoint_ms_gt_id_' + str(camera_info.camera_id) + '.png'
                            ms_image_gt_path = os.path.join(self.config.val_image_save_path, image_gt_name)
                            ms_image_gt = self.toPIL(image_gt_multispectral)
                            ms_image_gt.save(ms_image_gt_path)
                            # plot keypoint in reference images
                            img_pred_cv2 = cv2.imread(rgb_image_pred_path)
                            keypoints_pred_tuple = tuple(map(tuple, keypoints_in_pred))
                            # keypoints_pred_tuple = tuple(map(int, keypoints_pred_tuple))
                            # for points in keypoints_pred_tuple:
                            #     points = tuple(map(int, points))
                            #     cv2.circle(img_pred_cv2, points, 2, (0, 0, 255), -1)
                            # cv2.imwrite(rgb_image_pred_path, img_pred_cv2)
                            img_gt_cv2 = cv2.imread(ms_image_gt_path)
                            keypoints_gt_tuple = tuple(map(tuple, keypoints_in_gt))
                            # for points in keypoints_gt_tuple:
                            #     points = tuple(map(int, points))
                            #     cv2.circle(img_gt_cv2, points, 2, (0, 0, 255), -1)
                            # cv2.imwrite(ms_image_gt_path, img_gt_cv2)

                            # save pose to buffer
                            initial_ms_pose_r[camera_info.camera_id, :] = vector_rotation.squeeze(1)
                            initial_ms_pose_t[camera_info.camera_id, :] = vector_translation.squeeze(1)

                            # rasterize pose initialized image
                            gaussian_point_cloud_rasterisation_input_initial = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                                point_cloud=self.scene.point_cloud,
                                point_cloud_features=self.scene.point_cloud_features,
                                point_object_id=self.scene.point_object_id,
                                point_invalid_mask=self.scene.point_invalid_mask,
                                camera_info=camera_info_ms,
                                q_pointcloud_camera=q_pointcloud_camera_ms,
                                t_pointcloud_camera=t_pointcloud_camera_ms,
                                color_max_sh_band=3,
                            )
                            rasterized_image_initialized, _, _ = self.rasterisation(
                                gaussian_point_cloud_rasterisation_input_initial,
                                current_train_stage='s1',
                            )
                            image_pred_initial = torch.clamp(rasterized_image_initialized, min=0, max=1)
                            image_pred_initial = image_pred_initial.permute(2, 0, 1)
                            image_pred_name_initial = 'after_BA_initialize_ms_id_' + str(camera_info.camera_id) + '.png'
                            rgb_image_pred_path_initial = os.path.join(self.config.val_image_save_path,
                                                                       image_pred_name_initial)
                            rgb_image_pred_initial = self.toPIL(image_pred_initial)
                            rgb_image_pred_initial.save(rgb_image_pred_path_initial)

                del scene_view_num, initial_count, image_gt, q_pointcloud_camera, t_pointcloud_camera, image_gt_multispectral
                del q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, camera_info, camera_info_ms
                del _, gaussian_point_cloud_rasterisation_input_ms, gaussian_point_cloud_rasterisation_input_initial
                del image_depth, image_gt_name, image_pred, image_pred_initial, image_pred_name, image_pred_name_initial
                del img_gt_cv2, img_pred_cv2, keypoints_in_pred, keypoints_in_gt, keypoints_pred_tuple, keypoints_gt_tuple
                del ms_image_gt, ms_image_gt_path
                del image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, camera_info_ir

                # Initialize IR pose by RGB rasterisation
                # initialize pose buffer(use SO3)
                initial_ir_pose_r = np.zeros([len(cs_train_data_loader), 3])
                initial_ir_pose_t = np.zeros([len(cs_train_data_loader), 3])

                # pose estimate initialization
                with torch.no_grad():
                    scene_view_num = len(cs_train_data_loader)
                    initial_count = []
                    while len(initial_count) < scene_view_num:
                        image_gt, q_pointcloud_camera, t_pointcloud_camera, \
                            image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
                            image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
                            camera_info, camera_info_ms, camera_info_ir = next(cs_train_data_loader_iter)
                        assert camera_info.camera_id == camera_info_ms.camera_id == camera_info_ir.camera_id
                        if str(camera_info.camera_id) in initial_count:
                            pass
                        else:
                            print(f"Initializing IR Image id:{camera_info.camera_id}")
                            initial_count.append(str(camera_info.camera_id))
                            # send data to cuda
                            image_gt = image_gt.cuda()
                            image_gt_infrared = image_gt_infrared.cuda()
                            q_pointcloud_camera = q_pointcloud_camera.cuda()
                            t_pointcloud_camera = t_pointcloud_camera.cuda()
                            # Obviously there is redundancy here, and optimization will be carried out in the future
                            camera_info_ir.camera_intrinsics = camera_info_ir.camera_intrinsics.cuda()
                            camera_info_ir.camera_intrinsics_infrared = camera_info_ir.camera_intrinsics_infrared.cuda()
                            camera_info_ir.camera_width = int(camera_info_ir.camera_width)
                            camera_info_ir.camera_height = int(camera_info_ir.camera_height)
                            camera_info_ir.camera_width_infrared = int(camera_info_ir.camera_width_infrared)
                            camera_info_ir.camera_height_infrared = int(camera_info_ir.camera_height_infrared)
                            # use estimated MS pose to render RGB images
                            gaussian_point_cloud_rasterisation_input_ir = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                                point_cloud=self.scene.point_cloud,
                                point_cloud_features=self.scene.point_cloud_features,
                                point_object_id=self.scene.point_object_id,
                                point_invalid_mask=self.scene.point_invalid_mask,
                                camera_info=camera_info_ir,
                                q_pointcloud_camera=q_pointcloud_camera,
                                t_pointcloud_camera=t_pointcloud_camera,
                                color_max_sh_band=3,
                            )
                            rasterized_image, keypoints_in_pred, keypoints_in_gt, q_pointcloud_camera_ir, t_pointcloud_camera_ir, vector_rotation, vector_translation = \
                                self.rasterisation(gaussian_point_cloud_rasterisation_input_ir,
                                                   current_train_stage='pose_initialize',
                                                   reference_ms_image=image_gt_infrared)
                            # save the images with keypoints
                            image_pred = torch.clamp(rasterized_image, min=0, max=1)
                            image_pred = image_pred.permute(2, 0, 1)
                            image_pred_name = 'before_BA_initialize_ir_id_' + str(camera_info.camera_id) + '.png'
                            rgb_image_pred_path = os.path.join(self.config.val_image_save_path, image_pred_name)
                            rgb_image_pred = self.toPIL(image_pred)
                            rgb_image_pred.save(rgb_image_pred_path)
                            image_gt_name = 'keypoint_ir_gt_id_' + str(camera_info.camera_id) + '.png'
                            ir_image_gt_path = os.path.join(self.config.val_image_save_path, image_gt_name)
                            ir_image_gt = self.toPIL(image_gt_infrared)
                            ir_image_gt.save(ir_image_gt_path)
                            # plot keypoint in reference images
                            img_pred_cv2 = cv2.imread(rgb_image_pred_path)
                            keypoints_pred_tuple = tuple(map(tuple, keypoints_in_pred))
                            # for points in keypoints_pred_tuple:
                            #     points = tuple(map(int, points))
                            #     cv2.circle(img_pred_cv2, points, 3, (255, 0, 0), 3)
                            # cv2.imwrite(rgb_image_pred_path, img_pred_cv2)
                            img_gt_cv2 = cv2.imread(ir_image_gt_path)
                            keypoints_gt_tuple = tuple(map(tuple, keypoints_in_gt))
                            # for points in keypoints_gt_tuple:
                            #     points = tuple(map(int, points))
                            #     cv2.circle(img_gt_cv2, points, 3, (255, 0, 0), 3)
                            # cv2.imwrite(ir_image_gt_path, img_gt_cv2)

                            # save pose to buffer
                            initial_ir_pose_r[camera_info.camera_id, :] = vector_rotation.squeeze(1)
                            initial_ir_pose_t[camera_info.camera_id, :] = vector_translation.squeeze(1)

                            # rasterize pose initialized image
                            gaussian_point_cloud_rasterisation_input_initial = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                                point_cloud=self.scene.point_cloud,
                                point_cloud_features=self.scene.point_cloud_features,
                                point_object_id=self.scene.point_object_id,
                                point_invalid_mask=self.scene.point_invalid_mask,
                                camera_info=camera_info_ir,
                                q_pointcloud_camera=q_pointcloud_camera_ir,
                                t_pointcloud_camera=t_pointcloud_camera_ir,
                                color_max_sh_band=3,
                            )
                            rasterized_image_initialized, _, _ = self.rasterisation(
                                gaussian_point_cloud_rasterisation_input_initial,
                                current_train_stage='s1',
                            )
                            image_pred_initial = torch.clamp(rasterized_image_initialized, min=0, max=1)
                            image_pred_initial = image_pred_initial.permute(2, 0, 1)
                            image_pred_name_initial = 'after_BA_initialize_ir_id_' + str(camera_info.camera_id) + '.png'
                            rgb_image_pred_path_initial = os.path.join(self.config.val_image_save_path,
                                                                       image_pred_name_initial)
                            rgb_image_pred_initial = self.toPIL(image_pred_initial)
                            rgb_image_pred_initial.save(rgb_image_pred_path_initial)

                del scene_view_num, initial_count, image_gt, q_pointcloud_camera, t_pointcloud_camera, image_gt_multispectral
                del q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, camera_info, camera_info_ms
                del _, gaussian_point_cloud_rasterisation_input_ir, gaussian_point_cloud_rasterisation_input_initial
                del image_gt_name, image_pred, image_pred_initial, image_pred_name, image_pred_name_initial
                del img_gt_cv2, img_pred_cv2, keypoints_in_pred, keypoints_in_gt, keypoints_pred_tuple, keypoints_gt_tuple
                del ir_image_gt, ir_image_gt_path

                # initialize pose optimizer
                self.pose_estimated_multispectral = LearnPose(
                    initial_R=torch.tensor(initial_ms_pose_r, device=self.scene.point_cloud.device,
                                           dtype=torch.float32),
                    initial_t=torch.tensor(initial_ms_pose_t, device=self.scene.point_cloud.device,
                                           dtype=torch.float32),
                )
                ms_pose_optimizer = torch.optim.Adam(self.pose_estimated_multispectral.parameters(),
                                                     lr=self.config.pose_learning_rate, betas=(0.9, 0.999))
                scheduler_ms_pose = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=ms_pose_optimizer, gamma=self.config.position_learning_rate_decay_rate)

                # initialize pose optimizer
                self.pose_estimated_infrared = LearnPose(
                    initial_R=torch.tensor(initial_ir_pose_r, device=self.scene.point_cloud.device,
                                           dtype=torch.float32),
                    initial_t=torch.tensor(initial_ir_pose_t, device=self.scene.point_cloud.device,
                                           dtype=torch.float32),
                )
                ir_pose_optimizer = torch.optim.Adam(self.pose_estimated_infrared.parameters(),
                                                     lr=self.config.pose_learning_rate, betas=(0.9, 0.999))
                scheduler_ir_pose = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=ir_pose_optimizer, gamma=self.config.position_learning_rate_decay_rate)

                del initial_ms_pose_r, initial_ms_pose_t
                del initial_ir_pose_r, initial_ir_pose_t
                print('Step2 Part0: Initial Bundle Adjustment Finished!')

                # render initial image
                self.validation_for_cross_spectral(val_data_loader, iteration)
                print('====Fine-tune Initialized Rendering Metrics====')

            # ->Step2 Part1: Using render loss to fine-tune MS camera pose and initialize MS color
            if self.config.warmup_single_modality_iterations < iteration <= self.config.fine_bundle_adjustment_iteration:
                current_modality = next(SH_cycle_pool)
                if current_modality == 'MS':
                    ms_pose_optimizer.zero_grad()
                    ir_pose_optimizer.zero_grad()
                    optimizer.zero_grad()
                    position_optimizer.zero_grad()
                    image_gt, q_pointcloud_camera, t_pointcloud_camera, \
                        image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
                        image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
                        camera_info, camera_info_ms, camera_info_ir = next(cs_train_data_loader_iter)
                    assert camera_info.camera_id == camera_info_ms.camera_id == camera_info_ir.camera_id
                    # get estimated multispectral image pose
                    q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, r_, t_ = \
                        self.pose_estimated_multispectral(camera_info.camera_id)
                    # split ms image to one channel
                    image_gt_multispectral_one_channel = image_gt_multispectral[0, :, :]
                    image_gt_multispectral = image_gt_multispectral_one_channel.unsqueeze(dim=0)

                    # send data to cuda
                    image_gt_multispectral = image_gt_multispectral.cuda()
                    q_pointcloud_camera_multispectral = q_pointcloud_camera_multispectral.cuda()
                    t_pointcloud_camera_multispectral = t_pointcloud_camera_multispectral.cuda()
                    # Obviously there is redundancy here, and optimization will be carried out in the future

                    camera_info_ms.camera_intrinsics = camera_info_ms.camera_intrinsics.cuda()
                    camera_info_ms.camera_intrinsics_multispectral = camera_info_ms.camera_intrinsics_multispectral.cuda()
                    camera_info_ms.camera_width = int(camera_info_ms.camera_width_multispectral)
                    camera_info_ms.camera_height = int(camera_info_ms.camera_height_multispectral)
                    camera_info_ms.camera_width_multispectral = int(camera_info_ms.camera_width_multispectral)
                    camera_info_ms.camera_height_multispectral = int(camera_info_ms.camera_height_multispectral)
                    # use estimated MS pose to render MS images only
                    gaussian_point_cloud_rasterisation_input_ms = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_object_id=self.scene.point_object_id,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        camera_info=camera_info_ms,
                        q_pointcloud_camera=q_pointcloud_camera_multispectral,
                        t_pointcloud_camera=t_pointcloud_camera_multispectral,
                        color_max_sh_band=iteration // self.config.increase_color_max_sh_band_interval,
                    )
                    rasterized_image, rasterized_depth, pixel_valid_point_count = \
                        self.rasterisation(gaussian_point_cloud_rasterisation_input_ms,
                                           current_train_stage='s2p2_ms')
                    image_pred = rasterized_image[:, :, 0]
                    image_depth = rasterized_depth
                    # clip to [0, 1]
                    image_pred = torch.clamp(image_pred, min=0, max=1)
                    # hxwx3->3xhxw
                    image_pred_ms = image_pred.unsqueeze(dim=2)
                    image_pred_ms = torch.clamp(image_pred_ms, min=0, max=1)
                    image_pred_ms = image_pred_ms.permute(2, 0, 1)
                    # image_pred_ms = image_pred_ms.repeat(3, 1, 1)
                    loss, l1_loss, ssim_loss = self.loss_function(
                        image_pred_ms,
                        image_gt_multispectral,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        pointcloud_features=self.scene.point_cloud_features)
                    loss.backward()
                    optimizer.step()
                    # position_optimizer.step()
                    ms_pose_optimizer.step()
                elif current_modality == 'IR':
                    ms_pose_optimizer.zero_grad()
                    ir_pose_optimizer.zero_grad()
                    optimizer.zero_grad()
                    position_optimizer.zero_grad()
                    image_gt, q_pointcloud_camera, t_pointcloud_camera, \
                    image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
                    image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
                    camera_info, camera_info_ms, camera_info_ir = next(cs_train_data_loader_iter)
                    assert camera_info.camera_id == camera_info_ms.camera_id == camera_info_ir.camera_id
                    # get estimated multispectral image pose
                    q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, r_, t_ = \
                        self.pose_estimated_infrared(camera_info.camera_id)
                    # split ms image to one channel
                    image_gt_infrared_one_channel = image_gt_infrared[0, :, :]
                    image_gt_infrared = image_gt_infrared_one_channel.unsqueeze(dim=0)

                    # send data to cuda
                    image_gt_infrared = image_gt_infrared.cuda()
                    q_pointcloud_camera_infrared = q_pointcloud_camera_infrared.cuda()
                    t_pointcloud_camera_infrared = t_pointcloud_camera_infrared.cuda()
                    # Obviously there is redundancy here, and optimization will be carried out in the future

                    camera_info_ir.camera_intrinsics = camera_info_ir.camera_intrinsics.cuda()
                    camera_info_ir.camera_intrinsics_infrared = camera_info_ir.camera_intrinsics_infrared.cuda()
                    camera_info_ir.camera_width = int(camera_info_ms.camera_width_infrared)
                    camera_info_ir.camera_height = int(camera_info_ms.camera_height_infrared)
                    camera_info_ir.camera_width_infrared = int(camera_info_ms.camera_width_infrared)
                    camera_info_ir.camera_height_infrared = int(camera_info_ms.camera_height_infrared)
                    # use estimated MS pose to render MS images only
                    gaussian_point_cloud_rasterisation_input_ir = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_object_id=self.scene.point_object_id,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        camera_info=camera_info_ir,
                        q_pointcloud_camera=q_pointcloud_camera_infrared,
                        t_pointcloud_camera=t_pointcloud_camera_infrared,
                        color_max_sh_band=iteration // self.config.increase_color_max_sh_band_interval,
                    )
                    rasterized_image, rasterized_depth, pixel_valid_point_count = \
                        self.rasterisation(gaussian_point_cloud_rasterisation_input_ir,
                                           current_train_stage='s2p2_ir')
                    image_pred = rasterized_image[:, :, 0]
                    image_depth = rasterized_depth
                    # clip to [0, 1]
                    image_pred = torch.clamp(image_pred, min=0, max=1)
                    # hxwx3->3xhxw
                    image_pred_ir = image_pred.unsqueeze(dim=2)
                    image_pred_ir = torch.clamp(image_pred_ir, min=0, max=1)
                    image_pred_ir = image_pred_ir.permute(2, 0, 1)
                    # image_pred_ms = image_pred_ms.repeat(3, 1, 1)
                    loss, l1_loss, ssim_loss = self.loss_function(
                        image_pred_ir,
                        image_gt_infrared,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        pointcloud_features=self.scene.point_cloud_features)
                    loss.backward()
                    optimizer.step()
                    ir_pose_optimizer.step()
                    # position_optimizer.step()
                    # ms_pose_optimizer.step()

            # ->Step2 Part2: Using render loss to joint optimize MS and RGB pointcloud
            if iteration > self.config.fine_bundle_adjustment_iteration:
                current_modality = next(modality_cycle_pool)
                if current_modality == 'MS':
                    # rendering MS images
                    ms_pose_optimizer.zero_grad()
                    ir_pose_optimizer.zero_grad()
                    optimizer.zero_grad()
                    position_optimizer.zero_grad()
                    image_gt, q_pointcloud_camera, t_pointcloud_camera, \
                        image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
                        image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
                        camera_info, camera_info_ms, camera_info_ir = next(cs_train_data_loader_iter)
                    assert camera_info.camera_id == camera_info_ms.camera_id == camera_info_ir.camera_id
                    # get estimated multispectral image pose
                    q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, r_, t_ = \
                        self.pose_estimated_multispectral(camera_info.camera_id)
                    # split ms image to one channel
                    image_gt_multispectral_one_channel = image_gt_multispectral[0, :, :]
                    image_gt_multispectral = image_gt_multispectral_one_channel.unsqueeze(dim=0)

                    # send data to cuda
                    image_gt_multispectral = image_gt_multispectral.cuda()
                    q_pointcloud_camera_multispectral = q_pointcloud_camera_multispectral.cuda()
                    t_pointcloud_camera_multispectral = t_pointcloud_camera_multispectral.cuda()
                    # Obviously there is redundancy here, and optimization will be carried out in the future

                    camera_info_ms.camera_intrinsics = camera_info_ms.camera_intrinsics.cuda()
                    camera_info_ms.camera_intrinsics_multispectral = camera_info_ms.camera_intrinsics_multispectral.cuda()
                    camera_info_ms.camera_width = int(camera_info_ms.camera_width_multispectral)
                    camera_info_ms.camera_height = int(camera_info_ms.camera_height_multispectral)
                    camera_info_ms.camera_width_multispectral = int(camera_info_ms.camera_width_multispectral)
                    camera_info_ms.camera_height_multispectral = int(camera_info_ms.camera_height_multispectral)
                    # use estimated MS pose to render MS images only
                    gaussian_point_cloud_rasterisation_input_ms = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_object_id=self.scene.point_object_id,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        camera_info=camera_info_ms,
                        q_pointcloud_camera=q_pointcloud_camera_multispectral,
                        t_pointcloud_camera=t_pointcloud_camera_multispectral,
                        color_max_sh_band=(iteration - self.config.fine_bundle_adjustment_iteration) // self.config.increase_color_max_sh_band_interval,
                    )
                    rasterized_image, rasterized_depth, pixel_valid_point_count = \
                        self.rasterisation(gaussian_point_cloud_rasterisation_input_ms,
                                           current_train_stage='s2p3_ms')
                    image_pred = rasterized_image[:, :, 0]
                    image_depth = rasterized_depth
                    # clip to [0, 1]
                    image_pred = torch.clamp(image_pred, min=0, max=1)
                    # hxwx3->3xhxw
                    image_pred_ms = image_pred.unsqueeze(dim=2)
                    image_pred_ms = torch.clamp(image_pred_ms, min=0, max=1)
                    image_pred_ms = image_pred_ms.permute(2, 0, 1)
                    # image_pred_ms = image_pred_ms.repeat(3, 1, 1)
                    loss, l1_loss, ssim_loss = self.loss_function(
                        image_pred_ms,
                        image_gt_multispectral,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        pointcloud_features=self.scene.point_cloud_features)
                    loss.backward()
                    optimizer.step()
                    position_optimizer.step()
                    ms_pose_optimizer.step()
                    # torch.cuda.synchronize()
                    self.adaptive_controller.refinement()
                elif current_modality == 'IR':
                    # rendering IR images
                    ms_pose_optimizer.zero_grad()
                    ir_pose_optimizer.zero_grad()
                    optimizer.zero_grad()
                    position_optimizer.zero_grad()
                    image_gt, q_pointcloud_camera, t_pointcloud_camera, \
                        image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
                        image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
                        camera_info, camera_info_ms, camera_info_ir = next(cs_train_data_loader_iter)
                    assert camera_info.camera_id == camera_info_ms.camera_id == camera_info_ir.camera_id
                    # get estimated multispectral image pose
                    q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, r_, t_ = \
                        self.pose_estimated_infrared(camera_info.camera_id)
                    # split ms image to one channel
                    image_gt_infrared_one_channel = image_gt_infrared[0, :, :]
                    image_gt_infrared = image_gt_infrared_one_channel.unsqueeze(dim=0)

                    # send data to cuda
                    image_gt_infrared = image_gt_infrared.cuda()
                    q_pointcloud_camera_infrared = q_pointcloud_camera_infrared.cuda()
                    t_pointcloud_camera_infrared = t_pointcloud_camera_infrared.cuda()
                    # Obviously there is redundancy here, and optimization will be carried out in the future

                    camera_info_ir.camera_intrinsics = camera_info_ir.camera_intrinsics.cuda()
                    camera_info_ir.camera_intrinsics_infrared = camera_info_ir.camera_intrinsics_infrared.cuda()
                    camera_info_ir.camera_width = int(camera_info_ir.camera_width_infrared)
                    camera_info_ir.camera_height = int(camera_info_ir.camera_height_infrared)
                    camera_info_ir.camera_width_infrared = int(camera_info_ir.camera_width_infrared)
                    camera_info_ir.camera_height_infrared = int(camera_info_ir.camera_height_infrared)
                    # use estimated MS pose to render MS images only
                    gaussian_point_cloud_rasterisation_input_ir = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_object_id=self.scene.point_object_id,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        camera_info=camera_info_ir,
                        q_pointcloud_camera=q_pointcloud_camera_infrared,
                        t_pointcloud_camera=t_pointcloud_camera_infrared,
                        color_max_sh_band=(iteration - self.config.fine_bundle_adjustment_iteration) // self.config.increase_color_max_sh_band_interval,
                    )
                    rasterized_image, rasterized_depth, pixel_valid_point_count = \
                        self.rasterisation(gaussian_point_cloud_rasterisation_input_ir,
                                           current_train_stage='s2p3_ir')
                    image_pred = rasterized_image[:, :, 0]
                    image_depth = rasterized_depth
                    # clip to [0, 1]
                    image_pred = torch.clamp(image_pred, min=0, max=1)
                    # hxwx3->3xhxw
                    image_pred_ir = image_pred.unsqueeze(dim=2)
                    image_pred_ir = torch.clamp(image_pred_ir, min=0, max=1)
                    image_pred_ir = image_pred_ir.permute(2, 0, 1)
                    # image_pred_ms = image_pred_ms.repeat(3, 1, 1)
                    loss, l1_loss, ssim_loss = self.loss_function(
                        image_pred_ir,
                        image_gt_infrared,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        pointcloud_features=self.scene.point_cloud_features)
                    loss.backward()
                    optimizer.step()
                    position_optimizer.step()
                    ir_pose_optimizer.step()
                    # ms_pose_optimizer.step()
                    # torch.cuda.synchronize()
                    self.adaptive_controller.refinement()
                elif current_modality == 'RGB':
                    # send data to dataloader
                    image_gt, q_pointcloud_camera, t_pointcloud_camera, \
                        image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
                        image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
                        camera_info, camera_info_ms, camera_info_ir = next(train_data_loader_iter)
                    # send data to cuda
                    image_gt = image_gt.cuda()
                    q_pointcloud_camera = q_pointcloud_camera.cuda()
                    t_pointcloud_camera = t_pointcloud_camera.cuda()
                    camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
                    camera_info.camera_width = int(camera_info.camera_width)
                    camera_info.camera_height = int(camera_info.camera_height)
                    camera_info.camera_intrinsics_multispectral = camera_info.camera_intrinsics_multispectral.cuda()
                    camera_info.camera_width = int(camera_info.camera_width)
                    camera_info.camera_height = int(camera_info.camera_height)
                    camera_info.camera_width_multispectral = int(camera_info.camera_width_multispectral)
                    camera_info.camera_height_multispectral = int(camera_info.camera_height_multispectral)

                    # Rasterisation RGB modality
                    ms_pose_optimizer.zero_grad()
                    ir_pose_optimizer.zero_grad()
                    optimizer.zero_grad()
                    position_optimizer.zero_grad()
                    gaussian_point_cloud_rasterisation_input = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_object_id=self.scene.point_object_id,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        camera_info=camera_info,
                        q_pointcloud_camera=q_pointcloud_camera,
                        t_pointcloud_camera=t_pointcloud_camera,
                        color_max_sh_band=(
                                                      iteration - self.config.fine_bundle_adjustment_iteration) // self.config.increase_color_max_sh_band_interval,
                    )
                    rasterized_image, rasterized_depth, pixel_valid_point_count = self.rasterisation(
                        gaussian_point_cloud_rasterisation_input,
                        current_train_stage='s1',
                    )
                    image_pred = rasterized_image
                    image_depth = rasterized_depth
                    # clip to [0, 1]
                    image_pred = torch.clamp(image_pred, min=0, max=1)
                    # hxwx3->3xhxw
                    image_pred = image_pred.permute(2, 0, 1)

                    loss, l1_loss, ssim_loss = self.loss_function(
                        image_pred,
                        image_gt,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        pointcloud_features=self.scene.point_cloud_features)
                    loss.backward()
                    optimizer.step()
                    position_optimizer.step()
                    recent_losses.append(loss.item())
                    # adaptive controller refine
                    self.adaptive_controller.refinement()

            # schedular learning rate by exponent
            if iteration > self.config.fine_bundle_adjustment_iteration and iteration % self.config.position_learning_rate_decay_interval == 0:
                scheduler.step()
            if iteration > self.config.fine_bundle_adjustment_iteration and iteration % self.config.pose_learning_rate_decay_interval == 0:
                scheduler_ms_pose.step()
                scheduler_ir_pose.step()

            # print taichi kernel profiler to console
            if self.config.enable_taichi_kernel_profiler and iteration % self.config.log_taichi_kernel_profile_interval == 0 and iteration > 0:
                ti.profiler.print_kernel_profiler_info("count")
                ti.profiler.clear_kernel_profiler_info()

            # write loss information to writer and console
            if iteration <= self.config.warmup_single_modality_iterations and iteration % self.config.log_loss_interval == 0:
                self.writer.add_scalar("train/loss", loss.item(), iteration)
                self.writer.add_scalar("train/l1 loss", l1_loss.item(), iteration)
                self.writer.add_scalar("train/ssim loss", ssim_loss.item(), iteration)
                if self.config.print_metrics_to_console:
                    print(f"train_loss={loss.item()};",
                          f"train_l1_loss={l1_loss.item()};",
                          f"train_ssim_loss={ssim_loss.item()};")

            # write loss information to writer and console
            if iteration > self.config.warmup_single_modality_iterations and iteration % self.config.log_loss_interval == 0:
                self.writer.add_scalar("train/loss", loss.item(), iteration)
                self.writer.add_scalar("train/l1 loss", l1_loss.item(), iteration)
                self.writer.add_scalar("train/ssim loss", ssim_loss.item(), iteration)
                if self.config.print_metrics_to_console:
                    print(f"train_loss={loss.item()};",
                          f"train_l1_loss={l1_loss.item()};",
                          f"train_ssim_loss={ssim_loss.item()};")

            # write training metrics to writer and console
            if iteration < self.config.warmup_single_modality_iterations and iteration % self.config.log_metrics_interval == 0:
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(image_pred=image_pred, image_gt=image_gt)
                self.writer.add_scalar("train/psnr", psnr_score.item(), iteration)
                self.writer.add_scalar("train/ssim", ssim_score.item(), iteration)
                if self.config.print_metrics_to_console:
                    print('<RGB pre-train metrics>',
                          f"train_psnr_rgb_{iteration}=%.4f;" % psnr_score.item(),
                          f"train_ssim_rgb_{iteration}=%.4f;" % ssim_score.item())

            if self.config.warmup_single_modality_iterations < iteration <= self.config.fine_bundle_adjustment_iteration \
                    and iteration % self.config.log_metrics_interval == 0 and current_modality == 'MS':
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(image_pred=image_pred_ms,
                                                                     image_gt=image_gt_multispectral)
                self.writer.add_scalar("train/psnr", psnr_score.item(), iteration)
                self.writer.add_scalar("train/ssim", ssim_score.item(), iteration)
                if self.config.print_metrics_to_console:
                    print('<MS color optimize metrics in joint optimization>',
                          f"train_psnr_ms_{iteration}=%.4f;" % psnr_score.item(),
                          f"train_ssim_ms_{iteration}=%.4f;" % ssim_score.item())
            elif self.config.warmup_single_modality_iterations < iteration <= self.config.fine_bundle_adjustment_iteration \
                    and iteration % self.config.log_metrics_interval == 0 and current_modality == 'IR':
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(image_pred=image_pred_ir,
                                                                     image_gt=image_gt_infrared)
                self.writer.add_scalar("train/psnr", psnr_score.item(), iteration)
                self.writer.add_scalar("train/ssim", ssim_score.item(), iteration)
                if self.config.print_metrics_to_console:
                    print('<MS color optimize metrics in joint optimization>',
                          f"train_psnr_ir_{iteration}=%.4f;" % psnr_score.item(),
                          f"train_ssim_ir_{iteration}=%.4f;" % ssim_score.item())

            if iteration > self.config.fine_bundle_adjustment_iteration and current_modality == 'RGB' and iteration % self.config.log_metrics_interval == 0:
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(image_pred=image_pred, image_gt=image_gt)
                self.writer.add_scalar("train/psnr", psnr_score.item(), iteration)
                self.writer.add_scalar("train/ssim", ssim_score.item(), iteration)
                if self.config.print_metrics_to_console:
                    print('<RGB color optimize metrics in joint optimization>',
                          f"train_psnr_rgb_{iteration}=%.4f;" % psnr_score.item(),
                          f"train_ssim_rgb_{iteration}=%.4f;" % ssim_score.item())
            elif iteration > self.config.fine_bundle_adjustment_iteration and current_modality == 'MS' and iteration % self.config.log_metrics_interval == 0:
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(image_pred=image_pred_ms,
                                                                     image_gt=image_gt_multispectral)
                self.writer.add_scalar("train/psnr", psnr_score.item(), iteration)
                self.writer.add_scalar("train/ssim", ssim_score.item(), iteration)
                if self.config.print_metrics_to_console:
                    print('<MS color optimize metrics in joint optimization>',
                          f"train_psnr_ms_{iteration}=%.4f;" % psnr_score.item(),
                          f"train_ssim_ms_{iteration}=%.4f;" % ssim_score.item())
            elif iteration > self.config.fine_bundle_adjustment_iteration and current_modality == 'IR' and iteration % self.config.log_metrics_interval == 0:
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(image_pred=image_pred_ir,
                                                                     image_gt=image_gt_infrared)
                self.writer.add_scalar("train/psnr", psnr_score.item(), iteration)
                self.writer.add_scalar("train/ssim", ssim_score.item(), iteration)
                if self.config.print_metrics_to_console:
                    print('<IR color optimize metrics in joint optimization>',
                          f"train_psnr_ir_{iteration}=%.4f;" % psnr_score.item(),
                          f"train_ssim_ir_{iteration}=%.4f;" % ssim_score.item())

            if iteration % self.config.val_interval == 0 and iteration != 0 and iteration <= self.config.warmup_single_modality_iterations:  # they use 7000 in paper, it's hard to set a interval so hard code it here
                self.validation(val_data_loader, iteration)

            if self.config.warmup_single_modality_iterations < iteration <= self.config.fine_bundle_adjustment_iteration and iteration % 1000 == 0:
                self.store_current_pose_list(iteration, modal='ms', num_cameras=len(cs_train_data_loader))
                self.store_current_pose_list(iteration, modal='ir', num_cameras=len(cs_train_data_loader))
                self.validation_for_cross_spectral(val_data_loader, iteration)
            elif iteration > self.config.fine_bundle_adjustment_iteration and iteration % self.config.val_interval == 0:
                self.store_current_pose_list(iteration, modal='ms', num_cameras=len(cs_train_data_loader))
                self.store_current_pose_list(iteration, modal='ir', num_cameras=len(cs_train_data_loader))
                self.validation_for_cross_spectral(val_data_loader, iteration)
            #
            # if iteration > self.config.fine_bundle_adjustment_iteration and iteration % self.config.val_interval == 0:
            #     # self.store_current_pose_list(iteration, modal='ms', num_cameras=len(cs_train_data_loader))
            #     # self.store_current_pose_list(iteration, modal='ir', num_cameras=len(cs_train_data_loader))
            #     self.validation_for_cross_spectral(val_data_loader, iteration)

            if iteration < self.config.warmup_single_modality_iterations:
                del image_gt, q_pointcloud_camera, t_pointcloud_camera, camera_info, camera_info_ms, camera_info_ir
                del gaussian_point_cloud_rasterisation_input, image_pred
                del loss, l1_loss, ssim_loss
            elif self.config.warmup_single_modality_iterations < iteration <= self.config.fine_bundle_adjustment_iteration and current_modality == 'MS':
                del image_gt, image_gt_multispectral, q_pointcloud_camera, q_pointcloud_camera_multispectral, t_pointcloud_camera, t_pointcloud_camera_multispectral
                del gaussian_point_cloud_rasterisation_input_ms, image_pred, loss, l1_loss, ssim_loss
                del camera_info, camera_info_ms, camera_info_ir
            elif self.config.warmup_single_modality_iterations < iteration <= self.config.fine_bundle_adjustment_iteration and current_modality == 'IR':
                del image_gt, image_gt_multispectral, q_pointcloud_camera, q_pointcloud_camera_multispectral, t_pointcloud_camera, t_pointcloud_camera_multispectral
                del gaussian_point_cloud_rasterisation_input_ir, image_pred, loss, l1_loss, ssim_loss
                del camera_info, camera_info_ir
                del image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared
            elif iteration > self.config.fine_bundle_adjustment_iteration and current_modality == 'RGB':
                del image_gt, image_gt_multispectral, q_pointcloud_camera, q_pointcloud_camera_multispectral, t_pointcloud_camera, t_pointcloud_camera_multispectral
                del gaussian_point_cloud_rasterisation_input, image_pred, loss, l1_loss, ssim_loss
                del camera_info, camera_info_ms
            elif iteration > self.config.fine_bundle_adjustment_iteration and current_modality == 'MS':
                del image_gt, image_gt_multispectral, q_pointcloud_camera, q_pointcloud_camera_multispectral, t_pointcloud_camera, t_pointcloud_camera_multispectral
                del gaussian_point_cloud_rasterisation_input_ms, image_pred, loss, l1_loss, ssim_loss
                del camera_info, camera_info_ms, rasterized_image, rasterized_depth, pixel_valid_point_count
            elif iteration > self.config.fine_bundle_adjustment_iteration and current_modality == 'MS':
                del image_gt, image_gt_multispectral, q_pointcloud_camera, q_pointcloud_camera_multispectral, t_pointcloud_camera, t_pointcloud_camera_multispectral
                del gaussian_point_cloud_rasterisation_input_ms, image_pred, loss, l1_loss, ssim_loss
                del camera_info, camera_info_ms, rasterized_image, rasterized_depth, pixel_valid_point_count
                del image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared

        self.validation_for_cross_spectral(val_data_loader, self.config.num_iterations)

        for iteration_ending in tqdm(range(self.config.ending_iterations)):
            optimizer.zero_grad()
            position_optimizer.zero_grad()
            ms_pose_optimizer.zero_grad()
            ir_pose_optimizer.zero_grad()
            # send data to dataloader
            image_gt, q_pointcloud_camera, t_pointcloud_camera, \
                image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
                image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
                camera_info, camera_info_ms, camera_info_ir = next(train_data_loader_iter)
            assert camera_info.camera_id == camera_info_ms.camera_id == camera_info_ir.camera_id

            # send data to cuda
            image_gt = image_gt.cuda()
            q_pointcloud_camera = q_pointcloud_camera.cuda()
            t_pointcloud_camera = t_pointcloud_camera.cuda()
            camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
            camera_info.camera_width = int(camera_info.camera_width)
            camera_info.camera_height = int(camera_info.camera_height)

            # Rasterisation RGB modality
            ms_pose_optimizer.zero_grad()
            ir_pose_optimizer.zero_grad()
            optimizer.zero_grad()
            position_optimizer.zero_grad()
            gaussian_point_cloud_rasterisation_input = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                point_cloud=self.scene.point_cloud,
                point_cloud_features=self.scene.point_cloud_features,
                point_object_id=self.scene.point_object_id,
                point_invalid_mask=self.scene.point_invalid_mask,
                camera_info=camera_info,
                q_pointcloud_camera=q_pointcloud_camera,
                t_pointcloud_camera=t_pointcloud_camera,
                color_max_sh_band=3,
            )
            rasterized_image, rasterized_depth, pixel_valid_point_count = self.rasterisation(
                gaussian_point_cloud_rasterisation_input,
                current_train_stage='ending_rgb',
            )
            image_pred = rasterized_image
            image_depth = rasterized_depth
            # clip to [0, 1]
            image_pred = torch.clamp(image_pred, min=0, max=1)
            # hxwx3->3xhxw
            image_pred = image_pred.permute(2, 0, 1)

            loss, l1_loss, ssim_loss = self.loss_function(
                image_pred,
                image_gt,
                point_invalid_mask=self.scene.point_invalid_mask,
                pointcloud_features=self.scene.point_cloud_features)
            loss.backward()
            optimizer.step()
            del image_gt, q_pointcloud_camera, t_pointcloud_camera, camera_info, gaussian_point_cloud_rasterisation_input, image_pred, loss, l1_loss, ssim_loss

        for iteration_ending in tqdm(range(self.config.ending_iterations)):
            # end MS training
            ms_pose_optimizer.zero_grad()
            ir_pose_optimizer.zero_grad()
            optimizer.zero_grad()
            position_optimizer.zero_grad()
            image_gt, q_pointcloud_camera, t_pointcloud_camera, \
            image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
            image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
            camera_info, camera_info_ms, camera_info_ir = next(cs_train_data_loader_iter)
            assert camera_info.camera_id == camera_info_ms.camera_id == camera_info_ir.camera_id
            # get estimated multispectral image pose
            q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, r_, t_ = \
                self.pose_estimated_multispectral(camera_info.camera_id)
            # split ms image to one channel
            image_gt_multispectral_one_channel = image_gt_multispectral[0, :, :]
            image_gt_multispectral = image_gt_multispectral_one_channel.unsqueeze(dim=0)

            # send data to cuda
            image_gt_multispectral = image_gt_multispectral.cuda()
            q_pointcloud_camera_multispectral = q_pointcloud_camera_multispectral.cuda()
            t_pointcloud_camera_multispectral = t_pointcloud_camera_multispectral.cuda()
            # Obviously there is redundancy here, and optimization will be carried out in the future

            camera_info_ms.camera_intrinsics = camera_info_ms.camera_intrinsics.cuda()
            camera_info_ms.camera_intrinsics_multispectral = camera_info_ms.camera_intrinsics_multispectral.cuda()
            camera_info_ms.camera_width = int(camera_info_ms.camera_width_multispectral)
            camera_info_ms.camera_height = int(camera_info_ms.camera_height_multispectral)
            camera_info_ms.camera_width_multispectral = int(camera_info_ms.camera_width_multispectral)
            camera_info_ms.camera_height_multispectral = int(camera_info_ms.camera_height_multispectral)
            # use estimated MS pose to render MS images only
            gaussian_point_cloud_rasterisation_input_ms = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                point_cloud=self.scene.point_cloud,
                point_cloud_features=self.scene.point_cloud_features,
                point_object_id=self.scene.point_object_id,
                point_invalid_mask=self.scene.point_invalid_mask,
                camera_info=camera_info_ms,
                q_pointcloud_camera=q_pointcloud_camera_multispectral,
                t_pointcloud_camera=t_pointcloud_camera_multispectral,
                color_max_sh_band=3,
            )
            rasterized_image, rasterized_depth, pixel_valid_point_count = \
                self.rasterisation(gaussian_point_cloud_rasterisation_input_ms,
                                   current_train_stage='ending_ms')
            image_pred = rasterized_image[:, :, 0]
            image_depth = rasterized_depth
            # clip to [0, 1]
            image_pred = torch.clamp(image_pred, min=0, max=1)
            # hxwx3->3xhxw
            image_pred_ms = image_pred.unsqueeze(dim=2)
            image_pred_ms = torch.clamp(image_pred_ms, min=0, max=1)
            image_pred_ms = image_pred_ms.permute(2, 0, 1)
            # image_pred_ms = image_pred_ms.repeat(3, 1, 1)
            loss, l1_loss, ssim_loss = self.loss_function(
                image_pred_ms,
                image_gt_multispectral,
                point_invalid_mask=self.scene.point_invalid_mask,
                pointcloud_features=self.scene.point_cloud_features)
            loss.backward()
            # just optimize MS color-SH
            optimizer.step()

        for iteration_ending in tqdm(range(self.config.ending_iterations)):
            # rendering IR images
            ms_pose_optimizer.zero_grad()
            ir_pose_optimizer.zero_grad()
            optimizer.zero_grad()
            position_optimizer.zero_grad()
            image_gt, q_pointcloud_camera, t_pointcloud_camera, \
            image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
            image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
            camera_info, camera_info_ms, camera_info_ir = next(cs_train_data_loader_iter)
            assert camera_info.camera_id == camera_info_ms.camera_id == camera_info_ir.camera_id
            # get estimated multispectral image pose
            q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, r_, t_ = \
                self.pose_estimated_infrared(camera_info.camera_id)
            # split ms image to one channel
            image_gt_infrared_one_channel = image_gt_infrared[0, :, :]
            image_gt_infrared = image_gt_infrared_one_channel.unsqueeze(dim=0)

            # send data to cuda
            image_gt_infrared = image_gt_infrared.cuda()
            q_pointcloud_camera_infrared = q_pointcloud_camera_infrared.cuda()
            t_pointcloud_camera_infrared = t_pointcloud_camera_infrared.cuda()
            # Obviously there is redundancy here, and optimization will be carried out in the future

            camera_info_ir.camera_intrinsics = camera_info_ir.camera_intrinsics.cuda()
            camera_info_ir.camera_intrinsics_infrared = camera_info_ir.camera_intrinsics_infrared.cuda()
            camera_info_ir.camera_width = int(camera_info_ir.camera_width_infrared)
            camera_info_ir.camera_height = int(camera_info_ir.camera_height_infrared)
            camera_info_ir.camera_width_infrared = int(camera_info_ir.camera_width_infrared)
            camera_info_ir.camera_height_infrared = int(camera_info_ir.camera_height_infrared)
            # use estimated MS pose to render MS images only
            gaussian_point_cloud_rasterisation_input_ir = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                point_cloud=self.scene.point_cloud,
                point_cloud_features=self.scene.point_cloud_features,
                point_object_id=self.scene.point_object_id,
                point_invalid_mask=self.scene.point_invalid_mask,
                camera_info=camera_info_ir,
                q_pointcloud_camera=q_pointcloud_camera_infrared,
                t_pointcloud_camera=t_pointcloud_camera_infrared,
                color_max_sh_band=3,
            )
            rasterized_image, rasterized_depth, pixel_valid_point_count = \
                self.rasterisation(gaussian_point_cloud_rasterisation_input_ir,
                                   current_train_stage='ending_ir')
            image_pred = rasterized_image[:, :, 0]
            image_depth = rasterized_depth
            # clip to [0, 1]
            image_pred = torch.clamp(image_pred, min=0, max=1)
            # hxwx3->3xhxw
            image_pred_ir = image_pred.unsqueeze(dim=2)
            image_pred_ir = torch.clamp(image_pred_ir, min=0, max=1)
            image_pred_ir = image_pred_ir.permute(2, 0, 1)
            # image_pred_ms = image_pred_ms.repeat(3, 1, 1)
            loss, l1_loss, ssim_loss = self.loss_function(
                image_pred_ir,
                image_gt_infrared,
                point_invalid_mask=self.scene.point_invalid_mask,
                pointcloud_features=self.scene.point_cloud_features)
            loss.backward()
            # only optimize IR color
            optimizer.step()

        self.store_current_pose_list(self.config.num_iterations + 3*self.config.ending_iterations, modal='ms',
                                     num_cameras=len(cs_train_data_loader))
        self.store_current_pose_list(self.config.num_iterations + 3*self.config.ending_iterations, modal='ir',
                                     num_cameras=len(cs_train_data_loader))
        self.validation_for_cross_spectral(val_data_loader, self.config.num_iterations + 3*self.config.ending_iterations)
        self.scene.to_parquet(os.path.join(self.config.output_model_dir, f"final_scene.parquet"))

    @staticmethod
    def _compute_pnsr_and_ssim(image_pred, image_gt):
        with torch.no_grad():
            psnr_score = 10 * \
                         torch.log10(1.0 / torch.mean((image_pred - image_gt) ** 2))
            ssim_score = ssim(image_pred.unsqueeze(0), image_gt.unsqueeze(
                0), data_range=1.0, size_average=True)
            return psnr_score, ssim_score

    @staticmethod
    def validation_for_cross_spectral(self, val_data_loader, iteration):
        os.makedirs(self.config.val_image_save_path, exist_ok=True)
        with torch.no_grad():
            total_loss_RGB = 0.0
            total_loss_MS = 0.0
            total_loss_IR = 0.0
            total_psnr_score_RGB = 0.0
            total_ssim_score_RGB = 0.0
            total_psnr_score_MS = 0.0
            total_ssim_score_MS = 0.0
            total_psnr_score_IR = 0.0
            total_ssim_score_IR = 0.0
            if self.config.enable_taichi_kernel_profiler:
                ti.profiler.print_kernel_profiler_info("count")
                ti.profiler.clear_kernel_profiler_info()
            total_inference_time = 0.0
            for idx, val_data in enumerate(tqdm(val_data_loader)):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                image_gt, q_pointcloud_camera, t_pointcloud_camera, \
                    image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
                    image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
                    camera_info, camera_info_ms, camera_info_ir = val_data
                assert camera_info.camera_id == camera_info_ms.camera_id == camera_info_ir.camera_id
                # get estimated multispectral image pose
                q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, r_, t_ = \
                    self.pose_estimated_multispectral(camera_info.camera_id)
                # get estimated infrared image pose
                q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, r_, t_ = \
                    self.pose_estimated_infrared(camera_info.camera_id)

                # send data to cuda
                image_gt = image_gt.cuda()
                image_gt_multispectral = image_gt_multispectral.cuda()
                image_gt_infrared = image_gt_infrared.cuda()
                q_pointcloud_camera = q_pointcloud_camera.cuda()
                t_pointcloud_camera = t_pointcloud_camera.cuda()
                q_pointcloud_camera_multispectral = q_pointcloud_camera_multispectral.cuda()
                t_pointcloud_camera_multispectral = t_pointcloud_camera_multispectral.cuda()
                q_pointcloud_camera_infrared = q_pointcloud_camera_infrared.cuda()
                t_pointcloud_camera_infrared = t_pointcloud_camera_infrared.cuda()
                camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
                camera_info.camera_intrinsics_multispectral = camera_info.camera_intrinsics_multispectral.cuda()
                camera_info.camera_intrinsics_infrared = camera_info.camera_intrinsics_multispectral.cuda()

                # make taichi happy
                camera_info.camera_width = int(camera_info.camera_width)
                camera_info.camera_height = int(camera_info.camera_height)
                camera_info.camera_width_multispectral = int(camera_info.camera_width_multispectral)
                camera_info.camera_height_multispectral = int(camera_info.camera_height_multispectral)
                # set RGB rasterisation options
                gaussian_point_cloud_rasterisation_input = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                    point_cloud=self.scene.point_cloud,
                    point_cloud_features=self.scene.point_cloud_features,
                    point_object_id=self.scene.point_object_id,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    camera_info=camera_info,
                    q_pointcloud_camera=q_pointcloud_camera,
                    t_pointcloud_camera=t_pointcloud_camera,
                    color_max_sh_band=3,
                )
                # set MS rasterisation options
                gaussian_point_cloud_rasterisation_input_ms = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                    point_cloud=self.scene.point_cloud,
                    point_cloud_features=self.scene.point_cloud_features,
                    point_object_id=self.scene.point_object_id,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    camera_info=camera_info_ms,
                    q_pointcloud_camera=q_pointcloud_camera_multispectral,
                    t_pointcloud_camera=t_pointcloud_camera_multispectral,
                    color_max_sh_band=3
                )
                # set IR rasterisation options
                gaussian_point_cloud_rasterisation_input_ir = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                    point_cloud=self.scene.point_cloud,
                    point_cloud_features=self.scene.point_cloud_features,
                    point_object_id=self.scene.point_object_id,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    camera_info=camera_info_ir,
                    q_pointcloud_camera=q_pointcloud_camera_infrared,
                    t_pointcloud_camera=t_pointcloud_camera_infrared,
                    color_max_sh_band=3
                )

                # render validation RGB image
                start_event.record()
                rasterized_image, rasterized_depth, pixel_valid_point_count = \
                    self.rasterisation(gaussian_point_cloud_rasterisation_input,
                                       current_train_stage='val_RGB', )
                end_event.record()
                torch.cuda.synchronize()
                time_taken = start_event.elapsed_time(end_event)
                total_inference_time += time_taken
                image_pred = rasterized_image
                # clip to [0, 1]
                image_pred = torch.clamp(image_pred, min=0, max=1)
                # hxwx3->3xhxw
                image_pred = image_pred.permute(2, 0, 1)
                # calculate loss and metrics
                loss_RGB, _, _ = self.loss_function(image_pred, image_gt)
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(image_pred=image_pred, image_gt=image_gt)
                total_loss_RGB += loss_RGB.item()
                total_psnr_score_RGB += psnr_score.item()
                total_ssim_score_RGB += ssim_score.item()

                # render validation MS image
                start_event.record()
                rasterized_image_ms, rasterized_depth_ms, pixel_valid_point_count_ms = \
                    self.rasterisation(gaussian_point_cloud_rasterisation_input_ms,
                                       current_train_stage='val_MS')
                end_event.record()
                torch.cuda.synchronize()
                time_taken = start_event.elapsed_time(end_event)
                total_inference_time += time_taken
                image_pred_ms = rasterized_image_ms[:, :, -1].unsqueeze(dim=2)
                image_pred_ms = torch.clamp(image_pred_ms, min=0, max=1)
                image_pred_ms = image_pred_ms.permute(2, 0, 1)
                image_pred_ms = image_pred_ms.repeat(3, 1, 1)
                loss_MS, _, _ = self.loss_function(image_pred_ms, image_gt_multispectral)
                psnr_score_ms, ssim_score_ms = self._compute_pnsr_and_ssim(image_pred=image_pred_ms,
                                                                           image_gt=image_gt_multispectral)
                total_loss_MS += loss_MS.item()
                total_psnr_score_MS += psnr_score_ms.item()
                total_ssim_score_MS += ssim_score_ms.item()

                # render validation IR image
                start_event.record()
                rasterized_image_ir, rasterized_depth_ir, pixel_valid_point_count_ir = \
                    self.rasterisation(gaussian_point_cloud_rasterisation_input_ir,
                                       current_train_stage='val_IR')
                end_event.record()
                torch.cuda.synchronize()
                time_taken = start_event.elapsed_time(end_event)
                total_inference_time += time_taken
                image_pred_ir = rasterized_image_ir[:, :, -1].unsqueeze(dim=2)
                image_pred_ir = torch.clamp(image_pred_ir, min=0, max=1)
                image_pred_ir = image_pred_ir.permute(2, 0, 1)
                image_pred_ir = image_pred_ir.repeat(3, 1, 1)
                loss_IR, _, _ = self.loss_function(image_pred_ir, image_gt_infrared)
                psnr_score_ir, ssim_score_ir = self._compute_pnsr_and_ssim(image_pred=image_pred_ir,
                                                                           image_gt=image_gt_infrared)
                total_loss_IR += loss_IR.item()
                total_psnr_score_IR += psnr_score_ir.item()
                total_ssim_score_IR += ssim_score_ir.item()

                rgb_image_pred_name = \
                    'rgb_image_pred_' + 'iter_' + str(iteration) + '_id_' + str(camera_info.camera_id) + '.jpg'
                rgb_image_gt_name = \
                    'rgb_image_gt_' + '_id_' + str(camera_info.camera_id) + '.jpg'
                ms_image_pred_name = \
                    'ms_image_pred_' + 'iter_' + str(iteration) + '_id_' + str(camera_info.camera_id) + '.jpg'
                ms_image_gt_name = \
                    'ms_image_gt_' + '_id_' + str(camera_info.camera_id) + '.jpg'
                ir_image_pred_name = \
                    'ir_image_pred_' + 'iter_' + str(iteration) + '_id_' + str(camera_info.camera_id) + '.jpg'
                ir_image_gt_name = \
                    'ir_image_gt_' + '_id_' + str(camera_info.camera_id) + '.jpg'

                rgb_image_pred_path = os.path.join(self.config.val_image_save_path, rgb_image_pred_name)
                rgb_image_gt_path = os.path.join(self.config.val_image_save_path, rgb_image_gt_name)
                ms_image_pred_path = os.path.join(self.config.val_image_save_path, ms_image_pred_name)
                ms_image_gt_path = os.path.join(self.config.val_image_save_path, ms_image_gt_name)
                ir_image_pred_path = os.path.join(self.config.val_image_save_path, ir_image_pred_name)
                ir_image_gt_path = os.path.join(self.config.val_image_save_path, ir_image_gt_name)

                rgb_image_pred = self.toPIL(image_pred)
                rgb_image_pred.save(rgb_image_pred_path)
                rgb_image_gt = self.toPIL(image_gt)
                rgb_image_gt.save(rgb_image_gt_path)
                ms_image_pred = self.toPIL(image_pred_ms)
                ms_image_pred.save(ms_image_pred_path)
                ms_image_gt = self.toPIL(image_gt_multispectral)
                ms_image_gt.save(ms_image_gt_path)
                ir_image_pred = self.toPIL(image_pred_ir)
                ir_image_pred.save(ir_image_pred_path)
                ir_image_gt = self.toPIL(image_gt_infrared)
                ir_image_gt.save(ir_image_gt_path)

            # print taichi kernel profiler
            if self.config.enable_taichi_kernel_profiler:
                ti.profiler.print_kernel_profiler_info("count")
                ti.profiler.clear_kernel_profiler_info()

            # calculate inference time(RGB+MS), actual inference time should be halve.
            average_inference_time = total_inference_time / len(val_data_loader)
            # calculate mean metrics for RGB and MS render
            mean_loss_RGB = total_loss_RGB / len(val_data_loader)
            mean_loss_MS = total_loss_MS / len(val_data_loader)
            mean_loss_IR = total_loss_IR / len(val_data_loader)

            mean_psnr_score_RGB = total_psnr_score_RGB / len(val_data_loader)
            mean_ssim_score_RGB = total_ssim_score_RGB / len(val_data_loader)
            mean_psnr_score_MS = total_psnr_score_MS / len(val_data_loader)
            mean_ssim_score_MS = total_ssim_score_MS / len(val_data_loader)
            mean_psnr_score_IR = total_psnr_score_IR / len(val_data_loader)
            mean_ssim_score_IR = total_ssim_score_IR / len(val_data_loader)

            # add information to writer
            self.writer.add_scalar("val/loss", mean_loss_RGB, iteration)
            self.writer.add_scalar("val/psnr", mean_psnr_score_RGB, iteration)
            self.writer.add_scalar("val/ssim", mean_ssim_score_RGB, iteration)
            self.writer.add_scalar("val/loss_MS", mean_loss_MS, iteration)
            self.writer.add_scalar("val/psnr_MS", mean_psnr_score_MS, iteration)
            self.writer.add_scalar("val/ssim_MS", mean_ssim_score_MS, iteration)
            self.writer.add_scalar("val/loss_IR", mean_loss_IR, iteration)
            self.writer.add_scalar("val/psnr_IR", mean_psnr_score_IR, iteration)
            self.writer.add_scalar("val/ssim_IR", mean_ssim_score_IR, iteration)
            self.writer.add_scalar("val/inference_time", average_inference_time, iteration)

            # print loss and metrics to console
            if self.config.print_metrics_to_console:
                print(f"val_loss_RGB={mean_loss_RGB};", f"val_loss_MS={mean_loss_MS};")
                print(f"val_psnr_RGB_{iteration}={mean_psnr_score_RGB};",
                      f"val_ssim_RGB_{iteration}={mean_ssim_score_RGB};")
                print(f"val_psnr_MS_{iteration}={mean_psnr_score_MS};",
                      f"val_ssim_MS_{iteration}={mean_ssim_score_MS};")
                print(f"val_psnr_IR_{iteration}={mean_psnr_score_IR};",
                      f"val_ssim_IR_{iteration}={mean_ssim_score_IR};")
                print(f"val_inference_time={average_inference_time};")

            # save 3DGS to parquet
            self.scene.to_parquet(
                os.path.join(self.config.output_model_dir, f"scene_{iteration}.parquet"))
            if mean_psnr_score_RGB > self.best_psnr_score:
                self.best_psnr_score = mean_psnr_score_RGB
                self.best_ssim_score = mean_ssim_score_RGB
                print('Best Metric RGB', self.best_psnr_score, self.best_ssim_score)
                print('Best Metric MS', self.best_psnr_score_ms, self.best_ssim_score_ms)
                print('Best Metric IR', self.best_psnr_score_ir, self.best_ssim_score_ir)
            if mean_psnr_score_MS > self.best_psnr_score_ms:
                self.best_psnr_score_ms = mean_psnr_score_MS
                self.best_ssim_score_ms = mean_ssim_score_MS
                print('Best Metric RGB', self.best_psnr_score, self.best_ssim_score)
                print('Best Metric MS', self.best_psnr_score_ms, self.best_ssim_score_ms)
                print('Best Metric IR', self.best_psnr_score_ir, self.best_ssim_score_ir)
            if mean_psnr_score_IR > self.best_psnr_score_ir:
                self.best_psnr_score_ir = mean_psnr_score_IR
                self.best_ssim_score_ir = mean_ssim_score_IR
                print('Best Metric RGB', self.best_psnr_score, self.best_ssim_score)
                print('Best Metric MS', self.best_psnr_score_ms, self.best_ssim_score_ms)
                print('Best Metric IR', self.best_psnr_score_ir, self.best_ssim_score_ir)
            if mean_psnr_score_RGB + mean_psnr_score_MS + mean_psnr_score_RGB > self.best_psnr_score_store_parquet:
                self.best_psnr_score_store_parquet = mean_psnr_score_RGB + mean_psnr_score_MS + mean_psnr_score_RGB
                self.scene.to_parquet(os.path.join(self.config.output_model_dir, f"best_scene.parquet"))
                print(f"<Best Parquet saved!> Metric RGB:{mean_psnr_score_RGB},"
                      f"Metric MS:{mean_psnr_score_MS}, Metric IR:{mean_psnr_score_IR}")
        del total_inference_time, total_loss_RGB, total_loss_MS, total_psnr_score_RGB, total_psnr_score_MS, total_ssim_score_RGB, total_ssim_score_MS
        del image_gt, q_pointcloud_camera, t_pointcloud_camera, image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, camera_info, camera_info_ms
        del gaussian_point_cloud_rasterisation_input, gaussian_point_cloud_rasterisation_input_ms
        del rasterized_image, rasterized_image_ms, rasterized_depth, rasterized_depth_ms, pixel_valid_point_count, pixel_valid_point_count_ms
        del loss_RGB, loss_MS, rgb_image_gt, rgb_image_pred, ms_image_gt, ms_image_pred
        del total_loss_IR, total_ssim_score_IR, image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared
        del gaussian_point_cloud_rasterisation_input_ir, rasterized_image_ir, rasterized_depth_ir, pixel_valid_point_count_ir
        del loss_IR, ir_image_gt, ir_image_pred

    def store_current_pose_list(self, iteration, modal, num_cameras):
        q_list = []
        t_list = []
        with torch.no_grad():
            for camera_id in range(num_cameras):
                if modal == 'ms':
                    q_pointcloud_camera_ms, t_pointcloud_camera_ms, q_vec, t_vec = \
                        self.pose_estimated_multispectral(camera_id)
                elif modal == 'ir':
                    q_pointcloud_camera_ms, t_pointcloud_camera_ms, q_vec, t_vec = \
                        self.pose_estimated_infrared(camera_id)
                q_numpy = q_pointcloud_camera_ms.detach().cpu().numpy()
                t_numpy = t_pointcloud_camera_ms.detach().cpu().numpy()
                q_list.append(q_numpy)
                t_list.append(t_numpy)
            q_list_np = np.array(q_list)
            t_list_np = np.array(t_list)
            np.save(os.path.join(self.config.summary_writer_log_dir,
                                 modal + '_iter_' + str(iteration).zfill(6)) + '_r_', q_list_np)
            np.save(os.path.join(self.config.summary_writer_log_dir,
                                 modal + '_iter_' + str(iteration).zfill(6)) + '_t_', t_list_np)
        return

    def validation(self, val_data_loader, iteration):
        os.makedirs(self.config.val_image_save_path, exist_ok=True)
        with torch.no_grad():
            total_loss_RGB = 0.0
            total_psnr_score_RGB = 0.0
            total_ssim_score_RGB = 0.0
            if self.config.enable_taichi_kernel_profiler:
                ti.profiler.print_kernel_profiler_info("count")
                ti.profiler.clear_kernel_profiler_info()
            total_inference_time = 0.0
            for idx, val_data in enumerate(tqdm(val_data_loader)):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                image_gt, q_pointcloud_camera, t_pointcloud_camera, \
                image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
                _, _, _, camera_info, camera_info_ms, _ = val_data
                # send data to cuda
                image_gt = image_gt.cuda()
                q_pointcloud_camera = q_pointcloud_camera.cuda()
                t_pointcloud_camera = t_pointcloud_camera.cuda()
                camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
                # make taichi happy
                camera_info.camera_width = int(camera_info.camera_width)
                camera_info.camera_height = int(camera_info.camera_height)
                camera_info.camera_width_multispectral = int(camera_info.camera_width_multispectral)
                camera_info.camera_height_multispectral = int(camera_info.camera_height_multispectral)
                # set RGB rasterisation options
                gaussian_point_cloud_rasterisation_input = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                    point_cloud=self.scene.point_cloud,
                    point_cloud_features=self.scene.point_cloud_features,
                    point_object_id=self.scene.point_object_id,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    camera_info=camera_info,
                    q_pointcloud_camera=q_pointcloud_camera,
                    t_pointcloud_camera=t_pointcloud_camera,
                    color_max_sh_band=3,
                )
                # render validation RGB image
                start_event.record()
                rasterized_image, rasterized_depth, pixel_valid_point_count = \
                    self.rasterisation(gaussian_point_cloud_rasterisation_input,
                                       current_train_stage='val_RGB', )
                end_event.record()
                torch.cuda.synchronize()
                time_taken = start_event.elapsed_time(end_event)
                total_inference_time += time_taken
                image_pred = rasterized_image
                # clip to [0, 1]
                image_pred = torch.clamp(image_pred, min=0, max=1)
                # hxwx3->3xhxw
                image_pred = image_pred.permute(2, 0, 1)
                # calculate loss and metrics
                loss_RGB, _, _ = self.loss_function(image_pred, image_gt)
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(image_pred=image_pred, image_gt=image_gt)
                total_loss_RGB += loss_RGB.item()
                total_psnr_score_RGB += psnr_score.item()
                total_ssim_score_RGB += ssim_score.item()
                rgb_image_pred_name = 'rgb_image_pred_' + 'iter_' + str(iteration) + '_id_' + str(
                    camera_info.camera_id) + '.jpg'
                rgb_image_gt_name = 'rgb_image_gt_' + '_id_' + str(camera_info.camera_id) + '.jpg'
                rgb_image_pred_path = os.path.join(self.config.val_image_save_path, rgb_image_pred_name)
                rgb_image_gt_path = os.path.join(self.config.val_image_save_path, rgb_image_gt_name)
                rgb_image_pred = self.toPIL(image_pred)
                rgb_image_pred.save(rgb_image_pred_path)
                rgb_image_gt = self.toPIL(image_gt)
                rgb_image_gt.save(rgb_image_gt_path)

            # print taichi kernel profiler
            if self.config.enable_taichi_kernel_profiler:
                ti.profiler.print_kernel_profiler_info("count")
                ti.profiler.clear_kernel_profiler_info()

            # calculate inference time(RGB+MS), actual inference time should be halve.
            average_inference_time = total_inference_time / len(val_data_loader)
            # calculate mean metrics for RGB and MS render
            mean_loss_RGB = total_loss_RGB / len(val_data_loader)
            mean_psnr_score_RGB = total_psnr_score_RGB / len(val_data_loader)
            mean_ssim_score_RGB = total_ssim_score_RGB / len(val_data_loader)
            # add information to writer
            self.writer.add_scalar("val/loss", mean_loss_RGB, iteration)
            self.writer.add_scalar("val/psnr", mean_psnr_score_RGB, iteration)
            self.writer.add_scalar("val/ssim", mean_ssim_score_RGB, iteration)
            self.writer.add_scalar("val/inference_time", average_inference_time, iteration)

            # print loss and metrics to console
            if self.config.print_metrics_to_console:
                print(f"val_psnr_RGB_{iteration}={mean_psnr_score_RGB};",
                      f"val_ssim_RGB_{iteration}={mean_ssim_score_RGB};")
                print(f"val_inference_time={average_inference_time};")

            # save 3DGS to parquet
            self.scene.to_parquet(
                os.path.join(self.config.output_model_dir, f"scene_{iteration}.parquet"))
            if mean_psnr_score_RGB > self.best_psnr_score:
                self.best_psnr_score = mean_psnr_score_RGB
                self.best_ssim_score = mean_ssim_score_RGB
                self.scene.to_parquet(os.path.join(self.config.output_model_dir, f"best_scene.parquet"))
                print('Best Metric RGB', self.best_psnr_score, self.best_ssim_score)
                # print('Best Metric MS', self.best_psnr_score_ms, self.best_ssim_score_ms)
        del total_inference_time, total_loss_RGB, total_psnr_score_RGB, total_ssim_score_RGB,
        del image_gt, q_pointcloud_camera, t_pointcloud_camera, image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, camera_info, camera_info_ms
        del gaussian_point_cloud_rasterisation_input
        del rasterized_image, rasterized_depth, pixel_valid_point_count,
        del loss_RGB, rgb_image_gt, rgb_image_pred, _
