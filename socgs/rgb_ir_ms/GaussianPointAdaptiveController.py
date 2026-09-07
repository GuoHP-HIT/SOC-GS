import numpy as np
import torch
from dataclasses import dataclass
from .GaussianPointCloudRasterisation import GaussianPointCloudRasterisation, \
    load_point_cloud_row_into_gaussian_point_3d
from dataclass_wizard import YAMLWizard
from typing import Optional
import taichi as ti
import matplotlib.pyplot as plt


@ti.kernel
def compute_ellipsoid_offset(
        pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
        pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, 56)
        ellipsoid_offset: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
):
    for idx in range(pointcloud.shape[0]):
        point = load_point_cloud_row_into_gaussian_point_3d(
            pointcloud=pointcloud,
            pointcloud_features=pointcloud_features,
            point_id=idx,
        )
        foci_vector = point.get_ellipsoid_foci_vector()
        ellipsoid_offset[idx, 0] = foci_vector[0]
        ellipsoid_offset[idx, 1] = foci_vector[1]
        ellipsoid_offset[idx, 2] = foci_vector[2]


@ti.kernel
def sample_from_point(
        pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
        pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, 56)
        sample_result: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
):
    for idx in range(pointcloud.shape[0]):
        point = load_point_cloud_row_into_gaussian_point_3d(
            pointcloud=pointcloud,
            pointcloud_features=pointcloud_features,
            point_id=idx,
        )
        foci_vector = point.sample()
        sample_result[idx, 0] = foci_vector[0]
        sample_result[idx, 1] = foci_vector[1]
        sample_result[idx, 2] = foci_vector[2]


class GaussianPointAdaptiveController:
    """
    For simplicity, I set the size of point cloud to be fixed during training. an extra mask is used to indicate whether a point is invalid or not.
    When initialising, the input point cloud is concatenated with extra points filled with zero. The mask is concatenated with extra True.
    When densifying and splitting points, new points are assigned to locations of invalid points.
    When removing points, we just set the mask to True.
    """

    @dataclass
    class GaussianPointAdaptiveControllerConfig(YAMLWizard):
        num_iterations_warm_up: int = 500
        num_iterations_densify: int = 100
        warmup_single_modality_iterations: int = 5000
        # from paper: densify every 100 iterations and remove any Gaussians that are essentially transparent, i.e., with 𝛼 less than a threshold 𝜖𝛼.
        transparent_alpha_threshold: float = -0.5
        # from paper: densify Gaussians with an average magnitude of view-space position gradients above a threshold 𝜏pos, which we set to 0.0002 in our tests.
        # I have no idea why their threshold is so low, may be their view space is normalized to [0, 1]?
        # TODO: find out a proper threshold
        densification_view_space_position_gradients_threshold: float = 6e-6
        densification_view_avg_space_position_gradients_threshold: float = 1e3
        densification_multi_frame_view_space_position_gradients_threshold: float = 1e3
        densification_multi_frame_view_pixel_avg_space_position_gradients_threshold: float = 1e3
        densification_multi_frame_position_gradients_threshold: float = 1e3
        # from paper:  large Gaussians in regions with high variance need to be split into smaller Gaussians. We replace such Gaussians by two new ones, and divide their scale by a factor of 𝜙 = 1.6
        gaussian_split_factor_phi: float = 1.6
        # in paper section 5.2, they describe a method to moderate the increase in the number of Gaussians is to set the 𝛼 value close to zero every
        # 3000 iterations. I have no idea how it is implemented. I just assume that it is a reset of 𝛼 to fixed value.
        num_iterations_reset_alpha: int = 3000
        reset_alpha_value: float = 0.1
        # the paper doesn't mention this value, but we need a value and method to determine whether a point is under-reconstructed or over-reconstructed
        # for now, the method is to threshold norm of exp(s)
        # TODO: find out a proper threshold
        floater_num_pixels_threshold: int = 10000
        floater_near_camrea_num_pixels_threshold: int = 10000
        floater_depth_threshold: float = 100
        iteration_start_remove_floater: int = 2000
        plot_densify_interval: int = 200
        under_reconstructed_num_pixels_threshold: int = 512
        under_reconstructed_move_factor: float = 100.0
        enable_ellipsoid_offset: bool = False
        enable_sample_from_point: bool = True

    @dataclass
    class GaussianPointAdaptiveControllerMaintainedParameters:
        pointcloud: torch.Tensor  # shape: [num_points, 3]
        # shape: [num_points, num_features], num_features is 56
        pointcloud_features: torch.Tensor
        # shape: [num_points], dtype: int8 because taichi doesn't support bool type
        point_invalid_mask: torch.Tensor
        point_object_id: torch.Tensor  # shape: [num_points]

    @dataclass
    class GaussianPointAdaptiveControllerDensifyPointInfo:
        floater_point_id: torch.Tensor  # shape: [num_floater_points]
        transparent_point_id: torch.Tensor  # shape: [num_transparent_points]
        densify_point_id: torch.Tensor  # shape: [num_points_to_densify]
        densify_point_position_before_optimization: torch.Tensor  # shape: [num_points_to_densify, 3]
        densify_size_reduction_factor: torch.Tensor  # shape: [num_points_to_densify]
        densify_point_grad_position: torch.Tensor  # shape: [num_points_to_densify, 3]

    def __init__(self,
                 config: GaussianPointAdaptiveControllerConfig,
                 maintained_parameters: GaussianPointAdaptiveControllerMaintainedParameters):
        # Iteration counters for RGB and MS optimize
        # MS counter works after warmup_single_modality_iterations and fine_bundle_adjustment_iteration
        self.iteration_counter = -1
        self.iteration_counter_ms = 0
        self.iteration_counter_ir = 0

        # ================>>>>public parameters(densify point info will be replaced by multi-modality version)
        self.config = config
        self.maintained_parameters = maintained_parameters
        self.input_data = None
        self.input_data_ms = None
        self.input_data_ir = None
        self.densify_point_info: Optional[
            GaussianPointAdaptiveController.GaussianPointAdaptiveControllerDensifyPointInfo] = None

        # accumulated parameter for RGB rasterisation
        self.accumulated_num_pixels = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
        self.accumulated_num_in_camera = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
        self.accumulated_view_space_position_gradients = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
        self.accumulated_view_space_position_gradients_avg = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
        self.accumulated_position_gradients = \
            torch.zeros_like(self.maintained_parameters.pointcloud, dtype=torch.float32)
        self.accumulated_position_gradients_norm = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)

        # accumulated parameter for MS rasterisation
        self.accumulated_num_pixels_ms = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
        self.accumulated_num_in_camera_ms = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
        self.accumulated_view_space_position_gradients_ms = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
        self.accumulated_view_space_position_gradients_avg_ms = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
        self.accumulated_position_gradients_ms = \
            torch.zeros_like(self.maintained_parameters.pointcloud, dtype=torch.float32)
        self.accumulated_position_gradients_norm_ms = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)

        # accumulated parameter for IR rasterisation
        self.accumulated_num_pixels_ir = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
        self.accumulated_num_in_camera_ir = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
        self.accumulated_view_space_position_gradients_ir = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
        self.accumulated_view_space_position_gradients_avg_ir = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
        self.accumulated_position_gradients_ir = \
            torch.zeros_like(self.maintained_parameters.pointcloud, dtype=torch.float32)
        self.accumulated_position_gradients_norm_ir = \
            torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)

    def update(self, input_data: GaussianPointCloudRasterisation.BackwardValidPointHookInput):
        if input_data.hook_modality == 'RGB':
            self.iteration_counter += 1
            with torch.no_grad():
                self.accumulated_num_in_camera[input_data.point_id_in_camera_list] += 1
                self.accumulated_num_pixels[input_data.point_id_in_camera_list] += input_data.num_affected_pixels
                grad_viewspace_norm = input_data.magnitude_grad_viewspace
                self.accumulated_view_space_position_gradients[input_data.point_id_in_camera_list] \
                    += grad_viewspace_norm
                avg_grad_viewspace_norm = grad_viewspace_norm / input_data.num_affected_pixels
                avg_grad_viewspace_norm[torch.isnan(avg_grad_viewspace_norm)] = 0
                self.accumulated_view_space_position_gradients_avg[input_data.point_id_in_camera_list] \
                    += avg_grad_viewspace_norm
                self.accumulated_position_gradients[input_data.point_id_in_camera_list] \
                    += input_data.grad_point_in_camera
                self.accumulated_position_gradients_norm[input_data.point_id_in_camera_list] \
                    += input_data.grad_point_in_camera.norm(dim=1)
        elif input_data.hook_modality == 'MS':
            self.iteration_counter_ms += 1
            with torch.no_grad():
                self.accumulated_num_in_camera_ms[input_data.point_id_in_camera_list] += 1
                self.accumulated_num_pixels_ms[input_data.point_id_in_camera_list] += input_data.num_affected_pixels
                grad_viewspace_norm_ms = input_data.magnitude_grad_viewspace
                self.accumulated_view_space_position_gradients_ms[input_data.point_id_in_camera_list] \
                    += grad_viewspace_norm_ms
                avg_grad_viewspace_norm_ms = grad_viewspace_norm_ms / input_data.num_affected_pixels
                avg_grad_viewspace_norm_ms[torch.isnan(avg_grad_viewspace_norm_ms)] = 0
                self.accumulated_view_space_position_gradients_avg_ms[input_data.point_id_in_camera_list] \
                    += avg_grad_viewspace_norm_ms
                self.accumulated_position_gradients_ms[input_data.point_id_in_camera_list] \
                    += input_data.grad_point_in_camera
                self.accumulated_position_gradients_norm_ms[input_data.point_id_in_camera_list] \
                    += input_data.grad_point_in_camera.norm(dim=1)
        elif input_data.hook_modality == 'IR':
            self.iteration_counter_ir += 1
            with torch.no_grad():
                self.accumulated_num_in_camera_ir[input_data.point_id_in_camera_list] += 1
                self.accumulated_num_pixels_ir[input_data.point_id_in_camera_list] += input_data.num_affected_pixels
                grad_viewspace_norm_ir = input_data.magnitude_grad_viewspace
                self.accumulated_view_space_position_gradients_ir[input_data.point_id_in_camera_list] \
                    += grad_viewspace_norm_ir
                avg_grad_viewspace_norm_ir = grad_viewspace_norm_ir / input_data.num_affected_pixels
                avg_grad_viewspace_norm_ir[torch.isnan(avg_grad_viewspace_norm_ir)] = 0
                self.accumulated_view_space_position_gradients_avg_ir[input_data.point_id_in_camera_list] \
                    += avg_grad_viewspace_norm_ir
                self.accumulated_position_gradients_ir[input_data.point_id_in_camera_list] \
                    += input_data.grad_point_in_camera
                self.accumulated_position_gradients_norm_ir[input_data.point_id_in_camera_list] \
                    += input_data.grad_point_in_camera.norm(dim=1)

        # print(f"iteration_counter_RGB:{self.iteration_counter}")
        # print(f"iteration_counter_IR:{self.iteration_counter_ir}")
        # print(f"iteration_counter_MS:{self.iteration_counter_ms}")
        if self.iteration_counter < self.config.num_iterations_warm_up:
            pass
        elif self.iteration_counter <= self.config.warmup_single_modality_iterations and self.iteration_counter % self.config.num_iterations_densify == 0:
            self._find_densify_points(input_data)
            self.input_data = input_data
        elif self.iteration_counter >= self.config.warmup_single_modality_iterations and self.iteration_counter % self.config.num_iterations_densify == 0:
            if self.iteration_counter_ms == 0 or self.iteration_counter_ir == 0:
                pass
            elif self.iteration_counter_ms % self.config.num_iterations_densify != 0 and self.iteration_counter_ir % self.config.num_iterations_densify != 0:
                self._find_densify_points(input_data)
                self.input_data = input_data
            elif self.iteration_counter_ms % self.config.num_iterations_densify == 0 and self.iteration_counter_ir % self.config.num_iterations_densify != 0:
                self._find_densify_points_ms(input_data)
                self.input_data_ms = input_data
            elif self.iteration_counter_ms % self.config.num_iterations_densify == 0 and self.iteration_counter_ir % self.config.num_iterations_densify == 0:
                self._find_densify_points_ir(input_data)
                self.input_data_ir = input_data

    def refinement(self):
        with torch.no_grad():
            if self.iteration_counter < self.config.num_iterations_warm_up:
                return
            if self.iteration_counter <= self.config.warmup_single_modality_iterations and self.iteration_counter % self.config.num_iterations_densify == 0:
                self._add_densify_points()
                self.accumulated_num_in_camera = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
                self.accumulated_num_pixels = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
                self.accumulated_view_space_position_gradients = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
                self.accumulated_view_space_position_gradients_avg = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
                self.accumulated_position_gradients = \
                    torch.zeros_like(self.maintained_parameters.pointcloud, dtype=torch.float32)
                self.accumulated_position_gradients_norm = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
            if self.iteration_counter > self.config.warmup_single_modality_iterations \
                    and self.iteration_counter % self.config.num_iterations_densify == 0 \
                    and self.iteration_counter_ms % self.config.num_iterations_densify == 0 \
                    and self.iteration_counter_ir % self.config.num_iterations_densify == 0:
                self._add_densify_points_cross_spectral()
                self.accumulated_num_in_camera = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
                self.accumulated_num_pixels = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
                self.accumulated_view_space_position_gradients = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
                self.accumulated_view_space_position_gradients_avg = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
                self.accumulated_position_gradients = \
                    torch.zeros_like(self.maintained_parameters.pointcloud, dtype=torch.float32)
                self.accumulated_position_gradients_norm = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)

                self.accumulated_num_in_camera_ms = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
                self.accumulated_num_pixels_ms = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
                self.accumulated_view_space_position_gradients_ms = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
                self.accumulated_view_space_position_gradients_avg_ms = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
                self.accumulated_position_gradients_ms = \
                    torch.zeros_like(self.maintained_parameters.pointcloud, dtype=torch.float32)
                self.accumulated_position_gradients_norm_ms = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)

                self.accumulated_num_in_camera_ir = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
                self.accumulated_num_pixels_ir = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
                self.accumulated_view_space_position_gradients_ir = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
                self.accumulated_view_space_position_gradients_avg_ir = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)
                self.accumulated_position_gradients_ir = \
                    torch.zeros_like(self.maintained_parameters.pointcloud, dtype=torch.float32)
                self.accumulated_position_gradients_norm_ir = \
                    torch.zeros_like(self.maintained_parameters.pointcloud[:, 0], dtype=torch.float32)

            if self.iteration_counter == self.config.num_iterations_reset_alpha:
                self.reset_alpha()
            self.input_data = None
            self.input_data_ms = None
            self.input_data_ir = None

    def _find_densify_points(self, input_data: GaussianPointCloudRasterisation.BackwardValidPointHookInput):
        """ find points to densify, it should happened in backward pass before optimiser step.
        so that the original point values are recorded, and when a point is cloned/split, the
        two points are not the same.

        Args:
            input_data (GaussianPointCloudRasterisation.BackwardValidPointHookInput): input
        """
        pointcloud = self.maintained_parameters.pointcloud
        pointcloud_features = self.maintained_parameters.pointcloud_features
        point_id_list = torch.arange(pointcloud.shape[0], device=pointcloud.device)
        point_id_in_camera_list: torch.Tensor = input_data.point_id_in_camera_list
        num_affected_pixels: torch.Tensor = input_data.num_affected_pixels
        point_depth_in_camera: torch.Tensor = input_data.point_depth
        point_uv_in_camera: torch.Tensor = input_data.point_uv_in_camera
        average_num_affect_pixels = self.accumulated_num_pixels / self.accumulated_num_in_camera
        average_num_affect_pixels[torch.isnan(average_num_affect_pixels)] = 0

        # Note that transparent points are apply on all valid points
        # while floater and densification only apply on points in camera in the current frame
        floater_mask = torch.zeros_like(point_id_list, dtype=torch.bool)
        floater_mask_in_camera = torch.zeros_like(point_id_in_camera_list, dtype=torch.bool)

        floater_point_id = torch.empty(0, dtype=torch.int32, device=pointcloud.device)
        if self.iteration_counter > self.config.iteration_start_remove_floater:
            floater_mask_in_camera = ((num_affected_pixels > self.config.floater_near_camrea_num_pixels_threshold) & \
                                      (point_depth_in_camera < self.config.floater_depth_threshold))

            # floater_mask_in_camera = (num_affected_pixels > self.config.floater_num_pixels_threshold)
            floater_point_id = point_id_in_camera_list[floater_mask_in_camera]
            # floater_mask = average_num_affect_pixels > self.config.floater_num_pixels_threshold
            floater_mask[floater_point_id] = True
            floater_mask = floater_mask & (self.maintained_parameters.point_invalid_mask == 0)

        point_alpha = pointcloud_features[:, 7]  # alpha before sigmoid
        nan_mask = torch.isnan(pointcloud_features).any(dim=1)
        transparent_point_mask = ((point_alpha < self.config.transparent_alpha_threshold) | nan_mask) & \
                                 (self.maintained_parameters.point_invalid_mask == 0) & \
                                 (~floater_mask)  # ensure floater points and transparent points don't overlap
        transparent_point_id = point_id_list[transparent_point_mask]
        will_be_remove_mask = floater_mask | transparent_point_mask

        # find points that are under-reconstructed or over-reconstructed
        # point_features_in_camera = pointcloud_features[point_id_in_camera_list]
        in_camera_will_be_remove_mask = floater_mask_in_camera | transparent_point_mask[point_id_in_camera_list]
        # shape: [num_points_in_camera, 2]
        grad_viewspace_norm = input_data.magnitude_grad_viewspace
        # shape: [num_points_in_camera, num_features]
        # all these three masks are on num_points_in_camera, not num_points
        in_camera_to_densify_mask = (
                grad_viewspace_norm > self.config.densification_view_space_position_gradients_threshold)
        in_camera_to_densify_mask &= (~in_camera_will_be_remove_mask)  # don't densify floater or transparent points
        num_to_densify_by_viewspace = in_camera_to_densify_mask.sum().item()
        in_camera_to_densify_mask |= (
                grad_viewspace_norm / num_affected_pixels > self.config.densification_view_avg_space_position_gradients_threshold)
        in_camera_to_densify_mask &= (~in_camera_will_be_remove_mask)  # don't densify floater or transparent points
        num_to_densify = in_camera_to_densify_mask.sum().item()
        num_to_densify_by_viewspace_avg = num_to_densify - num_to_densify_by_viewspace
        print(
            f"num_to_densify: {num_to_densify}, num_to_densify_by_viewspace: {num_to_densify_by_viewspace}, num_to_densify_by_viewspace_avg: {num_to_densify_by_viewspace_avg}")

        single_frame_densify_point_id = point_id_in_camera_list[in_camera_to_densify_mask]
        single_frame_densify_point_mask = torch.zeros_like(point_id_list, dtype=torch.bool)
        single_frame_densify_point_mask[single_frame_densify_point_id] = True

        multi_frame_average_accumulated_view_space_position_gradients = self.accumulated_view_space_position_gradients / self.accumulated_num_in_camera
        # fill in nan with 0
        multi_frame_average_accumulated_view_space_position_gradients[
            torch.isnan(multi_frame_average_accumulated_view_space_position_gradients)] = 0
        multi_frame_densify_mask = multi_frame_average_accumulated_view_space_position_gradients > self.config.densification_multi_frame_view_space_position_gradients_threshold

        multi_frame_average_accumulated_avg_pixel_view_space_position_gradients = self.accumulated_view_space_position_gradients_avg / self.accumulated_num_in_camera
        # fill in nan with 0
        multi_frame_average_accumulated_avg_pixel_view_space_position_gradients[
            torch.isnan(multi_frame_average_accumulated_avg_pixel_view_space_position_gradients)] = 0
        multi_frame_densify_mask |= (
                multi_frame_average_accumulated_avg_pixel_view_space_position_gradients / average_num_affect_pixels > self.config.densification_multi_frame_view_pixel_avg_space_position_gradients_threshold)
        multi_frame_average_position_gradients_norm = self.accumulated_position_gradients_norm / self.accumulated_num_in_camera
        multi_frame_densify_mask |= (
                multi_frame_average_position_gradients_norm > self.config.densification_multi_frame_position_gradients_threshold)
        to_densify_mask = (single_frame_densify_point_mask | multi_frame_densify_mask) & (~will_be_remove_mask)
        num_merged_densify = to_densify_mask.sum().item()
        print(f"num_merged_densify_with_multi_frame: {num_merged_densify}")
        densify_point_id = point_id_list[to_densify_mask]

        densify_point_position_before_optimization = pointcloud[densify_point_id]
        densify_point_grad_position = self.accumulated_position_gradients[densify_point_id] / \
                                      self.accumulated_num_in_camera[densify_point_id].unsqueeze(-1)
        # although acummulated_num_in_camera shall not be 0, but we still need to check it/fill in nan
        densify_point_grad_position[torch.isnan(densify_point_grad_position)] = 0
        densify_size_reduction_factor = torch.zeros_like(densify_point_id, dtype=torch.float32,
                                                         device=pointcloud.device)
        over_reconstructed_mask = (
                self.accumulated_num_pixels[to_densify_mask] > self.config.under_reconstructed_num_pixels_threshold)
        densify_size_reduction_factor[over_reconstructed_mask] = \
            np.log(self.config.gaussian_split_factor_phi)
        densify_size_reduction_factor = densify_size_reduction_factor.unsqueeze(-1)
        self.densify_point_info = GaussianPointAdaptiveController.GaussianPointAdaptiveControllerDensifyPointInfo(
            floater_point_id=floater_point_id,
            transparent_point_id=transparent_point_id,
            densify_point_id=densify_point_id,
            densify_point_position_before_optimization=densify_point_position_before_optimization,
            densify_size_reduction_factor=densify_size_reduction_factor,
            densify_point_grad_position=densify_point_grad_position,
        )

    def _find_densify_points_ir(self, input_data: GaussianPointCloudRasterisation.BackwardValidPointHookInput):
        """ for additional modality adaptive controller, let the densify points invisible in main modality.

        Args:
            input_data (GaussianPointCloudRasterisation.BackwardValidPointHookInput): input
        """
        pointcloud = self.maintained_parameters.pointcloud
        pointcloud_features = self.maintained_parameters.pointcloud_features
        point_id_list = torch.arange(pointcloud.shape[0], device=pointcloud.device)
        point_id_in_camera_list: torch.Tensor = input_data.point_id_in_camera_list
        num_affected_pixels: torch.Tensor = input_data.num_affected_pixels
        point_depth_in_camera: torch.Tensor = input_data.point_depth
        point_uv_in_camera: torch.Tensor = input_data.point_uv_in_camera
        average_num_affect_pixels = self.accumulated_num_pixels_ir / self.accumulated_num_in_camera_ir
        average_num_affect_pixels[torch.isnan(average_num_affect_pixels)] = 0

        # Note that transparent points are apply on all valid points
        # while floater and densification only apply on points in camera in the current frame
        floater_mask = torch.zeros_like(point_id_list, dtype=torch.bool)
        floater_mask_in_camera = torch.zeros_like(point_id_in_camera_list, dtype=torch.bool)

        floater_point_id = torch.empty(0, dtype=torch.int32, device=pointcloud.device)
        if self.iteration_counter > self.config.iteration_start_remove_floater:
            floater_mask_in_camera = ((num_affected_pixels > self.config.floater_near_camrea_num_pixels_threshold) & \
                                      (point_depth_in_camera < self.config.floater_depth_threshold))

            # floater_mask_in_camera = (num_affected_pixels > self.config.floater_num_pixels_threshold)
            floater_point_id = point_id_in_camera_list[floater_mask_in_camera]
            # floater_mask = average_num_affect_pixels > self.config.floater_num_pixels_threshold
            floater_mask[floater_point_id] = True
            floater_mask = floater_mask & (self.maintained_parameters.point_invalid_mask == 0)

        point_alpha = pointcloud_features[:, 7]  # alpha before sigmoid
        nan_mask = torch.isnan(pointcloud_features).any(dim=1)
        transparent_point_mask = ((point_alpha < self.config.transparent_alpha_threshold) | nan_mask) & \
                                 (self.maintained_parameters.point_invalid_mask == 0) & \
                                 (~floater_mask)  # ensure floater points and transparent points don't overlap
        transparent_point_id = point_id_list[transparent_point_mask]
        will_be_remove_mask = floater_mask | transparent_point_mask

        # find points that are under-reconstructed or over-reconstructed
        # point_features_in_camera = pointcloud_features[point_id_in_camera_list]
        in_camera_will_be_remove_mask = floater_mask_in_camera | transparent_point_mask[point_id_in_camera_list]
        # shape: [num_points_in_camera, 2]
        grad_viewspace_norm = input_data.magnitude_grad_viewspace
        # shape: [num_points_in_camera, num_features]
        # all these three masks are on num_points_in_camera, not num_points
        in_camera_to_densify_mask = (
                grad_viewspace_norm > self.config.densification_view_space_position_gradients_threshold)
        in_camera_to_densify_mask &= (~in_camera_will_be_remove_mask)  # don't densify floater or transparent points
        num_to_densify_by_viewspace = in_camera_to_densify_mask.sum().item()
        in_camera_to_densify_mask |= (
                grad_viewspace_norm / num_affected_pixels > self.config.densification_view_avg_space_position_gradients_threshold)
        in_camera_to_densify_mask &= (~in_camera_will_be_remove_mask)  # don't densify floater or transparent points
        num_to_densify = in_camera_to_densify_mask.sum().item()
        num_to_densify_by_viewspace_avg = num_to_densify - num_to_densify_by_viewspace
        print(
            f"num_to_densify: {num_to_densify}, num_to_densify_by_viewspace: {num_to_densify_by_viewspace}, num_to_densify_by_viewspace_avg: {num_to_densify_by_viewspace_avg}")

        single_frame_densify_point_id = point_id_in_camera_list[in_camera_to_densify_mask]
        single_frame_densify_point_mask = torch.zeros_like(point_id_list, dtype=torch.bool)
        single_frame_densify_point_mask[single_frame_densify_point_id] = True

        multi_frame_average_accumulated_view_space_position_gradients = self.accumulated_view_space_position_gradients_ir / self.accumulated_num_in_camera_ir
        # fill in nan with 0
        multi_frame_average_accumulated_view_space_position_gradients[
            torch.isnan(multi_frame_average_accumulated_view_space_position_gradients)] = 0
        multi_frame_densify_mask = multi_frame_average_accumulated_view_space_position_gradients > self.config.densification_multi_frame_view_space_position_gradients_threshold

        multi_frame_average_accumulated_avg_pixel_view_space_position_gradients = self.accumulated_view_space_position_gradients_avg_ir / self.accumulated_num_in_camera_ir
        # fill in nan with 0
        multi_frame_average_accumulated_avg_pixel_view_space_position_gradients[
            torch.isnan(multi_frame_average_accumulated_avg_pixel_view_space_position_gradients)] = 0
        multi_frame_densify_mask |= (
                multi_frame_average_accumulated_avg_pixel_view_space_position_gradients / average_num_affect_pixels > self.config.densification_multi_frame_view_pixel_avg_space_position_gradients_threshold)
        multi_frame_average_position_gradients_norm = self.accumulated_position_gradients_norm_ir / self.accumulated_num_in_camera_ir
        multi_frame_densify_mask |= (
                multi_frame_average_position_gradients_norm > self.config.densification_multi_frame_position_gradients_threshold)
        to_densify_mask = (single_frame_densify_point_mask | multi_frame_densify_mask) & (~will_be_remove_mask)
        num_merged_densify = to_densify_mask.sum().item()
        print(f"num_merged_densify_with_multi_frame: {num_merged_densify}")
        densify_point_id = point_id_list[to_densify_mask]

        densify_point_position_before_optimization = pointcloud[densify_point_id]
        densify_point_grad_position = self.accumulated_position_gradients_ir[densify_point_id] / \
                                      self.accumulated_num_in_camera_ir[densify_point_id].unsqueeze(-1)
        # although acummulated_num_in_camera shall not be 0, but we still need to check it/fill in nan
        densify_point_grad_position[torch.isnan(densify_point_grad_position)] = 0
        densify_size_reduction_factor = torch.zeros_like(densify_point_id, dtype=torch.float32,
                                                         device=pointcloud.device)
        over_reconstructed_mask = (
                self.accumulated_num_pixels_ir[to_densify_mask] > self.config.under_reconstructed_num_pixels_threshold)
        densify_size_reduction_factor[over_reconstructed_mask] = \
            np.log(self.config.gaussian_split_factor_phi)
        densify_size_reduction_factor = densify_size_reduction_factor.unsqueeze(-1)
        self.densify_point_info_ir = GaussianPointAdaptiveController.GaussianPointAdaptiveControllerDensifyPointInfo(
            floater_point_id=floater_point_id,
            transparent_point_id=transparent_point_id,
            densify_point_id=densify_point_id,
            densify_point_position_before_optimization=densify_point_position_before_optimization,
            densify_size_reduction_factor=densify_size_reduction_factor,
            densify_point_grad_position=densify_point_grad_position,
        )

    def _find_densify_points_ms(self, input_data: GaussianPointCloudRasterisation.BackwardValidPointHookInput):
        """ for additional modality adaptive controller, let the densify points invisible in main modality.

        Args:
            input_data (GaussianPointCloudRasterisation.BackwardValidPointHookInput): input
        """
        pointcloud = self.maintained_parameters.pointcloud
        pointcloud_features = self.maintained_parameters.pointcloud_features
        point_id_list = torch.arange(pointcloud.shape[0], device=pointcloud.device)
        point_id_in_camera_list: torch.Tensor = input_data.point_id_in_camera_list
        num_affected_pixels: torch.Tensor = input_data.num_affected_pixels
        point_depth_in_camera: torch.Tensor = input_data.point_depth
        point_uv_in_camera: torch.Tensor = input_data.point_uv_in_camera
        average_num_affect_pixels = self.accumulated_num_pixels_ms / self.accumulated_num_in_camera_ms
        average_num_affect_pixels[torch.isnan(average_num_affect_pixels)] = 0

        # Note that transparent points are apply on all valid points
        # while floater and densification only apply on points in camera in the current frame
        floater_mask = torch.zeros_like(point_id_list, dtype=torch.bool)
        floater_mask_in_camera = torch.zeros_like(point_id_in_camera_list, dtype=torch.bool)

        floater_point_id = torch.empty(0, dtype=torch.int32, device=pointcloud.device)
        if self.iteration_counter > self.config.iteration_start_remove_floater:
            floater_mask_in_camera = ((num_affected_pixels > self.config.floater_near_camrea_num_pixels_threshold) & \
                                      (point_depth_in_camera < self.config.floater_depth_threshold))

            # floater_mask_in_camera = (num_affected_pixels > self.config.floater_num_pixels_threshold)
            floater_point_id = point_id_in_camera_list[floater_mask_in_camera]
            # floater_mask = average_num_affect_pixels > self.config.floater_num_pixels_threshold
            floater_mask[floater_point_id] = True
            floater_mask = floater_mask & (self.maintained_parameters.point_invalid_mask == 0)

        point_alpha = pointcloud_features[:, 7]  # alpha before sigmoid
        nan_mask = torch.isnan(pointcloud_features).any(dim=1)
        transparent_point_mask = ((point_alpha < self.config.transparent_alpha_threshold) | nan_mask) & \
                                 (self.maintained_parameters.point_invalid_mask == 0) & \
                                 (~floater_mask)  # ensure floater points and transparent points don't overlap
        transparent_point_id = point_id_list[transparent_point_mask]
        will_be_remove_mask = floater_mask | transparent_point_mask

        # find points that are under-reconstructed or over-reconstructed
        # point_features_in_camera = pointcloud_features[point_id_in_camera_list]
        in_camera_will_be_remove_mask = floater_mask_in_camera | transparent_point_mask[point_id_in_camera_list]
        # shape: [num_points_in_camera, 2]
        grad_viewspace_norm = input_data.magnitude_grad_viewspace
        # shape: [num_points_in_camera, num_features]
        # all these three masks are on num_points_in_camera, not num_points
        in_camera_to_densify_mask = (
                grad_viewspace_norm > self.config.densification_view_space_position_gradients_threshold)
        in_camera_to_densify_mask &= (~in_camera_will_be_remove_mask)  # don't densify floater or transparent points
        num_to_densify_by_viewspace = in_camera_to_densify_mask.sum().item()
        in_camera_to_densify_mask |= (
                grad_viewspace_norm / num_affected_pixels > self.config.densification_view_avg_space_position_gradients_threshold)
        in_camera_to_densify_mask &= (~in_camera_will_be_remove_mask)  # don't densify floater or transparent points
        num_to_densify = in_camera_to_densify_mask.sum().item()
        num_to_densify_by_viewspace_avg = num_to_densify - num_to_densify_by_viewspace
        print(
            f"num_to_densify: {num_to_densify}, num_to_densify_by_viewspace: {num_to_densify_by_viewspace}, num_to_densify_by_viewspace_avg: {num_to_densify_by_viewspace_avg}")

        single_frame_densify_point_id = point_id_in_camera_list[in_camera_to_densify_mask]
        single_frame_densify_point_mask = torch.zeros_like(point_id_list, dtype=torch.bool)
        single_frame_densify_point_mask[single_frame_densify_point_id] = True

        multi_frame_average_accumulated_view_space_position_gradients = self.accumulated_view_space_position_gradients_ms / self.accumulated_num_in_camera_ms
        # fill in nan with 0
        multi_frame_average_accumulated_view_space_position_gradients[
            torch.isnan(multi_frame_average_accumulated_view_space_position_gradients)] = 0
        multi_frame_densify_mask = multi_frame_average_accumulated_view_space_position_gradients > self.config.densification_multi_frame_view_space_position_gradients_threshold

        multi_frame_average_accumulated_avg_pixel_view_space_position_gradients = self.accumulated_view_space_position_gradients_avg_ms / self.accumulated_num_in_camera_ms
        # fill in nan with 0
        multi_frame_average_accumulated_avg_pixel_view_space_position_gradients[
            torch.isnan(multi_frame_average_accumulated_avg_pixel_view_space_position_gradients)] = 0
        multi_frame_densify_mask |= (
                multi_frame_average_accumulated_avg_pixel_view_space_position_gradients / average_num_affect_pixels > self.config.densification_multi_frame_view_pixel_avg_space_position_gradients_threshold)
        multi_frame_average_position_gradients_norm = self.accumulated_position_gradients_norm_ms / self.accumulated_num_in_camera_ms
        multi_frame_densify_mask |= (
                multi_frame_average_position_gradients_norm > self.config.densification_multi_frame_position_gradients_threshold)
        to_densify_mask = (single_frame_densify_point_mask | multi_frame_densify_mask) & (~will_be_remove_mask)
        num_merged_densify = to_densify_mask.sum().item()
        print(f"num_merged_densify_with_multi_frame: {num_merged_densify}")
        densify_point_id = point_id_list[to_densify_mask]

        densify_point_position_before_optimization = pointcloud[densify_point_id]
        densify_point_grad_position = self.accumulated_position_gradients_ms[densify_point_id] / \
                                      self.accumulated_num_in_camera_ms[densify_point_id].unsqueeze(-1)
        # although acummulated_num_in_camera shall not be 0, but we still need to check it/fill in nan
        densify_point_grad_position[torch.isnan(densify_point_grad_position)] = 0
        densify_size_reduction_factor = torch.zeros_like(densify_point_id, dtype=torch.float32,
                                                         device=pointcloud.device)
        over_reconstructed_mask = (
                self.accumulated_num_pixels_ms[to_densify_mask] > self.config.under_reconstructed_num_pixels_threshold)
        densify_size_reduction_factor[over_reconstructed_mask] = \
            np.log(self.config.gaussian_split_factor_phi)
        densify_size_reduction_factor = densify_size_reduction_factor.unsqueeze(-1)
        self.densify_point_info_ms = GaussianPointAdaptiveController.GaussianPointAdaptiveControllerDensifyPointInfo(
            floater_point_id=floater_point_id,
            transparent_point_id=transparent_point_id,
            densify_point_id=densify_point_id,
            densify_point_position_before_optimization=densify_point_position_before_optimization,
            densify_size_reduction_factor=densify_size_reduction_factor,
            densify_point_grad_position=densify_point_grad_position,
        )

    def _add_densify_points_cross_spectral(self):
        assert self.densify_point_info is not None and self.densify_point_info_ms is not None and self.densify_point_info_ir is not None
        total_valid_points_before_densify = self.maintained_parameters.point_invalid_mask.shape[
                                                0] - self.maintained_parameters.point_invalid_mask.sum()

        # set transparent points invalid
        # current state: set all transparent points invalid(RGB or MS)
        num_transparent_points_RGB = self.densify_point_info.transparent_point_id.shape[0]
        num_transparent_points_MS = self.densify_point_info_ms.transparent_point_id.shape[0]
        num_transparent_points_IR = self.densify_point_info_ir.transparent_point_id.shape[0]
        self.maintained_parameters.point_invalid_mask[self.densify_point_info.transparent_point_id] = 1
        self.maintained_parameters.point_invalid_mask[self.densify_point_info_ms.transparent_point_id] = 1
        self.maintained_parameters.point_invalid_mask[self.densify_point_info_ir.transparent_point_id] = 1

        # set floater points invalid
        # ===============>current state: set all floater points invalid(RGB or MS)
        num_floaters_points_RGB = self.densify_point_info.floater_point_id.shape[0]
        num_floaters_points_MS = self.densify_point_info_ms.floater_point_id.shape[0]
        num_floaters_points_IR = self.densify_point_info_ir.floater_point_id.shape[0]
        self.maintained_parameters.point_invalid_mask[self.densify_point_info.floater_point_id] = 1
        self.maintained_parameters.point_invalid_mask[self.densify_point_info_ms.floater_point_id] = 1
        self.maintained_parameters.point_invalid_mask[self.densify_point_info_ir.floater_point_id] = 1

        # process densify points separately
        # note: implied bug: num of invalid points is not enough
        num_of_densify_points = self.densify_point_info.densify_point_id.shape[0] + \
                                self.densify_point_info_ms.densify_point_id.shape[0] + \
                                self.densify_point_info_ir.densify_point_id.shape[0]
        num_of_densify_points_RGB = self.densify_point_info.densify_point_id.shape[0]
        num_of_densify_points_MS = self.densify_point_info_ms.densify_point_id.shape[0]
        num_of_densify_points_IR = self.densify_point_info_ir.densify_point_id.shape[0]
        invalid_point_id_to_fill_sum = \
            torch.where(self.maintained_parameters.point_invalid_mask == 1)[0][:num_of_densify_points]
        invalid_point_id_to_fill_RGB = invalid_point_id_to_fill_sum[:num_of_densify_points_RGB]
        invalid_point_id_to_fill_MS = \
            invalid_point_id_to_fill_sum[num_of_densify_points_RGB:num_of_densify_points_RGB+num_of_densify_points_MS]
        invalid_point_id_to_fill_IR = \
            invalid_point_id_to_fill_sum[num_of_densify_points_RGB+num_of_densify_points_MS:]

        if num_of_densify_points > 0:
            # copy RGB densify points parameters
            self.maintained_parameters.pointcloud[invalid_point_id_to_fill_RGB] = \
                self.densify_point_info.densify_point_position_before_optimization[:num_of_densify_points_RGB]

            self.maintained_parameters.pointcloud_features[invalid_point_id_to_fill_RGB, :] = \
                self.maintained_parameters.pointcloud_features[
                self.densify_point_info.densify_point_id[:num_of_densify_points_RGB], :]

            self.maintained_parameters.point_object_id[invalid_point_id_to_fill_RGB] = \
                self.maintained_parameters.point_object_id[
                    self.densify_point_info.densify_point_id[:num_of_densify_points_RGB]]

            self.maintained_parameters.pointcloud_features[invalid_point_id_to_fill_RGB, 4:7] -= \
                self.densify_point_info.densify_size_reduction_factor[:num_of_densify_points_RGB]

            # copy MS densify points parameters
            self.maintained_parameters.pointcloud[invalid_point_id_to_fill_MS] = \
                self.densify_point_info_ms.densify_point_position_before_optimization[:num_of_densify_points_MS]

            self.maintained_parameters.pointcloud_features[invalid_point_id_to_fill_MS, :] = \
                self.maintained_parameters.pointcloud_features[
                self.densify_point_info_ms.densify_point_id[:num_of_densify_points_MS], :]

            self.maintained_parameters.point_object_id[invalid_point_id_to_fill_MS] = \
                self.maintained_parameters.point_object_id[
                    self.densify_point_info_ms.densify_point_id[:num_of_densify_points_MS]]

            self.maintained_parameters.pointcloud_features[invalid_point_id_to_fill_MS, 4:7] -= \
                self.densify_point_info_ms.densify_size_reduction_factor[:num_of_densify_points_MS]

            # copy IR densify points parameters
            self.maintained_parameters.pointcloud[invalid_point_id_to_fill_IR] = \
                self.densify_point_info_ir.densify_point_position_before_optimization[:num_of_densify_points_IR]

            self.maintained_parameters.pointcloud_features[invalid_point_id_to_fill_IR, :] = \
                self.maintained_parameters.pointcloud_features[
                self.densify_point_info_ir.densify_point_id[:num_of_densify_points_IR], :]

            self.maintained_parameters.point_object_id[invalid_point_id_to_fill_IR] = \
                self.maintained_parameters.point_object_id[
                    self.densify_point_info_ir.densify_point_id[:num_of_densify_points_IR]]

            self.maintained_parameters.pointcloud_features[invalid_point_id_to_fill_IR, 4:7] -= \
                self.densify_point_info_ir.densify_size_reduction_factor[:num_of_densify_points_IR]

            # separate over_reconstructed/under_reconstructed point
            over_reconstructed_mask_RGB = (self.densify_point_info.densify_size_reduction_factor[
                                           :num_of_densify_points_RGB] > 1e-6).reshape(-1)
            under_reconstructed_mask_RGB = ~over_reconstructed_mask_RGB
            num_over_reconstructed_RGB = over_reconstructed_mask_RGB.sum().item()
            num_under_reconstructed_RGB = num_of_densify_points_RGB - num_over_reconstructed_RGB

            over_reconstructed_mask_MS = (self.densify_point_info_ms.densify_size_reduction_factor[
                                          :num_of_densify_points_MS] > 1e-6).reshape(-1)
            under_reconstructed_mask_MS = ~over_reconstructed_mask_MS
            num_over_reconstructed_MS = over_reconstructed_mask_MS.sum().item()
            num_under_reconstructed_MS = num_of_densify_points_MS - num_over_reconstructed_MS

            over_reconstructed_mask_IR = (self.densify_point_info_ir.densify_size_reduction_factor[
                                          :num_of_densify_points_IR] > 1e-6).reshape(-1)
            under_reconstructed_mask_IR = ~over_reconstructed_mask_IR
            num_over_reconstructed_IR = over_reconstructed_mask_IR.sum().item()
            num_under_reconstructed_IR = num_of_densify_points_IR - num_over_reconstructed_IR
            print("<Densify info>")
            print(
                f"<RGB rasterisation densify> num_over_reconstructed: {num_over_reconstructed_RGB},"
                f"num_under_reconstructed: {num_under_reconstructed_RGB}")
            print(
                f"<MS rasterisation densify> num_over_reconstructed: {num_over_reconstructed_MS},"
                f"num_under_reconstructed: {num_under_reconstructed_MS}")
            print(
                f"<IR rasterisation densify> num_over_reconstructed: {num_over_reconstructed_IR},"
                f"num_under_reconstructed: {num_under_reconstructed_IR}")

            # synchronous change to original points
            densify_point_id_RGB = self.densify_point_info.densify_point_id[:num_of_densify_points_RGB]
            self.maintained_parameters.pointcloud_features[densify_point_id_RGB, 4:7] -= \
                self.densify_point_info.densify_size_reduction_factor[:num_of_densify_points_RGB]

            densify_point_id_MS = self.densify_point_info_ms.densify_point_id[:num_of_densify_points_MS]
            densify_point_id_IR = self.densify_point_info_ir.densify_point_id[:num_of_densify_points_IR]
            # densify_point_id_MS_without_overlap = \
            #     densify_point_id_MS[torch.isin(densify_point_id_MS, densify_point_id_RGB)]
            # print(
            #     f"<MS rasterisation densify> num_densify_without_overlap: {densify_point_id_MS_without_overlap.shape[0]}")
            # self.maintained_parameters.pointcloud_features[densify_point_id_MS_without_overlap, 4:7] -= \
            #     self.densify_point_info_ms.densify_size_reduction_factor[:num_of_densify_points_MS][
            #         torch.isin(densify_point_id_MS, densify_point_id_RGB)]

            # sample point for over_reconstructed points
            if self.config.enable_sample_from_point:
                # split over_reconstructed points
                over_reconstructed_point_id_RGB = densify_point_id_RGB[over_reconstructed_mask_RGB]
                over_reconstructed_point_id_to_fill_RGB = invalid_point_id_to_fill_RGB[over_reconstructed_mask_RGB]
                over_reconstructed_point_id_MS = densify_point_id_MS[over_reconstructed_mask_MS]
                over_reconstructed_point_id_to_fill_MS = invalid_point_id_to_fill_MS[over_reconstructed_mask_MS]
                over_reconstructed_point_id_IR = densify_point_id_IR[over_reconstructed_mask_IR]
                over_reconstructed_point_id_to_fill_IR = invalid_point_id_to_fill_IR[over_reconstructed_mask_IR]
                assert over_reconstructed_point_id_RGB.shape[0] == over_reconstructed_point_id_to_fill_RGB.shape[0]
                assert over_reconstructed_point_id_MS.shape[0] == over_reconstructed_point_id_to_fill_MS.shape[0]
                assert over_reconstructed_point_id_IR.shape[0] == over_reconstructed_point_id_to_fill_IR.shape[0]

                point_position_RGB = self._sample_from_point(
                    point_to_split=self.maintained_parameters.pointcloud[over_reconstructed_point_id_RGB],
                    point_feature_to_split=self.maintained_parameters.pointcloud_features[
                        over_reconstructed_point_id_RGB])
                point_position_MS = self._sample_from_point(
                    point_to_split=self.maintained_parameters.pointcloud[over_reconstructed_point_id_MS],
                    point_feature_to_split=self.maintained_parameters.pointcloud_features[
                        over_reconstructed_point_id_MS])
                point_position_IR = self._sample_from_point(
                    point_to_split=self.maintained_parameters.pointcloud[over_reconstructed_point_id_IR],
                    point_feature_to_split=self.maintained_parameters.pointcloud_features[
                        over_reconstructed_point_id_IR])

                # ensure that if conflict occurs, the original point change with main-modality
                # the to-fill points change as normal
                # self.maintained_parameters.pointcloud[over_reconstructed_point_id_MS] = point_position_MS
                self.maintained_parameters.pointcloud[over_reconstructed_point_id_RGB] = point_position_RGB

                self.maintained_parameters.pointcloud[over_reconstructed_point_id_to_fill_RGB] = point_position_RGB
                self.maintained_parameters.pointcloud[over_reconstructed_point_id_to_fill_MS] = point_position_MS
                self.maintained_parameters.pointcloud[over_reconstructed_point_id_to_fill_IR] = point_position_IR

                # clone under_reconstructed points
                under_reconstructed_point_id_to_fill_RGB = invalid_point_id_to_fill_RGB[under_reconstructed_mask_RGB]
                under_reconstructed_point_grad_position_RGB = \
                    self.densify_point_info.densify_point_grad_position[:num_of_densify_points_RGB][
                        under_reconstructed_mask_RGB]
                self.maintained_parameters.pointcloud[under_reconstructed_point_id_to_fill_RGB] += \
                    under_reconstructed_point_grad_position_RGB * self.config.under_reconstructed_move_factor

                under_reconstructed_point_id_to_fill_MS = invalid_point_id_to_fill_MS[under_reconstructed_mask_MS]
                under_reconstructed_point_grad_position_MS = \
                    self.densify_point_info_ms.densify_point_grad_position[:num_of_densify_points_MS][
                        under_reconstructed_mask_MS]
                self.maintained_parameters.pointcloud[under_reconstructed_point_id_to_fill_MS] += \
                    under_reconstructed_point_grad_position_MS * self.config.under_reconstructed_move_factor

                under_reconstructed_point_id_to_fill_IR = invalid_point_id_to_fill_IR[under_reconstructed_mask_IR]
                under_reconstructed_point_grad_position_IR = \
                    self.densify_point_info_ir.densify_point_grad_position[:num_of_densify_points_IR][
                        under_reconstructed_mask_IR]
                self.maintained_parameters.pointcloud[under_reconstructed_point_id_to_fill_IR] += \
                    under_reconstructed_point_grad_position_IR * self.config.under_reconstructed_move_factor

                self.maintained_parameters.point_invalid_mask[invalid_point_id_to_fill_sum] = 0

            total_valid_points_after_densify = \
                self.maintained_parameters.point_invalid_mask.shape[
                    0] - self.maintained_parameters.point_invalid_mask.sum()
            # maybe conflict inside, so cannot validation
            # assert total_valid_points_after_densify == total_valid_points_before_densify - num_transparent_points - num_floaters_points + num_fillable_densify_points
            print(
                f"total valid points: {total_valid_points_before_densify} -> {total_valid_points_after_densify},"
                f"num_densify_points: {num_of_densify_points}")
            print(
                f"<RGB rasterisation densify> num_transparent_points: {num_transparent_points_RGB},"
                f"num_floaters_points: {num_floaters_points_RGB},"
                f"num_fillable_densify_points: {num_of_densify_points_RGB}")
            print(
                f"<MS rasterisation densify> num_transparent_points: {num_transparent_points_MS}, "
                f"num_floaters_points: {num_floaters_points_MS},"
                f"num_fillable_densify_points: {num_of_densify_points_MS}")
            print(
                f"<IR rasterisation densify> num_transparent_points: {num_transparent_points_IR},"
                f"num_floaters_points: {num_floaters_points_IR},"
                f"num_fillable_densify_points: {num_of_densify_points_IR}")

            # clear densify point info
            self.densify_point_info = None
            self.densify_point_info_ms = None
            self.densify_point_info_ir = None

    def _add_densify_points(self):
        assert self.densify_point_info is not None
        total_valid_points_before_densify = self.maintained_parameters.point_invalid_mask.shape[0] - \
                                            self.maintained_parameters.point_invalid_mask.sum()
        num_transparent_points = self.densify_point_info.transparent_point_id.shape[0]
        self.maintained_parameters.point_invalid_mask[self.densify_point_info.transparent_point_id] = 1
        num_floaters_points = self.densify_point_info.floater_point_id.shape[0]
        self.maintained_parameters.point_invalid_mask[self.densify_point_info.floater_point_id] = 1
        num_of_densify_points = self.densify_point_info.densify_point_id.shape[0]
        invalid_point_id_to_fill = torch.where(self.maintained_parameters.point_invalid_mask == 1)[0][
                                   :num_of_densify_points]

        # for position, we use the position before optimization for new points, so that original points and new points have different positions
        num_fillable_densify_points = 0
        if num_of_densify_points > 0:
            # num_fillable_over_reconstructed_points = over_reconstructed_point_id_to_fill.shape[0]
            num_fillable_densify_points = min(num_of_densify_points, invalid_point_id_to_fill.shape[0])
            self.maintained_parameters.pointcloud[invalid_point_id_to_fill] = \
                self.densify_point_info.densify_point_position_before_optimization[:num_fillable_densify_points]

            self.maintained_parameters.pointcloud_features[invalid_point_id_to_fill] = \
                self.maintained_parameters.pointcloud_features[
                    self.densify_point_info.densify_point_id[:num_fillable_densify_points]]

            self.maintained_parameters.point_object_id[invalid_point_id_to_fill] = \
                self.maintained_parameters.point_object_id[
                    self.densify_point_info.densify_point_id[:num_fillable_densify_points]]

            self.maintained_parameters.pointcloud_features[invalid_point_id_to_fill, 4:7] -= \
                self.densify_point_info.densify_size_reduction_factor[:num_fillable_densify_points]

            over_reconstructed_mask = (self.densify_point_info.densify_size_reduction_factor[
                                       :num_fillable_densify_points] > 1e-6).reshape(-1)
            under_reconstructed_mask = ~over_reconstructed_mask
            num_over_reconstructed = over_reconstructed_mask.sum().item()
            num_under_reconstructed = num_fillable_densify_points - num_over_reconstructed
            print(
                f"num_over_reconstructed: {num_over_reconstructed}, num_under_reconstructed: {num_under_reconstructed}")
            densify_point_id = self.densify_point_info.densify_point_id[:num_fillable_densify_points]
            self.maintained_parameters.pointcloud_features[densify_point_id, 4:7] -= \
                self.densify_point_info.densify_size_reduction_factor[:num_fillable_densify_points]
            # current False
            if self.config.enable_ellipsoid_offset:
                point_offset = self._generate_point_offset(
                    point_to_split=self.maintained_parameters.pointcloud[densify_point_id],
                    point_feature_to_split=self.maintained_parameters.pointcloud_features[densify_point_id])
                self.maintained_parameters.pointcloud[invalid_point_id_to_fill] += point_offset
                self.maintained_parameters.pointcloud[densify_point_id] -= point_offset
            # current True
            if self.config.enable_sample_from_point:
                over_reconstructed_point_id = densify_point_id[over_reconstructed_mask]
                over_reconstructed_point_id_to_fill = invalid_point_id_to_fill[over_reconstructed_mask]
                assert over_reconstructed_point_id.shape[0] == over_reconstructed_point_id_to_fill.shape[0]
                point_position = self._sample_from_point(
                    point_to_split=self.maintained_parameters.pointcloud[over_reconstructed_point_id],
                    point_feature_to_split=self.maintained_parameters.pointcloud_features[over_reconstructed_point_id])
                self.maintained_parameters.pointcloud[over_reconstructed_point_id_to_fill] = point_position
                # rerun same function to also sample position for original points
                point_position = self._sample_from_point(
                    point_to_split=self.maintained_parameters.pointcloud[over_reconstructed_point_id],
                    point_feature_to_split=self.maintained_parameters.pointcloud_features[over_reconstructed_point_id])
                self.maintained_parameters.pointcloud[over_reconstructed_point_id] = point_position
                under_reconstructed_point_id_to_fill = invalid_point_id_to_fill[under_reconstructed_mask]
                under_reconstructed_point_grad_position = \
                    self.densify_point_info.densify_point_grad_position[:num_fillable_densify_points][
                        under_reconstructed_mask]
                self.maintained_parameters.pointcloud[
                    under_reconstructed_point_id_to_fill] += under_reconstructed_point_grad_position * \
                                                             self.config.under_reconstructed_move_factor

            self.maintained_parameters.point_invalid_mask[invalid_point_id_to_fill] = 0
        total_valid_points_after_densify = self.maintained_parameters.point_invalid_mask.shape[0] - \
                                           self.maintained_parameters.point_invalid_mask.sum()
        assert total_valid_points_after_densify == total_valid_points_before_densify - num_transparent_points - num_floaters_points + num_fillable_densify_points
        print(
            f"total valid points: {total_valid_points_before_densify} -> {total_valid_points_after_densify}, num_densify_points: {num_of_densify_points}, num_fillable_densify_points: {num_fillable_densify_points}")
        print(f"num_transparent_points: {num_transparent_points}, num_floaters_points: {num_floaters_points}")
        self.densify_point_info = None  # clear densify point info

    def reset_alpha(self):
        pointcloud_features = self.maintained_parameters.pointcloud_features
        pointcloud_features[:, 7] = torch.clamp(pointcloud_features[:, 7],
                                                max=self.config.reset_alpha_value)

    def _generate_point_offset(self,
                               point_to_split: torch.Tensor,  # (N, 3)
                               point_feature_to_split: torch.Tensor,  # (N, 56)
                               ):
        # generate extra offset for the point to split. The point is modeled as ellipsoid, with center at point_to_split,
        # and axis length specified by s, and rotation specified by q.
        # For this solution, we want to put the two new points on the foci of the ellipsoid, so the offset
        # is the vector from the center to the foci.
        select_points = point_to_split.contiguous()
        select_point_features = point_feature_to_split.contiguous()
        point_offset = torch.zeros_like(select_points)
        compute_ellipsoid_offset(
            pointcloud=select_points,
            pointcloud_features=select_point_features,
            ellipsoid_offset=point_offset,
        )
        return point_offset

    def _sample_from_point(self,
                           point_to_split: torch.Tensor,  # (N, 3)
                           point_feature_to_split: torch.Tensor,  # (N, 56)
                           ):
        select_points = point_to_split.contiguous()
        select_point_features = point_feature_to_split.contiguous()
        point_sampled = torch.zeros_like(select_points)
        sample_from_point(
            pointcloud=select_points,
            pointcloud_features=select_point_features,
            sample_result=point_sampled)
        return point_sampled
