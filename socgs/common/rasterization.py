"""Shared allocation and geometry helpers used by the rasterizers."""

import math
from typing import Optional, Tuple

import numpy as np
import torch


BOUNDARY_TILES = 3
TILE_WIDTH = 16
TILE_HEIGHT = 16


def allocate_tile_ranges(
    camera_width: int,
    camera_height: int,
    pointcloud: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate the inclusive/exclusive point offsets for every image tile."""
    tile_count = (camera_width // TILE_WIDTH) * (camera_height // TILE_HEIGHT)
    starts = torch.zeros(tile_count, dtype=torch.int32, device=pointcloud.device)
    ends = torch.zeros(tile_count, dtype=torch.int32, device=pointcloud.device)
    return starts, ends


def allocate_render_buffers(
    camera_width: int,
    camera_height: int,
    pointcloud: torch.Tensor,
    rendered_channel: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate output and bookkeeping buffers for a rasterization pass."""
    device = pointcloud.device
    image = torch.empty(
        camera_height, camera_width, rendered_channel, dtype=torch.float32, device=device
    )
    depth = torch.empty(camera_height, camera_width, dtype=torch.float32, device=device)
    accumulated_alpha = torch.empty(
        camera_height, camera_width, dtype=torch.float32, device=device
    )
    last_effective_point = torch.empty(
        camera_height, camera_width, dtype=torch.int32, device=device
    )
    valid_point_count = torch.empty(
        camera_height, camera_width, dtype=torch.int32, device=device
    )
    return image, depth, accumulated_alpha, last_effective_point, valid_point_count


def allocate_projected_point_buffers(
    num_points_in_camera: int,
    pointcloud: torch.Tensor,
    color_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate per-point buffers produced by camera projection."""
    device = pointcloud.device
    point_uv = torch.empty((num_points_in_camera, 2), dtype=torch.float32, device=device)
    point_alpha = torch.empty(num_points_in_camera, dtype=torch.float32, device=device)
    point_in_camera = torch.empty((num_points_in_camera, 3), dtype=torch.float32, device=device)
    conic_and_rescale = torch.empty((num_points_in_camera, 4), dtype=torch.float32, device=device)
    point_color = torch.zeros(
        (num_points_in_camera, color_channels), dtype=torch.float32, device=device
    )
    point_radii = torch.empty(num_points_in_camera, dtype=torch.float32, device=device)
    return (
        point_uv,
        point_alpha,
        point_in_camera,
        conic_and_rescale,
        point_color,
        point_radii,
    )


def intersect_line_with_ellipsoid(
    line_equations: np.ndarray,
    ellipsoid_translation: np.ndarray,
    ellipsoid_rotation: np.ndarray,
    ellipsoid_scale: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return the two intersections of a 3-D line and a transformed unit sphere."""
    transform = ellipsoid_translation @ ellipsoid_rotation @ ellipsoid_scale
    a1, b1, c1, d1 = line_equations[0, :]
    a2, b2, c2, d2 = line_equations[1, :]
    denominator = a1 * b2 - a2 * b1
    line_origin = np.array(
        [(b1 * d2 - d1 * b2) / denominator, (a2 * d1 - a1 * d2) / denominator, 0, 1.]
    )
    line_direction = np.array(
        [(b1 * c2 - b2 * c1), (a2 * c1 - a1 * c2), denominator, 1.]
    )
    transformed_origin = np.linalg.inv(transform) @ line_origin
    transformed_direction = np.linalg.inv(transform) @ line_direction
    coefficient_a = np.dot(transformed_direction[:-1], transformed_direction[:-1])
    coefficient_b = 2 * np.dot(transformed_direction[:-1], transformed_origin[:-1])
    coefficient_c = np.dot(transformed_origin[:-1], transformed_origin[:-1]) - 1.
    discriminant = coefficient_b**2 - 4 * coefficient_a * coefficient_c

    if discriminant < 0:
        return None, None
    if discriminant == 0:
        parameter = -coefficient_b / (2 * coefficient_a)
        intersection = line_origin[:-1] + parameter * line_direction[:-1]
        return intersection, intersection.copy()

    first_parameter = (-coefficient_b + math.sqrt(discriminant)) / (2 * coefficient_a)
    second_parameter = (-coefficient_b - math.sqrt(discriminant)) / (2 * coefficient_a)
    return (
        line_origin[:-1] + first_parameter * line_direction[:-1],
        line_origin[:-1] + second_parameter * line_direction[:-1],
    )


# Historical aliases remain available for downstream code that imported these helpers.
calculate_tiles_start_and_end = allocate_tile_ranges
allocate_memory_for_rendered_images = allocate_render_buffers
allocate_memory_for_filter_points = allocate_projected_point_buffers


def find_intersection_of_line_and_ellipse(
    line_general_equation: np.ndarray,
    ellipse_translation: np.ndarray,
    ellipse_rotation: np.ndarray,
    ellipse_scale: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compatibility wrapper for the name and keywords used by the paper code."""
    return intersect_line_with_ellipsoid(
        line_equations=line_general_equation,
        ellipsoid_translation=ellipse_translation,
        ellipsoid_rotation=ellipse_rotation,
        ellipsoid_scale=ellipse_scale,
    )
