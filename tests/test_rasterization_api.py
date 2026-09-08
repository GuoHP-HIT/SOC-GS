"""CPU-only regression checks for the rasterization module split."""

import unittest

import numpy as np
import torch

from socgs.common.rasterization import (
    allocate_projected_point_buffers,
    allocate_render_buffers,
    allocate_tile_ranges,
    find_intersection_of_line_and_ellipse,
)


class SharedRasterizationHelpersTest(unittest.TestCase):
    def test_allocated_buffer_layouts_match_the_renderer_contract(self):
        point_cloud = torch.empty((3, 3))

        starts, ends = allocate_tile_ranges(32, 16, point_cloud)
        self.assertEqual(starts.shape, (2,))
        self.assertEqual(ends.shape, (2,))
        self.assertEqual(starts.dtype, torch.int32)
        self.assertEqual(torch.count_nonzero(starts).item(), 0)

        render_buffers = allocate_render_buffers(32, 16, point_cloud, 4)
        self.assertEqual(
            [tuple(buffer.shape) for buffer in render_buffers],
            [(16, 32, 4), (16, 32), (16, 32), (16, 32), (16, 32)],
        )

        point_buffers = allocate_projected_point_buffers(3, point_cloud, 1)
        self.assertEqual(
            [tuple(buffer.shape) for buffer in point_buffers],
            [(3, 2), (3,), (3, 3), (3, 4), (3, 1), (3,)],
        )

    def test_line_ellipsoid_intersections_keep_historical_keyword_api(self):
        line = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        first, second = find_intersection_of_line_and_ellipse(
            line_general_equation=line,
            ellipse_translation=np.eye(4),
            ellipse_rotation=np.eye(4),
            ellipse_scale=np.eye(4),
        )
        np.testing.assert_array_equal(first, np.array([0., 0., 1.]))
        np.testing.assert_array_equal(second, np.array([0., 0., -1.]))


class RasterizationCompatibilityTest(unittest.TestCase):
    def test_legacy_modules_export_the_canonical_classes(self):
        from socgs.rgb_ms.rasterization import GaussianPointCloudRasterisation as Bimodal
        from socgs.rgb_ms.GaussianPointCloudRasterisation import (
            GaussianPointCloudRasterisation as LegacyBimodal,
        )
        from socgs.rgb_ir_ms.rasterization import GaussianPointCloudRasterisation as Trimodal
        from socgs.rgb_ir_ms.GaussianPointCloudRasterisation import (
            GaussianPointCloudRasterisation as LegacyTrimodal,
        )

        self.assertIs(Bimodal, LegacyBimodal)
        self.assertIs(Trimodal, LegacyTrimodal)
        self.assertEqual(Bimodal.GaussianPointCloudRasterisationConfig().near_plane, 0.8)
        self.assertEqual(Trimodal.GaussianPointCloudRasterisationConfig().near_plane, 0.4)


if __name__ == "__main__":
    unittest.main()
