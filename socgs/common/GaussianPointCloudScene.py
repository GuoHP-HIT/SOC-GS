"""Gaussian point-cloud scene: the learnable 3D representation shared by all
SOC-GS experiments (RGB+MS bimodal and RGB+IR+MS trimodal).

A scene stores one tensor of point positions ``point_cloud`` (N x 3) and one
tensor of per-point attributes ``point_cloud_features`` (N x ``num_of_features``).

Attribute layout
----------------
``num_of_features`` is composed of 8 + 16 * C slots, where C is the number of
color channels modeled by the experiment (the "spectral channels"):

* slots 0-3:    log-covariance rotation quaternion ``cov_q``
* slots 4-6:    log-covariance scale ``cov_s``
* slot  7:      alpha (before sigmoid activation)
* slots 8..:    16 spherical-harmonic coefficients per channel, in the order
  ``r, g, b, ms, ir`` (the bimodal RGB+MS experiment uses the first 4 channels,
  i.e. 72 features; the trimodal RGB+IR+MS experiment uses all 5, i.e. 88).

This layout is identical to the one used by the released SOC-GS checkpoints,
so a scene can be exchanged with the original implementation through the
parquet files written by :meth:`to_parquet` / read by
:meth:`from_trained_parquet`.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass
from dataclass_wizard import YAMLWizard
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from typing import List, Optional, Union

#: Number of attribute slots preceding the SH coefficients (see module docstring).
FEATURE_OFFSET = 8
#: Number of SH coefficients kept per spectral channel (SH degree 0-3).
SH_COEFFICIENTS = 16
#: Spectral channel column/name order used in the parquet files.
CHANNEL_NAMES = ["r", "g", "b", "ms", "ir"]


def num_color_channels(num_of_features: int) -> int:
    """Number of spectral color channels encoded in ``num_of_features``."""
    n = (num_of_features - FEATURE_OFFSET) // SH_COEFFICIENTS
    assert n * SH_COEFFICIENTS + FEATURE_OFFSET == num_of_features, \
        f"invalid num_of_features {num_of_features}: must be 8 + 16 * num_channels"
    return n


def color_channel_names(num_channels: int) -> List[str]:
    """Names of the first ``num_channels`` spectral channels."""
    return CHANNEL_NAMES[:num_channels]


def _feature_column_names(num_channels: int) -> List[str]:
    """Parquet column names of the per-point attributes (no x/y/z)."""
    columns = [f"cov_q{i}" for i in range(4)] + \
              [f"cov_s{i}" for i in range(3)] + \
              [f"alpha{i}" for i in range(1)]
    for channel in color_channel_names(num_channels):
        columns += [f"{channel}_sh{i}" for i in range(SH_COEFFICIENTS)]
    return columns


class GaussianPointCloudScene(torch.nn.Module):
    """Learnable 3D Gaussian point cloud.

    Besides the two nn.Parameters (``point_cloud``, ``point_cloud_features``)
    the module registers the ``point_invalid_mask`` buffer (1 for "empty" slots
    reserved for later densification) and the ``point_object_id`` buffer.
    """

    @dataclass
    class PointCloudSceneConfig(YAMLWizard):
        # 4 quat + 3 scale + 1 alpha + 16*C SH coefficients = 8 + 16*C features.
        num_of_features: int = 72  # 4 channels (r,g,b,ms); 88 for the IR variant
        max_num_points_ratio: Optional[float] = None  # e.g. 10.0: pre-allocate 10x slots
        add_sphere: bool = False  # add a shell of helper points around the scene
        sphere_radius_factor: float = 4.0
        num_points_sphere: int = 10000
        max_initial_covariance: Optional[float] = None
        initial_alpha: float = -2.0
        initial_covariance_ratio: float = 1.0

    def __init__(
        self,
        point_cloud: Union[np.ndarray, torch.Tensor],
        config: PointCloudSceneConfig,
        point_cloud_features: Optional[torch.Tensor] = None,
        point_object_id: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert len(point_cloud.shape) == 2, "point_cloud must be a 2D array"
        assert point_cloud.shape[1] == 3, "point_cloud must have 3 columns (x,y,z)"

        # convert point_cloud to float32
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

        # expand the number of points from num_points to
        # max_num_points_ratio * num_points (zero-filled placeholder slots)
        if config.max_num_points_ratio is not None:
            num_points = point_cloud.shape[0]
            max_num_points = int(num_points * config.max_num_points_ratio)
            assert max_num_points > num_points, "max_num_points_ratio should be greater than 1.0"
            point_cloud = torch.cat(
                [point_cloud, torch.zeros((max_num_points - num_points, 3))], dim=0)
            if point_cloud_features is not None:
                point_cloud_features = torch.cat(
                    [point_cloud_features,
                     torch.zeros((max_num_points - num_points, config.num_of_features))], dim=0)

        # send the position of points into optimizer parameter
        self.point_cloud = nn.Parameter(point_cloud)
        self.config = config
        if point_cloud_features is not None:
            self.point_cloud_features = nn.Parameter(point_cloud_features)
        else:
            self.point_cloud_features = nn.Parameter(
                torch.zeros(self.point_cloud.shape[0], self.config.num_of_features))

        # generate mask for points state
        self.register_buffer(
            "point_invalid_mask",
            torch.zeros(self.point_cloud.shape[0], dtype=torch.int8),
        )
        if point_object_id is None:
            point_object_id = torch.zeros(self.point_cloud.shape[0], dtype=torch.int32)
        self.register_buffer("point_object_id", point_object_id)

        # mark placeholder points as invalid
        if config.max_num_points_ratio is not None:
            self.point_invalid_mask[num_points:] = 1

    def forward(self):
        return self.point_cloud, self.point_cloud_features

    # ------------------------------------------------------------------ #
    # initialisation
    # ------------------------------------------------------------------ #
    def initialize(self, point_cloud_rgb: Optional[torch.Tensor] = None):
        """Initialise covariance, quaternion, alpha and SH attributes.

        ``point_cloud_rgb`` (optional, N x C in [0, 255]) seeds the DC
        component of each spectral channel from the (registered) point colors.
        """
        with torch.no_grad():
            # estimate the initial covariance matrix as an isotropic Gaussian
            # with axes equal to the mean of the distance to the closest
            # three points
            valid_point_cloud_np = self.point_cloud[self.point_invalid_mask == 0].detach(
            ).cpu().numpy()  # shape: [num_points, 3]
            nearest_neighbor_tree = cKDTree(valid_point_cloud_np)
            nearest_three_neighbor_distance, _ = nearest_neighbor_tree.query(
                valid_point_cloud_np, k=3 + 1)
            initial_covariance = np.mean(nearest_three_neighbor_distance[:, 1:], axis=1) * \
                self.config.initial_covariance_ratio
            # clip the initial covariance to [1e-6, max_initial_covariance]
            initial_covariance = np.clip(
                initial_covariance, 1e-6, self.config.max_initial_covariance)
            # s is log of the covariance, so we take log of the initial covariance
            self.point_cloud_features[(self.point_invalid_mask == 0), 4:7] = torch.tensor(
                np.log(initial_covariance), dtype=torch.float32).unsqueeze(1)

            # for rotation quaternion (x,y,z,w) we set a random normalised value
            self.point_cloud_features[:, 0:4] = torch.rand_like(self.point_cloud_features[:, 0:4])
            self.point_cloud_features[:, 0:4] = self.point_cloud_features[:, 0:4] / \
                torch.norm(self.point_cloud_features[:, 0:4], dim=1, keepdim=True)

            # alpha before sigmoid: sigmoid(alpha) = 0.5 at alpha = 0.0
            self.point_cloud_features[:, 7] = self.config.initial_alpha

            # DC (first) SH coefficient of every channel starts at 1.0,
            # higher-order coefficients are zero
            num_channels = num_color_channels(self.config.num_of_features)
            for channel in range(num_channels):
                start = FEATURE_OFFSET + channel * SH_COEFFICIENTS
                self.point_cloud_features[:, start] = 1.0
                self.point_cloud_features[:, start + 1:start + SH_COEFFICIENTS] = 0.0

            if point_cloud_rgb is not None:
                point_cloud_rgb = torch.tensor(
                    point_cloud_rgb, dtype=torch.float32, requires_grad=False,
                    device=self.point_cloud_features.device)
                point_cloud_rgb = point_cloud_rgb / 255.0
                point_cloud_rgb = torch.clamp(point_cloud_rgb, 0.0, 0.99)
                c0 = 0.28209479177387814
                for channel in range(num_channels):
                    self.point_cloud_features[(self.point_invalid_mask == 0),
                                              FEATURE_OFFSET + channel * SH_COEFFICIENTS] = \
                        self._logit(point_cloud_rgb[:, channel]) / c0

    @staticmethod
    def _logit(x: torch.Tensor) -> torch.Tensor:
        return torch.log(x / (1.0 - x))

    # ------------------------------------------------------------------ #
    # serialisation
    # ------------------------------------------------------------------ #
    def to_parquet(self, path: str):
        """Store the trained scene (xyz + attributes) as a parquet file."""
        valid_point_cloud = self.point_cloud[self.point_invalid_mask == 0]
        valid_point_cloud_features = self.point_cloud_features[self.point_invalid_mask == 0]
        point_cloud_df = pd.DataFrame(
            valid_point_cloud.detach().cpu().numpy(), columns=["x", "y", "z"])
        num_channels = num_color_channels(self.point_cloud_features.shape[1])
        point_cloud_features_df = pd.DataFrame(
            valid_point_cloud_features.detach().cpu().numpy(),
            columns=_feature_column_names(num_channels))
        scene_df = pd.concat([point_cloud_df, point_cloud_features_df], axis=1)
        scene_df.to_parquet(path)

    def to_ply(self, path: str):
        """Export the trained scene to a PLY point cloud (for external viewers)."""
        valid_point_cloud = self.point_cloud[self.point_invalid_mask == 0].detach().cpu()
        valid_point_cloud_features = self.point_cloud_features[self.point_invalid_mask == 0].detach().cpu()
        xyz = valid_point_cloud.numpy()
        normals = np.zeros_like(xyz)
        num_channels = num_color_channels(valid_point_cloud_features.shape[1])
        f_sh = valid_point_cloud_features[:, FEATURE_OFFSET:].reshape(-1, num_channels, SH_COEFFICIENTS)
        f_dc = f_sh[..., 0].numpy()
        f_rest = f_sh[..., 1:].reshape(-1, num_channels * (SH_COEFFICIENTS - 1)).numpy()
        opacities = valid_point_cloud_features[:, 7:8].numpy()
        scale = valid_point_cloud_features[:, 4:7].numpy()
        rotation = valid_point_cloud_features[:, [3, 0, 1, 2]].numpy()

        def construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # All channels except the DC component
            for i in range(f_dc.shape[1]):
                l.append('f_dc_{}'.format(i))
            for i in range(f_rest.shape[1]):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(scale.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def from_trained_parquet(path: str, config=PointCloudSceneConfig()):
        """Load a trained scene checkpoint written by :meth:`to_parquet`.

        The number of spectral channels is inferred from the columns stored in
        the parquet file, so bimodal (4-channel) and trimodal (5-channel)
        checkpoints produced by the original SOC-GS implementation load
        transparently. Helper points are never appended when loading a trained
        scene (``config.add_sphere`` is ignored for this loader).
        """
        scene_df = pd.read_parquet(path)

        # number of spectral channels encoded in the stored feature columns
        num_channels = None
        for candidate in range(1, len(CHANNEL_NAMES) + 1):
            if set(_feature_column_names(candidate)).issubset(set(scene_df.columns)):
                num_channels = candidate
        if num_channels is None:
            raise ValueError(
                f"no SOC-GS feature columns found in {path}; expected columns "
                f"{_feature_column_names(1)[:8]}...")

        config = GaussianPointCloudScene._config_for_num_channels(config, num_channels)
        feature_columns = _feature_column_names(num_channels)
        point_cloud = scene_df[["x", "y", "z"]].to_numpy()

        if not set(feature_columns).issubset(set(scene_df.columns)):
            # raw colmap-style parquet with r/g/b colors: initialise a fresh scene
            scene = GaussianPointCloudScene(point_cloud, config)
            color_names = color_channel_names(num_channels)
            df_has_color = all(name in scene_df.columns for name in color_names)
            point_cloud_colors = scene_df[color_names].to_numpy() if df_has_color else None
            scene.initialize(point_cloud_rgb=point_cloud_colors)
        else:
            valid_point_cloud_features = torch.from_numpy(
                scene_df[feature_columns].to_numpy())
            scene = GaussianPointCloudScene(
                point_cloud, config, point_cloud_features=valid_point_cloud_features)
        return scene

    @staticmethod
    def from_parquet(path: str, config=PointCloudSceneConfig()):
        """Load a raw (un-trained) point cloud and initialise a fresh scene.

        Used at the beginning of training. When the parquet file contains only
        COLMAP colors (r/g/b), the missing spectral channels (ms, and ir for
        the trimodal experiment) are initialised with the mean of r/g/b.
        """
        scene_df = pd.read_parquet(path)

        # inject the mean RGB color into the spectral channels of this model
        num_channels = num_color_channels(config.num_of_features)
        ms_initial = (scene_df['r'] + scene_df['g'] + scene_df['b']) // 3
        for name in color_channel_names(num_channels)[3:]:
            scene_df.loc[:, name] = ms_initial

        if config.add_sphere:
            scene_df = GaussianPointCloudScene._add_sphere(scene_df, config)

        feature_columns = _feature_column_names(num_channels)
        point_cloud = scene_df[["x", "y", "z"]].to_numpy()

        if not set(feature_columns).issubset(set(scene_df.columns)):
            scene = GaussianPointCloudScene(point_cloud, config)
            color_names = color_channel_names(num_channels)
            df_has_color = all(name in scene_df.columns for name in color_names)
            point_cloud_colors = scene_df[color_names].to_numpy() if df_has_color else None
            scene.initialize(point_cloud_rgb=point_cloud_colors)
        else:
            valid_point_cloud_features = torch.from_numpy(
                scene_df[feature_columns].to_numpy())
            scene = GaussianPointCloudScene(
                point_cloud, config, point_cloud_features=valid_point_cloud_features)
        return scene

    @staticmethod
    def _config_for_num_channels(config: PointCloudSceneConfig, num_channels: int):
        """Return ``config`` with ``num_of_features`` set for ``num_channels`` channels."""
        num_of_features = FEATURE_OFFSET + num_channels * SH_COEFFICIENTS
        if config.num_of_features == num_of_features:
            return config
        return GaussianPointCloudScene.PointCloudSceneConfig(
            num_of_features=num_of_features,
            max_num_points_ratio=config.max_num_points_ratio,
            add_sphere=config.add_sphere,
            sphere_radius_factor=config.sphere_radius_factor,
            num_points_sphere=config.num_points_sphere,
            max_initial_covariance=config.max_initial_covariance,
            initial_alpha=config.initial_alpha,
            initial_covariance_ratio=config.initial_covariance_ratio,
        )

    @staticmethod
    def _add_sphere(scene_df: pd.DataFrame, config: PointCloudSceneConfig):
        """Add a shell of helper points around the scene.

        The sphere radius is ``sphere_radius_factor`` times half the bounding
        box diagonal; helper points get the same neutral color as the scene
        centre so that background/out-of-view regions have geometry to densify
        from (used to fill the non-overlapping field-of-view areas between
        cross-spectral cameras).
        """
        num_channels = num_color_channels(config.num_of_features)
        color_names = color_channel_names(num_channels)
        df_has_color = all(name in scene_df.columns for name in color_names)

        # calculate the radius of the sphere
        x_min, x_max = scene_df["x"].min(), scene_df["x"].max()
        y_min, y_max = scene_df["y"].min(), scene_df["y"].max()
        z_min, z_max = scene_df["z"].min(), scene_df["z"].max()
        far_distance = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
        radius = far_distance * config.sphere_radius_factor

        # sample points on the sphere
        phi = 2.0 * np.pi * np.random.rand(config.num_points_sphere)
        theta = np.arccos(2.0 * np.random.rand(config.num_points_sphere) - 1.0)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        points = np.stack([x, y, z], axis=1)
        columns = ["x", "y", "z"]

        if df_has_color:
            neutral = np.ones((config.num_points_sphere, 1)) * (255 // 2)
            points = np.concatenate([points, neutral.repeat(3, axis=1)] +
                                    [neutral for _ in color_names[3:]], axis=1)
            columns += color_names
        scene_df = pd.concat([scene_df, pd.DataFrame(points, columns=columns)])
        return scene_df
