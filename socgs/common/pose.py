"""Learnable cross-spectral camera pose module.

The RGB camera poses come from COLMAP; the poses of the co-located
multispectral / infrared cameras are initialised from the RGB pose and then
refined jointly with the scene through the learnable SO(3) + translation
parameters of :class:`LearnPose` ("matching-optimising pose estimation" in the
paper).

The per-view extrinsics are represented as camera-to-world transforms
(c2w) built from an axis-angle vector ``r`` and a translation ``t``.
"""

import numpy as np
import torch
import torch.nn as nn

from .utils import SE3_to_quaternion_and_translation_torch, inverse_SE3


def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4) camera-to-world matrix
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat(
                [input, torch.tensor([[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate(
                [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def Exp(r):
    """so(3) vector to SO(3) matrix (Rodrigues' formula)

    :param r: (3, ) axis-angle vector, torch tensor
    :return:  (3, 3) rotation matrix
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3) skew-symmetric matrix of v
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([zero, -v[2:3], v[1:2]])
    skew_v1 = torch.cat([v[2:3], zero, -v[0:1]])
    skew_v2 = torch.cat([-v[1:2], v[0:1], zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v


def calculate_output_c2w(r, t):
    """Build a camera-to-world transform from axis-angle + translation."""
    c2w = make_c2w(r, t)
    return c2w


# learn the camera pose of another modality respectively; the result of the
# initialisation bundle adjustment is used as the starting point
class LearnPose(nn.Module):
    def __init__(self, initial_R, initial_t):
        """
        :param initial_R: (n_cameras, 3) axis-angle parameters (tensor)
        :param initial_t: (n_cameras, 3) translation parameters (tensor)
        """
        super(LearnPose, self).__init__()
        self.r = nn.Parameter(initial_R, requires_grad=True)  # (n_cameras, 3)
        self.t = nn.Parameter(initial_t, requires_grad=True)  # (n_cameras, 3)

    def forward(self, camera_id):
        r_ = self.r[camera_id, :].squeeze()  # (3, ) axis-angle
        t_ = self.t[camera_id, :].squeeze()  # (3, )
        c2w_ms = calculate_output_c2w(r=r_, t=t_)
        T_pointcloud_camera_ms = inverse_SE3(c2w_ms)
        q_pointcloud_camera_infrared, t_pointcloud_camera_infrared = \
            SE3_to_quaternion_and_translation_torch(T_pointcloud_camera_ms)
        t_pointcloud_camera_infrared = t_pointcloud_camera_infrared.unsqueeze(0)
        return q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, r_.squeeze(), t_.squeeze()
