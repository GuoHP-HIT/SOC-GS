"""Checkpoint / pose-file helpers shared by both SOC-GS experiments.

Training stores two kinds of artifacts under the log directory of a scene
(``logs/<scene>/`` by default):

* scene checkpoints written by ``GaussianPointCloudScene.to_parquet``
  (``scene_<iter>.parquet`` / ``best_scene.parquet`` / ``final_scene.parquet``);
* optimised cross-spectral camera poses written by the trainer as
  ``<modal>_iter_<iteration:06d>_r_.npy`` (quaternion, shape (N, 1, 4)) and
  ``<modal>_iter_<iteration:06d>_t_.npy`` (translation, shape (N, 1, 3)),
  indexed by the absolute ``camera_id`` of the training json.

``modal`` is ``ms`` in the bimodal experiment and ``ms``/``ir`` in the
trimodal experiment.
"""

import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch

#: view labels used for the validation images of every scene (see README)
DEFAULT_TEST_IDS = ['8', '10', '14', '20', '22']

GPU_ID = 0
_nvml_available = None


def _nvml():
    """Lazy pynvml initialisation (used for the GPU memory measurement)."""
    global _nvml_available
    if _nvml_available is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _nvml_available = pynvml
        except Exception:  # noqa: BLE001 - memory reporting is optional
            _nvml_available = False
    return _nvml_available


def get_gpu_mem() -> float:
    """Currently used GPU memory in MB (0.0 if nvidia-ml is unavailable)."""
    pynvml = _nvml()
    if not pynvml:
        return 0.0
    handler = pynvml.nvmlDeviceGetHandleByIndex(GPU_ID)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    return round(meminfo.used / 1024 / 1024, 2)


# --------------------------------------------------------------------- #
# learned-pose files
# --------------------------------------------------------------------- #
def list_pose_iterations(log_dir: str, modal: str) -> List[int]:
    """Return the iterations for which ``<modal>_iter_*_r_.npy`` exists."""
    iterations = []
    pattern = re.compile(rf"^{modal}_iter_(\d+)_r_\.npy$")
    if os.path.isdir(log_dir):
        for name in os.listdir(log_dir):
            match = pattern.match(name)
            if match:
                iterations.append(int(match.group(1)))
    return sorted(iterations)


def load_learned_poses(log_dir: str, modal: str,
                       iteration: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load the optimised ``modal`` poses of the latest (or given) iteration.

    Returns ``(q, t)`` numpy arrays of shape (num_cameras, 1, 4) and
    (num_cameras, 1, 3), indexed by absolute camera id.
    """
    iterations = list_pose_iterations(log_dir, modal)
    if not iterations:
        raise FileNotFoundError(
            f"no '{modal}_iter_*_r_.npy' pose files found under {log_dir!r} "
            f"(train the scene first, see README)")
    selected = iteration if iteration is not None else iterations[-1]
    if selected not in iterations:
        raise FileNotFoundError(
            f"pose iteration {selected} not found under {log_dir!r}; "
            f"available iterations: {iterations}")
    q_path = os.path.join(log_dir, f"{modal}_iter_{selected:06d}_r_.npy")
    t_path = os.path.join(log_dir, f"{modal}_iter_{selected:06d}_t_.npy")
    return np.load(q_path), np.load(t_path)


def learned_poses_to_cameras(q_np, t_np, camera_ids) -> torch.Tensor:
    """Build the (V, 4, 4) camera matrices of the learned poses for given ids."""
    from .utils import quaternion_to_rotation_matrix_torch

    num_cameras = len(camera_ids)
    cameras = torch.zeros((num_cameras, 4, 4))
    for idx, camera_id in enumerate(camera_ids):
        q = torch.tensor(q_np[camera_id])
        t = torch.tensor(t_np[camera_id])
        cameras[idx, :3, :3] = quaternion_to_rotation_matrix_torch(q)
        cameras[idx, :3, 3] = t
        cameras[idx, 3, 3] = 1.0
    return cameras
