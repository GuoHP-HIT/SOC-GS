"""Selecting the experiment implementation.

SOC-GS releases two pipelines in one package:

* ``rgb_ms``    -- bimodal  RGB + multispectral  (4 spectral channels, 72 features);
* ``rgb_ir_ms`` -- trimodal RGB + infrared + MS  (5 spectral channels, 88 features).

The two pipelines share their scene representation and differ only in the
model stack (rasteriser / controller / trainer / dataset). Entry scripts
resolve the experiment either from the yaml path (``config/rgb_ms/...``,
``config/rgb_ir_ms/...``) or - when that is ambiguous - from the number of
feature columns stored in a checkpoint/dataset file.
"""

import importlib
import os
import sys

EXPERIMENTS = ("rgb_ms", "rgb_ir_ms")
#: feature count used by the bimodal experiment (8 + 4*16 SH coefficients)
N_FEATURES_RGB_MS = 72
#: feature count used by the trimodal experiment (8 + 5*16 SH coefficients)
N_FEATURES_RGB_IR_MS = 88

_SYS_PATH_INSERTED = False


def _ensure_importable():
    global _SYS_PATH_INSERTED
    if not _SYS_PATH_INSERTED:
        # allow running the top-level entry scripts from the repository root
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        _SYS_PATH_INSERTED = True


def experiment_from_features(num_features: int) -> str:
    """Map a feature count to the experiment implementing it."""
    if num_features == N_FEATURES_RGB_MS:
        return "rgb_ms"
    if num_features == N_FEATURES_RGB_IR_MS:
        return "rgb_ir_ms"
    raise ValueError(
        f"unsupported number of features {num_features}: SOC-GS checkpoints use "
        f"{N_FEATURES_RGB_MS} (rgb_ms) or {N_FEATURES_RGB_IR_MS} (rgb_ir_ms) features")


def experiment_from_config_path(train_config_path: str) -> str:
    """Infer the experiment from the config path or its num-of-features value."""
    normalized = train_config_path.replace("\\", "/")
    parent = os.path.basename(os.path.dirname(normalized))
    if parent in EXPERIMENTS:
        return parent
    try:
        import yaml
        with open(train_config_path, "r") as f:
            data = yaml.safe_load(f)
        n_features = (data.get("gaussian-point-cloud-scene-config") or {}).get("num-of-features")
        if n_features is not None:
            return experiment_from_features(int(n_features))
    except Exception:  # noqa: BLE001 - fall through to the default below
        pass
    return "rgb_ms"


def _num_features_of_parquet(path: str) -> int:
    """Feature count encoded in a SOC-GS scene parquet (``alpha0`` + SH columns)."""
    import pandas as pd

    df = pd.read_parquet(path)
    columns = set(df.columns)
    base = {"cov_q0", "cov_q1", "cov_q2", "cov_q3", "cov_s0", "cov_s1", "cov_s2", "alpha0"}
    if not base.issubset(columns):
        raise ValueError(
            f"{path} does not look like a SOC-GS scene checkpoint "
            f"(missing {base - columns or 'expected columns'})")
    channels = {"r": 16, "g": 16, "b": 16, "ms": 16, "ir": 16}
    num_channels = 0
    for name, width in channels.items():
        expected = {f"{name}_sh{i}" for i in range(width)}
        if expected.issubset(columns):
            num_channels += 1
        elif any(c.startswith(f"{name}_sh") for c in columns):
            raise ValueError(f"{path} has a partial column set for channel {name!r}")
    return 8 + 16 * num_channels


def experiment_from_parquet(parquet_path: str) -> str:
    """Infer the experiment from the feature columns of a saved scene.

    Used by the rendering/evaluation tools so that bimodal and trimodal
    checkpoints of the original release are handled transparently.
    """
    return experiment_from_features(_num_features_of_parquet(parquet_path))


def import_experiment(experiment: str, module: str):
    """Import ``module`` from the experiment's package.

    Example: ``import_experiment('rgb_ms', 'GaussianPointTrainer')`` returns
    the module ``socgs.rgb_ms.GaussianPointTrainer``.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r} (expected {EXPERIMENTS})")
    _ensure_importable()
    return importlib.import_module(f"socgs.{experiment}.{module}")
