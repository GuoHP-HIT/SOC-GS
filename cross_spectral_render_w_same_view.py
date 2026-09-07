"""Same-view cross-spectral rendering & evaluation (entry point, original CLI).

Renders the validation views of a trained scene from the RGB camera poses
(COLMAP) or from the optimised multispectral/infrared poses and saves the
predicted RGB/MS(/IR) channels plus the ground-truth views, then reports the
average PSNR/SSIM of the chosen view and the average rendering time/GPU memory.

The experiment (bimodal ``rgb_ms`` or trimodal ``rgb_ir_ms``) is detected
automatically from the checkpoint stored in ``<log_dir>/<scene>/``.

Usage:
    python cross_spectral_render_w_same_view.py --scene orange --render_view ms \
        --log_dir logs/rgb_ir_ms --output_prefix render --poses_dir datasets
    python cross_spectral_render_w_same_view.py --scene hall2 --render_view rgb \
        --log_dir logs/rgb_ms --output_prefix render --poses_dir datasets
"""

import argparse
import os

from socgs.registry import (
    EXPERIMENTS,
    experiment_from_parquet,
    import_experiment,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=str, default="bluechair")
    parser.add_argument("--render_view", type=str, default="rgb",
                        help="camera set used for the evaluation: 'rgb' (COLMAP poses), "
                             "'ms' (learned multispectral poses) or - trimodal only - 'ir'")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="directory holding <scene>/best_scene.parquet and poses")
    parser.add_argument("--output_prefix", type=str, default="./render")
    parser.add_argument("--poses_dir", type=str, default="./datasets",
                        help="directory holding the RGB pose json files")
    parser.add_argument("--parquet_path", type=str, default=None,
                        help="explicit checkpoint; otherwise <log_dir>/<scene>/best_scene.parquet")
    parser.add_argument("--experiment", type=str, default=None, choices=EXPERIMENTS,
                        help="override the experiment (auto-detected from the checkpoint)")
    parser.add_argument("--pose_iteration", type=int, default=None,
                        help="use the learned poses of this training iteration "
                             "(default: the latest snapshot in the log directory)")
    parser.add_argument("--val_ids", type=str, default="8,10,14,20,22",
                        help="view labels used for the output file names")
    parser.add_argument("--taichi_memory_gb", type=float, default=4.0)
    args = parser.parse_args()

    parquet_path = args.parquet_path or os.path.join(
        args.log_dir, args.scene, "best_scene.parquet")
    experiment = args.experiment or experiment_from_parquet(parquet_path)

    render_module = import_experiment(experiment, "render")
    render_module.render_same_view_evaluation(
        scene=args.scene,
        render_view=args.render_view,
        log_dir=args.log_dir,
        output_prefix=args.output_prefix,
        poses_dir=args.poses_dir,
        parquet_path=args.parquet_path,
        pose_iteration=args.pose_iteration,
        val_ids=[v.strip() for v in args.val_ids.split(",") if v.strip()],
        taichi_memory_gb=args.taichi_memory_gb,
    )


if __name__ == "__main__":
    main()
