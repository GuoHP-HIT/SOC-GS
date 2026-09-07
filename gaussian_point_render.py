"""Render a trained scene from validation poses (entry point, original CLI).

Kept CLI-compatible with the original ``gaussian_point_render.py`` (the legacy
RGB frame renderer): renders every validation view of the chosen camera
modality and saves the frames as ``frame_000.png, frame_001, ...``; with
``--gt_prefix`` the corresponding ground-truth views are saved as well, and
average PSNR/SSIM are reported for the chosen view.

Usage:
    python gaussian_point_render.py --parquet_path logs/rgb_ir_ms/orange/best_scene.parquet \
        --poses datasets/orange/val.json --output_prefix result/orange --render_view ms
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
    parser.add_argument("--parquet_path", type=str, required=True,
                        help="trained scene checkpoint (best/final/scene_<iter>.parquet)")
    parser.add_argument("--poses", type=str, required=True,
                        help="validation json with RGB poses (datasets/<scene>/val.json)")
    parser.add_argument("--output_prefix", type=str, required=True,
                        help="directory for the rendered frames (rgb/, ms/, ir/ sub-dirs)")
    parser.add_argument("--gt_prefix", type=str, default=None,
                        help="optional directory for the ground-truth frames of --render_view")
    parser.add_argument("--pose_path", type=str, default=None,
                        help="log directory of the scene with the learned poses "
                             "(defaults to the parent of --parquet_path)")
    parser.add_argument("--render_view", type=str, default="rgb",
                        help="camera modality to render: 'rgb' (json poses), 'ms' (learned "
                             "poses) or - trimodal only - 'ir'")
    parser.add_argument("--experiment", type=str, default=None, choices=EXPERIMENTS,
                        help="override the experiment (auto-detected from the checkpoint)")
    parser.add_argument("--pose_iteration", type=int, default=None,
                        help="iteration of the learned pose snapshots (default: latest)")
    parser.add_argument("--taichi_memory_gb", type=float, default=4.0)
    args = parser.parse_args()

    if not os.path.exists(args.parquet_path):
        raise FileNotFoundError(f"checkpoint not found: {args.parquet_path}")
    experiment = args.experiment or experiment_from_parquet(args.parquet_path)
    scene = os.path.basename(os.path.dirname(args.parquet_path))
    log_dir = os.path.dirname(os.path.dirname(args.parquet_path))
    poses_dir = os.path.dirname(args.poses)

    render_module = import_experiment(experiment, "render")
    render_module.render_same_view_evaluation(
        scene=scene,
        render_view=args.render_view,
        log_dir=log_dir,
        output_prefix=args.output_prefix,
        poses_dir=poses_dir,
        parquet_path=args.parquet_path,
        pose_iteration=args.pose_iteration,
        val_ids=[],
        taichi_memory_gb=args.taichi_memory_gb,
        frame_mode='frames',
        gt_prefix=args.gt_prefix,
    )


if __name__ == "__main__":
    main()
