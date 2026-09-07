"""Export a trained SOC-GS scene (parquet checkpoint) to a PLY point cloud.

The number of spectral channels (bimodal 4-channel or trimodal 5-channel
checkpoints) is detected automatically from the file, so checkpoints produced
by both experiments of the original release are supported.

Usage:
    python parquet_to_ply.py --parquet_path logs/rgb_ir_ms/orange/best_scene.parquet \
                             --ply_path output/orange.ply
"""

import argparse

from socgs.common.GaussianPointCloudScene import GaussianPointCloudScene
from socgs.registry import experiment_from_parquet  # noqa: F401 (validates the file)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet_path", type=str, required=True,
                        help="trained scene checkpoint (.parquet) written by to_parquet()")
    parser.add_argument("--ply_path", type=str, required=True,
                        help="output .ply file path")
    args = parser.parse_args()

    scene = GaussianPointCloudScene.from_trained_parquet(
        args.parquet_path,
        config=GaussianPointCloudScene.PointCloudSceneConfig(
            max_num_points_ratio=None, add_sphere=False))
    scene.to_ply(args.ply_path)
    num_points = int((scene.point_invalid_mask == 0).sum())
    print(f"[parquet_to_ply] exported {num_points} points to {args.ply_path}")


if __name__ == "__main__":
    main()
