"""Train a SOC-GS Gaussian point-cloud scene.

Entry point of the original release, kept CLI-compatible:
``python gaussian_point_train.py --train_config config/<experiment>/<scene>_train.yaml``

The experiment (``rgb_ms`` bimodal RGB+MS or ``rgb_ir_ms`` trimodal RGB+IR+MS)
is selected automatically from the config path; pass ``--experiment`` to
override, or ``--gen_template_only`` to dump a default TrainConfig template.
"""

import argparse
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from socgs.registry import (  # noqa: E402
    EXPERIMENTS,
    experiment_from_config_path,
    import_experiment,
)


def main():
    parser = argparse.ArgumentParser("Train a Gaussian Point Cloud Scene")
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None, choices=EXPERIMENTS,
                        help="override experiment (auto-detected from the config path)")
    parser.add_argument("--gen_template_only", action="store_true", default=False)
    args = parser.parse_args()

    experiment = args.experiment or experiment_from_config_path(args.train_config)
    print(f"[SOC-GS] experiment: {experiment}")

    trainer_module = import_experiment(experiment, "GaussianPointTrainer")
    GaussianPointCloudTrainer = trainer_module.GaussianPointCloudTrainer
    TrainConfig = GaussianPointCloudTrainer.TrainConfig

    if args.gen_template_only:
        TrainConfig().to_yaml_file(args.train_config)
        return

    config = TrainConfig.from_yaml_file(args.train_config)
    trainer = GaussianPointCloudTrainer(config)
    trainer.train()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    main()
