#!/usr/bin/env bash
# Train every scene of the bimodal (RGB+MS) experiment.
# Run from the repository root; requires datasets/ to be prepared (see README).
set -e

SCENES=(bins black bluechair cvlab dino fruits green hall hall2 hall3 hall4 \
        orange penguin penguin2 puppets tech)

for scene in "${SCENES[@]}"; do
    python gaussian_point_train.py \
        --train_config "config/rgb_ms/${scene}_train.yaml"
done
