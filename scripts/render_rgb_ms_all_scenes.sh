#!/usr/bin/env bash
# Same-view cross-spectral rendering + evaluation + render-cost measurement
# for the bimodal (RGB+MS) experiment (equivalent to the original
# script_render_and_cost_calculate.sh). Run from the repository root after
# training (outputs land in ./render/<scene>/<view>_view/{rgb,ms,GT}).
set -e

SCENES=(bins black bluechair cvlab dino fruits green hall hall2 hall3 hall4 \
        orange penguin penguin2 puppets tech)

for scene in "${SCENES[@]}"; do
    python cross_spectral_render_w_same_view.py \
        --log_dir logs/rgb_ms --poses_dir datasets --scene "$scene" --render_view rgb
    python cross_spectral_render_w_same_view.py \
        --log_dir logs/rgb_ms --poses_dir datasets --scene "$scene" --render_view ms
done
