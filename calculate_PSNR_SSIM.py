"""PSNR/SSIM metrics for rendered image pairs (entry point).

Computes MATLAB-compatible PSNR and SSIM (see ``socgs.common.metrics``) for
every image pair in two directories (same file names) and prints the average.
Typically used on the output of ``cross_spectral_render_w_same_view.py``:

    python calculate_PSNR_SSIM.py --pred_dir render/orange/rgb_view/rgb \
                                  --gt_dir render/orange/rgb_view/GT
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from socgs.common.metrics import calculate_psnr, calculate_ssim  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="directory with the rendered (predicted) images")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="directory with the ground-truth images")
    parser.add_argument("--suffix", type=str, default=".png")
    args = parser.parse_args()

    pred_names = sorted(name for name in os.listdir(args.pred_dir)
                        if name.endswith(args.suffix))
    gt_names = sorted(name for name in os.listdir(args.gt_dir)
                      if name.endswith(args.suffix))
    assert len(pred_names) == len(gt_names) > 0, \
        f"image lists do not match: {len(pred_names)} preds vs {len(gt_names)} GTs"
    # pair by sorted order (file names are identical when produced by the
    # same-view renderer, so sorted order pairs them correctly)
    psnr_list, ssim_list = [], []
    for pred_name, gt_name in zip(pred_names, gt_names):
        pred = cv2.imread(os.path.join(args.pred_dir, pred_name))
        gt = cv2.imread(os.path.join(args.gt_dir, gt_name))
        if pred is None or gt is None:
            raise FileNotFoundError(f"cannot read pair {pred_name} / {gt_name}")
        psnr = calculate_psnr(gt.astype(np.float64), pred.astype(np.float64))
        ssim = calculate_ssim(gt.astype(np.float64), pred.astype(np.float64))
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print(f"{pred_name}: PSNR {psnr:.6f} dB, SSIM {ssim:.6f}")
    print("Average: PSNR {:.6f} dB, SSIM {:.6f}".format(
        float(np.mean(psnr_list)), float(np.mean(ssim_list))))


if __name__ == "__main__":
    main()
