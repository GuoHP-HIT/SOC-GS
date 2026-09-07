"""Image-quality metrics (PSNR / SSIM) used by SOC-GS training and evaluation.

Two families are provided:

* MATLAB-compatible metrics operating on numpy arrays in the [0, 255] range
  (``calculate_psnr`` / ``calculate_ssim`` / ``ssim``). These are the metrics
  reported for the paper's quantitative tables and are computed by the
  evaluation scripts over rendered vs. ground-truth images.

* Torch metrics on [0, 1] tensors (``psnr_torch`` / ``ssim_torch`` /
  ``compute_psnr_ssim_torch``) used for the per-iteration validation during
  training (identical to what the original trainer logged).

The CV2-based SSIM uses an 11x11 Gaussian window with sigma 1.5 on the valid
(central) image region, which matches the reference MATLAB implementation.
"""

import cv2
import numpy as np
import torch


# --------------------------------------------------------------------- #
# MATLAB-compatible metrics for numpy images in the [0, 255] range
# --------------------------------------------------------------------- #
def calculate_psnr(img1, img2):
    """PSNR of two images with range [0, 255] (same as MATLAB's)."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_rgb_psnr(img1, img2):
    """Calculate PSNR channel-wise and average; images have range [0, 255]."""
    n_channels = np.ndim(img1)
    sum_psnr = 0
    for i in range(n_channels):
        this_psnr = calculate_psnr(img1[:, :, i], img2[:, :, i])
        sum_psnr += this_psnr
    return sum_psnr / n_channels


def ssim(img1, img2):
    """SSIM of two single-channel images with range [0, 255] (MATLAB-like)."""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """Calculate SSIM; same outputs as MATLAB's. Images have range [0, 255].

    Accepts 2D (single channel) or 3D (HxWx1 / HxWx3) arrays.
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(img1.shape[2]):
                ssims.append(ssim(img1[..., i], img2[..., i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


# --------------------------------------------------------------------- #
# Torch metrics on [0, 1] images used during training validation
# --------------------------------------------------------------------- #
def psnr_torch(image_pred, image_gt):
    """PSNR (dB) of two [0, 1] torch tensors, as used in the trainer."""
    return 10 * torch.log10(1.0 / torch.mean((image_pred - image_gt) ** 2))


def ssim_torch(image_pred, image_gt):
    """MS-SSIM-style SSIM from pytorch-msssim, as used in the trainer."""
    from pytorch_msssim import ssim  # optional import: not needed for pure eval
    return ssim(image_pred.unsqueeze(0), image_gt.unsqueeze(0),
                data_range=1.0, size_average=True)


def compute_psnr_ssim_torch(image_pred, image_gt):
    """(PSNR, SSIM) pair in the same way the original scripts computed it."""
    with torch.no_grad():
        psnr_score = psnr_torch(image_pred, image_gt)
        image_pred_np = image_pred.permute(1, 2, 0).cpu().detach().numpy()
        image_gt_np = image_gt.permute(1, 2, 0).cpu().detach().numpy()
        ssim_score = ssim_numpy_01(image_pred_np, image_gt_np)
        return psnr_score, ssim_score


def ssim_numpy_01(img1, img2):
    """SSIM on float [0, 1] HxWxC numpy images (cv2 window, MATLAB-like)."""
    return calculate_ssim(img1 * 255.0, img2 * 255.0)
