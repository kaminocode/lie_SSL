"""Per-image rendering metrics on unit-range grayscale images, without extras."""

import torch
from torch.nn import functional as F

METRIC_SETTINGS = {
    "data_range": 1.0, "aggregation": "mean over images and both directions",
    "psnr_mse_floor": 1e-12, "ssim_window": 11, "ssim_sigma": 1.5,
    "ssim_k1": 0.01, "ssim_k2": 0.03, "ssim_padding": "valid",
    "ssim_covariance": "population", "ssim_channels": "grayscale",
}


def image_metrics(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 4 or predicted.shape[1] != 1:
        raise ValueError("expected matching [batch, 1, height, width] grayscale images")
    if min(predicted.shape[-2:]) < 11:
        raise ValueError("SSIM requires images at least 11x11")
    predicted, target = predicted.float(), target.float()
    mse = (predicted - target).square().flatten(1).mean(1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    coordinates = torch.arange(11, device=predicted.device, dtype=predicted.dtype) - 5
    gaussian = torch.exp(-coordinates.square() / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :])[None, None]
    moments = F.conv2d(torch.cat((predicted, target, predicted.square(), target.square(), predicted * target)), kernel)
    mu_x, mu_y, exx, eyy, exy = moments.chunk(5)
    var_x = (exx - mu_x.square()).clamp_min(0)
    var_y = (eyy - mu_y.square()).clamp_min(0)
    covariance = exy - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mu_x.square() + mu_y.square() + 0.01**2) * (var_x + var_y + 0.03**2)
    )
    return {"mse": mse, "psnr": psnr, "ssim": ssim_map.flatten(1).mean(1)}
