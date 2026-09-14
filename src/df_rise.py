"""DF-RISE: Diffusion Randomized Input Sampling Explanation (Park et al. 2024).

External (black-box) saliency for a diffusion U-Net at each denoising step.

Key design points from the paper:
- Perturb the LATENT `R_t` (U-Net input), not the raw image.
- Masks are per-pixel, Gaussian-thresholded to binary, covering the whole
  latent (unlike RISE's object-focused input-grid masks).
- Similarity is the SSIM **structure** term only (luminance/contrast rejected
  because the compared objects are noise predictions).
- Accumulate over N masks: S = Σ_n  M_n ⊙ s(f(R_t ⊙ M_n), f(R_t)), then min-max.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_binary_masks(n: int, h: int, w: int, threshold: float | None = None,
                          device: str = "cuda") -> torch.Tensor:
    """Sample N binary masks (n,1,h,w) by thresholding Gaussian noise.

    The paper samples a Gaussian value per pixel element and thresholds it:
    value >= threshold -> 1 (keep), value < threshold -> 0 (mask out).
    Vary threshold per mask so across N draws every pixel config is covered.
    """
    g = torch.randn(n, 1, h, w, device=device)
    if threshold is None:
        # random threshold in [0,1] per mask gives varied sparsity patterns
        thresh = torch.rand(n, 1, 1, 1, device=device)
    else:
        thresh = torch.tensor(float(threshold), device=device)
    return (g >= thresh).float()


def structure_similarity(a: torch.Tensor, b: torch.Tensor,
                         window_size: int = 7,
                         C3: float = 1e-4) -> torch.Tensor:
    """SSIM structure term s(a,b) = (σ_ab + C3)/(σ_a·σ_b + C3).

    Args:
        a, b: (B, C, H, W) latents to compare.
        window_size: local window for computing local stats.
    Returns:
        (B, C, H, W) local structure-similarity in texturing order.
    """
    a = a.float()
    b = b.float()
    # local means
    kernel = torch.ones((1, 1, window_size, window_size),
                        device=a.device, dtype=a.dtype)
    kernel /= kernel.sum()
    pad = window_size // 2

    def avg(x):
        x = F.pad(x, [pad] * 4, mode="replicate")
        return F.conv2d(x, kernel, padding=0)

    mu_a, mu_b = avg(a), avg(b)
    mu_a2, mu_b2 = mu_a.pow(2), mu_b.pow(2)
    mu_ab = mu_a * mu_b

    # variances & covariance (unbiased ~ with N-1 inside window)
    sigma_a2 = avg(a.pow(2)) - mu_a2
    sigma_b2 = avg(b.pow(2)) - mu_b2
    sigma_ab = avg(a * b) - mu_ab

    # guard against trivial zero-variance regions
    denom = sigma_a2 * sigma_b2
    denom = denom.clamp(min=1e-8).sqrt()
    return (sigma_ab + C3) / (denom + C3)


@torch.no_grad()
def df_rise_step(
    unet,
    vae,
    text_emb,           # (2, 77, 768) uncond + cond
    latent,             # (1, 4, 64, 64) current x_t
    t: int,
    n_masks: int = 100,
    guidance_scale: float = 7.5,
    window_size: int = 7,
    device: str = "cuda",
    mask_threshold: float | None = None,
) -> torch.Tensor:
    """Compute a DF-RISE saliency map for a single denoising step t.

    Returns:
        S: (h, w) normalized saliency map in [0, 1].
    """
    h, w = latent.shape[-2:]
    masks = gaussian_binary_masks(n_masks, h, w, mask_threshold, device)

    # The "unperturbed output" f(R_t): noise prediction on the vanilla latent.
    # Note: CFG doubles channel count, so we use masked/vanilla latents per pass.
    def predict(z):
        zz = torch.cat([z] * 2)
        pred = unet(zz, torch.as_tensor([t] * 2, device=device), text_emb).sample
        u, c = pred.chunk(2)
        return u + guidance_scale * (c - u)

    f_orig = predict(latent)

    acc = torch.zeros((h, w), device=device, dtype=torch.float32)
    for i in range(n_masks):
        m = masks[i].to(dtype=latent.dtype, device=device)   # match latent (fp16)
        perturbed = latent * m
        f_masked = predict(perturbed)
        # structure similarity between predicted-noises (on latent channel 0)
        s = structure_similarity(f_orig, f_masked, window_size)
        acc += (m[0, 0] * s[0, 0])         # weighted mask accumulation

    S = acc / n_masks
    S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    return S