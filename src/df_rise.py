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
from kornia.filters import get_gaussian_kernel2d


def binary_masks(n: int, h: int, w: int, prob_thresh: float = 0.5,
                 device: str = "cuda", seed: int | None = None) -> torch.Tensor:
    """Sample N binary masks (n,1,h,w), each latent element kept i.i.d. w.p. prob_thresh.

    Matches the reference implementation (X-Diffusion, models/stablediffusion.py):
    ``torch.rand(n, 1, h, w, device=device) < prob_thresh``.

    Note: thresholding zero-mean Gaussian noise at 0.0 is distributionally
    identical (also Bernoulli(0.5)), but prob_thresh is an explicit *keep
    probability* -- do not read it as a Gaussian threshold level.

    seed: if None, draws from the global RNG (fresh, non-reproducible masks each
          call -> "refresh"). If an int, uses a local generator seeded with it,
          so the same seed reproduces the same masks.
    """
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    r = torch.rand(n, 1, h, w, device=device, generator=gen)
    return (r < float(prob_thresh)).float()


def structure_similarity(a: torch.Tensor, b: torch.Tensor,
                         window_size: int = 15,
                         C3: float = 4.5e-4) -> torch.Tensor:
    """SSIM structure term s(a,b) = (2*σ_ab + C3)/(σ_a·σ_b + C3).

    This is the `mode="structure"` variant of the reference ssim.py: the
    luminance and contrast terms are deliberately dropped, so identical inputs
    give s = 2 (not 1). That is expected, not a bug. C3 = 0.03**2 / 2 = 4.5e-4.

    Args:
        a, b: (B, C, H, W) latents to compare.
        window_size: local window for computing local stats.
    Returns:
        (B, C, H, W) local structure-similarity in texturing order.
    """
    a = a.float()
    b = b.float()
    # 'same' padding: the 15x15 window needs 7px of padding to keep 64x64.
    pad = (window_size - 1) // 2

    # per-channel, group convolution: one kernel per channel, depthwise
    c = a.shape[1]
    # kernel = torch.ones((c, 1, window_size, window_size),
    #                     device=a.device, dtype=a.dtype)
    # kernel = kernel / window_size**2
    kernel = get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5))
    kernel = kernel.repeat(c, 1, 1, 1).to(device=a.device, dtype=a.dtype)

    def avg(x):
        # Zero padding of (window_size-1)//2 keeps the spatial size unchanged
        # (matches the reference ssim.py compute_zero_padding).
        return F.conv2d(x, kernel, padding=pad, groups=c)

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
    ssim = (2 * sigma_ab + C3) / (denom + C3)
    return ssim.mean(1)


@torch.no_grad()
def df_rise_step(
    unet,
    text_emb,           # (2, 77, 768) uncond + cond (concatenated on dim 0)
    latent,             # (1, 4, 64, 64) current x_t
    t: int,
    target,             # (1, 4, 64, 64) x_{t-1} to compare against
    n_masks: int = 500,
    guidance_scale: float = 7.5,
    window_size: int = 15,
    device: str | torch.device | None = None,
    prob_thresh: float = 0.5,
    seed: int | None = None,
) -> torch.Tensor:
    """Compute a DF-RISE saliency map for a single denoising step t.

    target must be the *post-step* latent x_{t-1} (i.e. the scheduler's
    prev_sample for this step), NOT the noise prediction. This is what the
    reference implementation compares against; comparing two noise predictions
    instead is what produced the streaky, degenerate maps.
    device defaults to the latent's own device (auto-follows the pipeline).
    prob_thresh: per-element mask keep probability (paper default 0.5).
    seed: reproducibility for the mask draws. If None, fresh masks each call
        (refresh). If an int, the masks for this step are seeded with
        seed + t so different steps still get different masks, while the same
        (seed, t) always reproduces the same map.
    Returns:
        S: (h, w) raw accumulated saliency (NOT normalized).
    """
    h, w = latent.shape[-2:]
    device = device if device is not None else latent.device
    # The UNet weights define the canonical dtype; force everything to match it
    # so fp16 pipelines don't crash on fp32 latents/text_emb (dtype drift).
    tdtype = next(unet.parameters()).dtype
    text_emb = text_emb.to(dtype=tdtype)
    latent = latent.to(dtype=tdtype)
    target = target.to(dtype=tdtype)
    masks = binary_masks(
        n_masks, h, w, prob_thresh, device,
        seed=None if seed is None else seed + int(t),
    ).to(dtype=tdtype)

    # The masked prediction IS classifier-free-guided (matching the reference,
    # which concatenates the masked latent twice and combines the two halves).
    # Only the *target* is a latent rather than a prediction.
    # CFG doubles the BATCH dim (one uncond + one cond copy), so the masked
    # latent is duplicated and the halves combined inside predict().
    def predict(z):
        zz = torch.cat([z] * 2)
        pred = unet(zz, torch.as_tensor([t] * 2, device=device), text_emb).sample
        u, c = pred.chunk(2)
        return u + guidance_scale * (c - u)

    acc = torch.zeros((h, w), device=device, dtype=torch.float32)
    for i in range(n_masks):
        m = masks[i].to(dtype=latent.dtype, device=device)   # match latent (fp16)
        f_masked = predict(latent * m)
        # structure similarity between predicted noised images, averaged over
        # latent channels; weight by mask as in RISE.
        s = structure_similarity(target, f_masked, window_size)
        acc += (m[0, 0] * s[0])    # s is (B,H,W) -> take the single batch elem

    return acc
