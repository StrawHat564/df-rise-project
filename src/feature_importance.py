"""Novelty: feature-absence DF-RISE.

Extends the spatial DF-RISE with perceptual feature masks (colour / sharpness /
texture) so saliency becomes *feature-resolved*. Design rationale documented in
`p4_research_notes.md` §F.17: the paper's rejection of luminance/contrast in the
*similarity* function does not preclude colour-important *perturbations* — but
the scorer must be consistent (be colour-sensitive) for the measure to work.

Two granularities produced here:
1. global per-feature-per-step curve (ablate whole latent's feature, measure Δ)
2. spatial per-feature map (spatial mask UNION feature ablation)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Feature operators. Operate on decoded RGB so the notion of "colour / texture"
# is meaningful (on the raw latent "colour" is ill-defined, per F.17).
# ---------------------------------------------------------------------------

def desaturate_rgb(img: torch.Tensor) -> torch.Tensor:
    """Colour-absence: convert RGB to luminance and replicate.

    img: (B,3,H,W) in [0,1]. Removes chroma, keeps structure.
    """
    gray = img[:, 0:1] * 0.299 + img[:, 1:2] * 0.587 + img[:, 2:3] * 0.114
    return gray.repeat(1, 3, 1, 1)


def blur_gaussian(img: torch.Tensor, kernel_size: int = 15, sigma: float = 5.0) -> torch.Tensor:
    """Sharpness-absence: low-pass filter (kills high-frequency detail)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = torch.arange(kernel_size, device=img.device).float() - kernel_size // 2
    k1d = torch.exp(-(x**2) / (2 * sigma**2))
    k1d = k1d / k1d.sum()
    kernel = (k1d[:, None] * k1d[None, :])[None, None].repeat(3, 1, 1, 1)
    return F.conv2d(F.pad(img, [kernel_size // 2] * 4, mode="replicate"), kernel, groups=3)


def texture_smooth(img: torch.Tensor) -> torch.Tensor:
    """Texture-absence: strong local averaging (removes fine pattern energy)."""
    return blur_gaussian(img, kernel_size=9, sigma=3.0)


# ---------------------------------------------------------------------------
# Feature-aware DF-RISE core.
# ---------------------------------------------------------------------------

class FeatureDFRISE:
    """Feature-resolved attribution over a denoising trajectory.

    Args:
        vae_decoder: callable(latent) -> (B,3,H,W) RGB in [0,1].
        features: subset of {'colour','sharpness','texture'}.
    """

    def __init__(self, vae_decoder, features=("colour", "sharpness", "texture")):
        self.decoder = vae_decoder
        self.features = features
        self.ops = {
            "colour": desaturate_rgb,
            "sharpness": blur_gaussian,
            "texture": texture_smooth,
        }

    def _feature_similarity(self, base: torch.Tensor, ablated: torch.Tensor,
                            feature: str) -> torch.Tensor:
        """Consistent scorer per feature (must "see" the ablated feature)."""
        if feature == "colour":
            # full SSIM / perceptual distance so chroma removal registers
            return self._full_ssim(base, ablated)
        # sharpness/texture: structure term is sufficient (high-freq change)
        return self._structure_sim(base, ablated)

    @staticmethod
    def _structure_sim(a, b):
        """Approximate SSIM-structure on small tensors (placeholder)."""
        a, b = a - a.mean(), b - b.mean()
        denom = (a.std() * b.std()).clamp(min=1e-8)
        return (a * b).mean() / denom

    @staticmethod
    def _full_ssim(a, b):
        """Full SSIM-like including chroma sensitivity (placeholder)."""
        return 1.0 - torch.nn.functional.l1_loss(a, b)

    def score_step(self, latent: torch.Tensor) -> dict[str, float]:
        """Decode, ablate each feature, score → {feature: importance}."""
        base = self.decoder(latent)
        out = {}
        for feat in self.features:
            ablated = self.ops[feat](base)
            out[feat] = float(self._feature_similarity(base, ablated, feat))
        return out