"""Visualization helpers: heatmaps, step series, curves."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def _upsample(saliency: np.ndarray, size) -> np.ndarray:
    """Bilinear-upscale a (H,W) saliency map to `size` (pixel-res for overlay)."""
    from torchvision.transforms import functional as F
    import torch
    s = torch.from_numpy(np.asarray(saliency, dtype=np.float32))[None, None]
    s = F.interpolate(s, size=size, mode="bilinear", align_corners=False)
    return s[0, 0].numpy()


def _ensure_dir(path: str):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def heatmap_overlay(image: np.ndarray, saliency: np.ndarray,
                    alpha: float = 0.5, cmap: str = "jet") -> np.ndarray:
    """Blend saliency onto image (any resolution). image (H,W,3)[0,1]; saliency (H,W)."""
    if saliency.shape[:2] != image.shape[:2]:
        saliency = _upsample(saliency, image.shape[:2])
    norm = Normalize(vmin=0, vmax=max(1e-6, float(saliency.max())))
    cm = plt.get_cmap(cmap)(norm(saliency))[..., :3]  # (H,W,3)
    return (1 - alpha) * image + alpha * cm


def plot_step_series(saliency_maps, timesteps, save_path=None,
                     ncols: int = 4, upscale: int = 256):
    """Grid of heatmaps across denoising steps (upscaled for legibility)."""
    n = len(saliency_maps)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax, sm, t in zip(axes, saliency_maps, timesteps):
        sm = _upsample(sm, (upscale, upscale))
        ax.imshow(sm, cmap="jet", aspect="auto")
        ax.set_title(f"t={t}")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_deletion_insertion(results: dict, save_path=None):
    """results: {method: {'deletion': curve, 'insertion': curve, 'fractions': f}}"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method, d in results.items():
        for ax, key, title in ((axes[0], "deletion", "Deletion game"),
                               (axes[1], "insertion", "Insertion game")):
            ax.plot(d["fractions"], d[key], label=method, marker="o")
            ax.set_title(title)
            ax.set_xlabel("Ratio of pixels")
            ax.set_ylabel("Score")
            ax.grid(alpha=0.3)
    axes[0].legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_curves(feature_curves: dict, timesteps, save_path=None):
    """feature_curves: {feature_name: [importance per step]}"""
    fig, ax = plt.subplots(figsize=(8, 4))
    for feat, vals in feature_curves.items():
        ax.plot(timesteps, vals, marker="o", label=feat)
    ax.set_xlabel("timestep")
    ax.set_ylabel("feature importance")
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig