"""Evaluation: deletion/insertion games and AUC scores (paper §4.1, Fig 4, Tab 1).

Deletion game:  remove pixels in decreasing saliency order, measure FID-like
                distance to the original → good map degrades fast (steep early
                rise). AUC HIGH is better (DF-RISE metric convention, y=dist).
Insertion game: add pixels in decreasing saliency order to a blank, measure
                distance → good map restores fast. AUC LOW is better.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Small helper: a stand-in "perceptual distance" for cheap prototyping.
# Replace with real FID (torch-fidelity) or SSIM as you scale up.

def _raster_order(saliency: np.ndarray, descending: bool = True) -> np.ndarray:
    """Return pixel indices sorted by saliency."""
    flat = saliency.ravel()
    order = np.argsort(flat)
    return order[::-1] if descending else order


def delete_pixels(image: np.ndarray, saliency: np.ndarray,
                  fraction: float, mode: str = "zero") -> np.ndarray:
    """Return image with the top-`fraction` salient pixels degraded."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction in [0,1]")
    out = image.copy().astype(np.float32)
    order = _raster_order(saliency, descending=True)
    n_delete = int(fraction * order.size)
    idx = order[:n_delete]
    h, w = saliency.shape
    ys, xs = np.unravel_index(idx, (h, w))
    if mode == "zero":
        out[:, ys, xs] = 0.0
    elif mode == "mean":
        out[:, ys, xs] = out.mean()
    else:
        raise ValueError(mode)
    return out


def insert_pixels(blank: np.ndarray, image: np.ndarray, saliency: np.ndarray,
                  fraction: float) -> np.ndarray:
    """Return `blank` with the top-`fraction` salient pixels copied from image."""
    out = blank.copy().astype(np.float32)
    order = _raster_order(saliency, descending=True)
    n_ins = int(fraction * order.size)
    idx = order[:n_ins]
    h, w = saliency.shape
    ys, xs = np.unravel_index(idx, (h, w))
    out[:, ys, xs] = image[:, ys, xs]
    return out


def game_curve(saliency: np.ndarray, original: np.ndarray, game: str,
               fractions=None, score_name: str = "l2") -> np.ndarray:
    """Compute a deletion/insertion curve (scores at each fraction).

    score_name: 'l2' cheap stand-in; use real similarity later.
    Returns: float array, one score per fraction.
    """
    if fractions is None:
        fractions = np.linspace(0.1, 1.0, 10)
    curve = []
    ref = original.astype(np.float32)
    if game == "deletion":
        for frac in fractions:
            cand = delete_pixels(original, saliency, frac)
            curve.append(_distance(ref, cand, score_name))
    elif game == "insertion":
        blank = np.zeros_like(ref)
        for frac in fractions:
            cand = insert_pixels(blank, original, saliency, frac)
            curve.append(_distance(ref, cand, score_name))
    else:
        raise ValueError(game)
    return np.asarray(curve)


def _distance(a: np.ndarray, b: np.ndarray, name: str) -> float:
    if name == "l2":
        return float(np.sqrt(((a - b) ** 2).sum()))
    raise ValueError(name)


def auc(curve: np.ndarray, fractions=None) -> float:
    """AUC under the curve via trapezoid rule."""
    if fractions is None:
        fractions = np.linspace(0.1, 1.0, cur if (cur := len(curve)) else 10)
    fractions = np.asarray(fractions)
    # if curve provided without matching fractions, resample linearly
    if fractions.shape[0] != curve.shape[0]:
        fractions = np.linspace(0.1, 1.0, curve.shape[0])
    from numpy import trapz
    return float(trapz(curve, fractions))