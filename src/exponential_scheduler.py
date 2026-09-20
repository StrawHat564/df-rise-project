"""Exponential time-step sampling for the DDIM scheduler (Park et al. 2024, Eq 12-14).

Purpose: the default scheduler picks uniform steps, so no single temporal stage
is emphasized. Exponential sampling concentrates steps in a chosen stage
(early 1000-800, or mirrored late 200-0) so the visual concepts of that stage
become observable. Keeps full coverage (doesn't break training-time distribution).

Applies to a DDIM scheduler when converting requested step count to concrete
timesteps; needs to be a drop-in replacement for scheduler.timesteps.
"""
from __future__ import annotations

import numpy as np


def exponential_timesteps(
    T: int = 1000,
    l: int = 30,
    gamma: float = 60.0,
    gear: str = "early",
) -> np.ndarray:
    """Return l timesteps from T..0, concentrated per `gear`.

    Math:
        δ^(l+γ) = T                       (12)   boundary condition
        δ       = e^(ln T / (l+γ))        (13)
        p_t     = T - δ^(t+γ)             (14)

    For gear='early', p_t as above ⇒ dense near T.
    For gear='late',  p_t = δ^(t+γ) (mirrored) ⇒ dense near 0.

    γ small (<30): cluster between 1000 and 800 (very early).
    γ=60: paper's early-stage default.
    """
    delta = np.exp(np.log(T) / (l + gamma))

    if gear == "early":
        steps = T - delta ** (np.arange(l) + gamma)
    elif gear == "late":
        steps = delta ** (np.arange(l) + gamma)
    else:
        raise ValueError(f"gear must be 'early' or 'late', got {gear}")

    # clip numerics, sort descending from T to 0, and return INT timesteps:
    # the UNet/scheduler index internal arrays by timestep (alphas_cumprod[t]),
    # so float64 steps crash or silently misbehave. Floats are rounded so the
    # counts stay exact (all l steps above).
    steps = np.clip(steps, 0, T)
    steps = np.sort(steps)[::-1]
    return np.round(steps).astype(int)


def uniform_timesteps(T: int = 1000, l: int = 30) -> np.ndarray:
    """Baseline uniform sampling: equal intervals of T/l (ints, same reason)."""
    return np.round(np.linspace(T, 0, l + 1)[1:]).astype(int)