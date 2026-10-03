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

    Transcribed from the reference notebook (X-Diffusion/Experiments.ipynb):

        alpha      = e^(ln(T) / (l + gamma))
        early:     [T - int(alpha**(i+1+gamma)) for i in range(l)], then
                   insert 999 at the front and drop the final entry
        latter:    [int(alpha**(i+1+gamma)) for i in range(l)], sorted
                   descending, with a leading 1000 clamped to 999

    The `i+1+gamma` exponent (not `i+gamma`) is what makes the first generated
    step 893 rather than 900, and the insert/pop pair is what pins the schedule
    to start at 999 and end above 0 -- both matter for matching the reference.

    gear='early'  -> dense near T (early denoising stages)
    gear='late'   -> dense near 0 (late denoising stages); 'latter' accepted
    """
    if gear not in ("early", "late", "latter"):
        raise ValueError(f"gear must be 'early' or 'late', got {gear}")
    alpha = np.exp(np.log(T) / (l + gamma))
    idx = np.arange(l) + 1 + gamma

    if gear == "early":
        steps = [T - int(alpha ** e) for e in idx]
        steps.insert(0, T - 1)   # 999: the true first step of the reverse process
        steps.pop(-1)            # drop the trailing T - int(T) == 0 entry
    else:
        steps = [int(alpha ** e) for e in idx]
        steps.sort(reverse=True)
        if steps[0] == T:
            steps[0] = T - 1

    # Int list already; keep as an int array for indexing parity with
    # scheduler.timesteps.
    return np.array(steps, dtype=int)


def uniform_timesteps(T: int = 1000, l: int = 30) -> np.ndarray:
    """Baseline uniform sampling: equal intervals of T/l (ints, same reason)."""
    return np.round(np.linspace(T, 0, l + 1)[1:]).astype(int)