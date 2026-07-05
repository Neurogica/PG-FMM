"""Beta-schedule utilities for I²SB-style Schrödinger Bridge.

Closely follows ``ext_repos/I2SB/i2sb/runner.py:make_beta_schedule`` so we
inherit the same training/sampling dynamics and can compare apples-to-apples
with the published numbers.
"""

from __future__ import annotations

import numpy as np
import torch


def make_beta_schedule(
    n_timestep: int = 1000,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
) -> np.ndarray:
    """Quadratic-spaced beta schedule used by guided-diffusion / I²SB.

    Returns
    -------
    np.ndarray of shape ``(n_timestep,)`` with values in ``[linear_start, linear_end]``.
    """
    # ``betas[i] = (sqrt(linear_start) + i/(N-1) * (sqrt(linear_end)-sqrt(linear_start)))**2``
    betas = (
        torch.linspace(
            linear_start**0.5,
            linear_end**0.5,
            n_timestep,
            dtype=torch.float64,
        )
        ** 2
    )
    return betas.numpy()


def make_symmetric_beta_schedule(
    interval: int = 1000,
    beta_max: float = 0.3,
    linear_start: float = 1e-4,
) -> np.ndarray:
    """Tent / symmetric schedule (rising then mirrored) used by I²SB.

    The total integral over ``[0, T]`` controls the bridge's noise budget.
    See ``ext_repos/I2SB/i2sb/runner.py``.

    Args
    ----
    interval     : number of discrete bridge steps (``N`` in the paper)
    beta_max     : maximum beta at the midpoint; the per-step max becomes
                   ``beta_max / interval``
    linear_start : minimum beta at endpoints

    Returns
    -------
    np.ndarray of shape ``(interval,)``.
    """
    betas = make_beta_schedule(
        n_timestep=interval,
        linear_start=linear_start,
        linear_end=beta_max / interval,
    )
    half = interval // 2
    return np.concatenate([betas[:half], np.flip(betas[:half])])


def space_indices(num_steps: int, count: int) -> list[int]:
    """Uniformly pick ``count`` integer indices from ``[0, num_steps-1]``.

    Used for picking the NFE sampling sub-grid (``nfe << interval``).
    Mirrors ``ext_repos/I2SB/i2sb/util.py:space_indices``.
    """
    assert count <= num_steps
    frac_stride = 1 if count <= 1 else (num_steps - 1) / (count - 1)
    cur, picks = 0.0, []
    for _ in range(count):
        picks.append(round(cur))
        cur += frac_stride
    return picks


def unsqueeze_xdim(z: torch.Tensor, xdim: tuple[int, ...]) -> torch.Tensor:
    """Add trailing singleton dims so that ``z`` broadcasts over an x-shaped tensor.

    e.g. ``z.shape == (B,)`` and ``xdim == (C, H, W)``  →  output shape ``(B, 1, 1, 1)``.
    """
    bc = (...,) + (None,) * len(xdim)
    return z[bc]
