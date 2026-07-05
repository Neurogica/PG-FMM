"""Transport source endpoint ``x_1`` (the *prior*) construction strategies.

The flow map transports the source ``x_1`` to the forecast ``x_0``; choosing
``x_1`` well is an important design knob.  We provide several lightweight
options that do **not** require a separately trained backbone.

All builders receive past frames ``y_in`` of shape ``(B, T_in, C, H, W)`` and
target length ``T_out`` and return ``x1`` of shape ``(B, T_out, C, H, W)`` to
match the future-frame target ``y_out``.
"""

from __future__ import annotations

import torch


def repeat_last_frame(y_in: torch.Tensor, T_out: int) -> torch.Tensor:
    """Persistence-style prior: replicate the most recent observed frame ``T_out`` times."""
    last = y_in[:, -1:, ...]  # (B, 1, C, H, W)
    return last.expand(-1, T_out, -1, -1, -1).contiguous()


def repeat_mean_past(y_in: torch.Tensor, T_out: int) -> torch.Tensor:
    """Replicate the temporal *mean* of all past frames ``T_out`` times."""
    mean = y_in.mean(dim=1, keepdim=True)  # (B, 1, C, H, W)
    return mean.expand(-1, T_out, -1, -1, -1).contiguous()


def linear_extrapolation(y_in: torch.Tensor, T_out: int) -> torch.Tensor:
    """Naive linear extrapolation in time (per pixel).

    Uses the slope from the first to the last past frame and projects forward.
    Cheap & sometimes already a meaningful prior for slow systems.
    """
    T_in = y_in.shape[1]
    if T_in < 2:
        return repeat_last_frame(y_in, T_out)
    slope = (y_in[:, -1] - y_in[:, 0]) / (T_in - 1)  # (B, C, H, W)
    last = y_in[:, -1]  # (B, C, H, W)
    out = torch.stack([last + slope * (k + 1) for k in range(T_out)], dim=1)
    return out


_PRIORS = {
    "last_frame": repeat_last_frame,
    "mean_past": repeat_mean_past,
    "linear_extrapolation": linear_extrapolation,
}


def build_prior(name: str, y_in: torch.Tensor, T_out: int) -> torch.Tensor:
    """Dispatcher.  Use ``name='last_frame'`` as the default safe choice."""
    if name not in _PRIORS:
        raise KeyError(f"Unknown prior {name!r}. Known: {sorted(_PRIORS)}")
    return _PRIORS[name](y_in, T_out)
