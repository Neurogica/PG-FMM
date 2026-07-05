"""I²SB-style tractable Schrödinger Bridge diffusion math.

This is a 1-to-1 port of ``ext_repos/I2SB/i2sb/diffusion.py`` with type hints,
docstrings, and a couple of small QoL helpers.  All formulas keep the same
notation as the I²SB paper so the code is auditable against it.

References
----------
* Liu et al., "I²SB: Image-to-Image Schrödinger Bridge", ICML 2023.
  arxiv 2302.05872 / `papers/schrodinger_bridge/2023_ICML_I2SB.pdf`
* Code: https://github.com/NVlabs/I2SB

Notation
--------
* ``x_0`` is the *clean* / target endpoint distribution (e.g. ground-truth
  future radar frames).
* ``x_1`` is the *prior* / starting endpoint (e.g. last observed radar frame
  replicated, or a deterministic backbone forecast).
* ``x_t`` for ``t ∈ {0, ..., N-1}`` is the bridge state at discrete time ``t``.
* ``betas[t]`` are per-step variance increments; the **symmetric** schedule
  (low at endpoints, high in the middle) is used by I²SB.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .schedule import unsqueeze_xdim


def compute_gaussian_product_coef(sigma1, sigma2):
    """Eq.5 of I²SB.  Given two Gaussians with the same mean placeholder and
    stds ``sigma1`` and ``sigma2``, their product is again Gaussian with

        coef1 = sigma2² / (sigma1² + sigma2²)
        coef2 = sigma1² / (sigma1² + sigma2²)
        var   = sigma1² · sigma2² / (sigma1² + sigma2²)

    so that ``mu_prod = coef1 * mu1 + coef2 * mu2`` and ``var_prod = var``.
    """
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


class I2SBDiffusion(nn.Module):
    """Tractable I²SB bridge with analytic forward sampler and DDPM-style sampler.

    Subclass of ``nn.Module`` so that the precomputed schedule tensors are
    registered as buffers and follow the parent runner around with ``.to(device)``
    / ``accelerator.prepare(...)``.

    Args
    ----
    betas  : 1D ``np.ndarray`` of shape ``(N,)`` (symmetric tent schedule).
    device : initial device.  Buffers are moved here at construction time;
             later ``.to(<other_device>)`` calls work as for any nn.Module.
    """

    def __init__(self, betas: np.ndarray, device: torch.device | str = "cpu"):
        super().__init__()

        # Forward / backward analytic stds used in the closed-form bridge marginal.
        std_fwd = np.sqrt(np.cumsum(betas))  # σ_fwd[t] = sqrt(Σ_{s≤t} β_s)
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))  # σ_bwd[t] = sqrt(Σ_{s≥t} β_s)
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        to_t = partial(torch.tensor, dtype=torch.float32)
        # ``persistent=False`` so they don't bloat checkpoints (re-derivable from cfg).
        self.register_buffer("betas", to_t(betas), persistent=False)
        self.register_buffer("std_fwd", to_t(std_fwd), persistent=False)
        self.register_buffer("std_bwd", to_t(std_bwd), persistent=False)
        self.register_buffer("std_sb", to_t(std_sb), persistent=False)
        self.register_buffer("mu_x0", to_t(mu_x0), persistent=False)
        self.register_buffer("mu_x1", to_t(mu_x1), persistent=False)

        self.to(torch.device(device))

    @property
    def device(self) -> torch.device:
        return self.betas.device

    @property
    def n_steps(self) -> int:
        return self.betas.shape[0]

    def get_std_fwd(self, step, xdim=None) -> torch.Tensor:
        s = self.std_fwd[step]
        return s if xdim is None else unsqueeze_xdim(s, xdim)

    # ---------------------------------------------------------------------
    # Analytic forward bridge marginal  q(x_t | x_0, x_1)   (Eq.11)
    # ---------------------------------------------------------------------
    def q_sample(
        self,
        step: torch.Tensor | int,
        x0: torch.Tensor,
        x1: torch.Tensor,
        ot_ode: bool = False,
    ) -> torch.Tensor:
        """Sample from ``q(x_t | x_0, x_1)``.

        Equivalent to:
            ``x_t = mu_x0[t] · x_0 + mu_x1[t] · x_1 + σ_sb[t] · ε``
        with ``ε ~ N(0, I)``.  When ``ot_ode=True``, the noise term is dropped
        (deterministic OT-flow flavour, ablation in I²SB paper).
        """
        assert x0.shape == x1.shape
        _, *xdim = x0.shape
        mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)
        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    # ---------------------------------------------------------------------
    # Training label (Eq.12)
    # ---------------------------------------------------------------------
    def compute_label(
        self,
        step: torch.Tensor | int,
        x0: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """Network regression target  ``(x_t - x_0) / σ_fwd[t]``."""
        std_fwd = self.get_std_fwd(step, xdim=x0.shape[1:])
        return ((xt - x0) / std_fwd).detach()

    def compute_pred_x0(
        self,
        step: torch.Tensor | int,
        xt: torch.Tensor,
        net_out: torch.Tensor,
        clamp: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Inverse of Eq.12: recover ``x_0`` from network ``ε``-style output."""
        std_fwd = self.get_std_fwd(step, xdim=xt.shape[1:])
        x0 = xt - std_fwd * net_out
        if clamp is not None:
            x0 = x0.clamp_(*clamp)
        return x0

    # ---------------------------------------------------------------------
    # Reverse posterior  p(x_{nprev} | x_n, x_0)   (Eq.4)
    # ---------------------------------------------------------------------
    def p_posterior(
        self,
        nprev: int,
        n: int,
        x_n: torch.Tensor,
        x0: torch.Tensor,
        ot_ode: bool = False,
    ) -> torch.Tensor:
        """One reverse step of the bridge: draw ``x_{nprev}`` given ``(x_n, x_0)``."""
        assert nprev < n
        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)
        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if (not ot_ode) and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)
        return xt_prev

    # ---------------------------------------------------------------------
    # Full DDPM-style sampler over a sub-grid of ``log_steps``
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def ddpm_sampling(
        self,
        steps: list[int],
        pred_x0_fn: Callable[[torch.Tensor, int], torch.Tensor],
        x1: torch.Tensor,
        mask: torch.Tensor | None = None,
        ot_ode: bool = False,
        log_steps: list[int] | None = None,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reverse-time integration starting from ``x_1`` along ``steps``.

        Args
        ----
        steps      : monotone-increasing list of bridge indices, ``steps[0]==0``,
                     ``steps[-1]==N-1`` (or the largest used).  Sampling proceeds
                     ``steps[-1] → steps[-2] → … → 0``.
        pred_x0_fn : callable ``(xt, int_step) -> pred_x0`` (e.g. wraps the net).
        x1         : starting point at ``t = N-1`` of shape ``(B, ..., H, W)``.
        mask       : optional inpainting mask.
        ot_ode     : if True, deterministic OT flow (no Brownian noise).
        log_steps  : sub-list of ``steps`` to log; defaults to all of ``steps``.

        Returns
        -------
        ``(xs, pred_x0s)`` each of shape ``(B, len(log_steps), ..., H, W)``,
        time-ordered from earliest to latest in the bridge (i.e. flipped so
        index 0 is the final sample at ``t=0``).
        """
        xt = x1.detach().to(self.device)
        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps_rev = steps[::-1]
        pair = zip(steps_rev[1:], steps_rev[:-1], strict=False)
        if verbose:
            pair = tqdm(pair, desc="SB ddpm sampling", total=len(steps_rev) - 1)

        xs, pred_x0s = [], []
        for prev_step, step in pair:
            assert prev_step < step
            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    p = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[p], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1.0 - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        # Time-order ascending (so [0] == final sample, [-1] == first reverse step)
        def stack_bwd(z):
            return torch.flip(torch.stack(z, dim=1), dims=(1,))

        return stack_bwd(xs), stack_bwd(pred_x0s)
