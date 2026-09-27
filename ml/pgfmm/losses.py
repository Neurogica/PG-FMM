"""Physics-informed auxiliary losses for precipitation nowcasting.

Encodes 3 physical priors of precipitation fields, applied to the **decoded
pixel** outputs in Stage 2 (PI-JEPA fine-tune).  Encoder remains frozen, only
predictor + decoder adapt to satisfy these constraints.

Why physics priors help here
----------------------------
Vanilla pixel-MSE training suffers two well-documented failure modes on
precipitation forecasting:
  1. **Regression-to-mean blur**: extreme rain cells get smoothed out
  2. **Long lead-time divergence**: the model becomes over-confident about
     rapid changes that violate basic conservation laws

These auxiliary losses inject inductive biases from the physics of
precipitation transport (mass conservation, temporal continuity, advection-
diffusion dynamics).  They are weighted by ``lambda_*`` config entries and
added to the L1 + LPIPS pixel loss in ``train_finetune_pi.py``.

References
----------
* Nowcast3D (Chen et al., 2025) - PDE-informed neural operator for nowcasting
* PBFM (Baldan et al., 2025) - Physics-informed flow matching with PDE residual
* PCFM (Utkarsh et al., NeurIPS 2025) - Physics-constrained flow matching
* PreDiff (Gao et al., NeurIPS 2023) - knowledge alignment in diffusion
* Bastek et al. (ICLR 2025) - Physics-informed diffusion models
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Loss 1: Mass conservation
# ---------------------------------------------------------------------------


def mass_conservation_loss(pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Penalise sudden changes in total precipitation between consecutive frames.

    Args
    ----
    pred : ``(B, T, 1, H, W)`` predicted radar frames in ``[0, 1]``.

    Returns
    -------
    Scalar loss = mean of  ``|sum_{t+1} - sum_t| / (sum_t + eps)`` over (B, T-1).

    Physical meaning
    ----------------
    The total precipitation amount over a sufficiently large area should not
    change discontinuously — strong rain cells don't appear or disappear in
    seconds.  If the model predicts a frame at t with 100 units of total
    rainfall but predicts t+1 with 50 units, that's physically implausible
    (no source/sink term would cause that).
    """
    totals = pred.sum(dim=(-1, -2, -3))  # (B, T)
    diffs = (totals[:, 1:] - totals[:, :-1]).abs()  # (B, T-1)
    rel_diffs = diffs / (totals[:, :-1] + eps)
    return rel_diffs.mean()


# ---------------------------------------------------------------------------
# Loss 2: Temporal smoothness
# ---------------------------------------------------------------------------


def temporal_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalise the 2nd-order temporal derivative of the predicted frames.

    Args
    ----
    pred : ``(B, T, 1, H, W)``.  Requires T >= 3.

    Returns
    -------
    Scalar loss = mean of  ``|d²R/dt²|``  over interior frames (B, T-2, ...).

    Physical meaning
    ----------------
    Atmospheric flow evolves smoothly over short time scales (5-min radar
    intervals).  Penalising the 2nd-order temporal derivative discourages
    "flickering" predictions where intensity oscillates frame-to-frame.
    Equivalent to a temporal Laplacian regulariser, also used in
    optical-flow estimation literature.
    """
    if pred.shape[1] < 3:
        return pred.new_zeros(())  # not enough frames
    dd = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]  # (B, T-2, 1, H, W)
    return dd.abs().mean()


# ---------------------------------------------------------------------------
# Loss 3: Advection-diffusion residual (no explicit velocity field)
# ---------------------------------------------------------------------------

# 3x3 Laplacian stencil used by ``diffusion_residual``.
_LAPLACIAN_STENCIL = torch.tensor(
    [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]],
    dtype=torch.float32,
)


def diffusion_residual(
    pred: torch.Tensor,
    kappa: float = 0.01,
) -> torch.Tensor:
    """Approximate residual of the *diffusion-only* PDE  ``∂R/∂t = κ·∇²R``.

    Args
    ----
    pred  : ``(B, T, 1, H, W)``.
    kappa : diffusion coefficient.  ``0.01`` corresponds to mild smoothing
            (paper-default for radar VIL data).

    Returns
    -------
    Scalar loss = mean of  ``|∂R/∂t - κ·∇²R|``  over (B, T-1, 1, H, W).

    Why drop advection?
    -------------------
    A full advection-diffusion residual would be
    ``∂R/∂t + v·∇R - κ·∇²R = 0``,
    which requires an external velocity-field estimate (optical flow).
    To keep this loss computable without an optical-flow dependency, we
    drop the advective term.  Combined with ``mass_conservation_loss``, the
    pure-diffusion residual still provides a useful smoothness inductive bias
    that has been shown to suppress non-physical "blooming" of rain cells.

    For follow-up work, plug in pre-computed optical flow ``v`` and add
    ``(v · grad_R)`` to the residual; the rest of the pipeline is unchanged.
    """
    B, T, C, H, W = pred.shape
    if T < 2:
        return pred.new_zeros(())

    # ∂R/∂t  (forward difference)
    dr_dt = pred[:, 1:] - pred[:, :-1]  # (B, T-1, C, H, W)

    # ∇²R via the 3x3 Laplacian stencil applied per (frame, channel)
    R = pred.reshape(B * T, C, H, W)
    kernel = _LAPLACIAN_STENCIL.to(device=pred.device, dtype=pred.dtype)
    if C != 1:  # extend to per-channel kernel
        kernel = kernel.expand(C, 1, 3, 3).contiguous()
        lap = F.conv2d(R, kernel, padding=1, groups=C)
    else:
        lap = F.conv2d(R, kernel, padding=1)
    lap = lap.reshape(B, T, C, H, W)[:, :-1]  # align with dr_dt

    residual = dr_dt - kappa * lap
    return residual.abs().mean()


# ---------------------------------------------------------------------------
# Loss 4: Advection-diffusion residual WITH learnable flow (PI-JEPA v2)
# ---------------------------------------------------------------------------

# 3x3 Sobel kernels for spatial gradients.
_SOBEL_X = (
    torch.tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        dtype=torch.float32,
    )
    / 8.0
)  # divide by 8 for unit-pixel gradient scale


def advection_diffusion_residual(
    pred: torch.Tensor,
    flow: torch.Tensor,
    kappa: float = 0.01,
) -> torch.Tensor:
    """Approximate residual of  ``∂R/∂t + v·∇R - κ·∇²R = 0``.

    Args
    ----
    pred  : ``(B, T, 1, H, W)`` predicted radar frames.
    flow  : ``(B, T-1, 2, H, W)`` learnable flow field (channel 0 = v_x,
            channel 1 = v_y).  Output of :class:`FlowHead`.
    kappa : diffusion coefficient.

    Returns
    -------
    Scalar loss = mean of  ``|∂R/∂t + v·∇R - κ·∇²R|``  over (B, T-1, 1, H, W).

    Physical meaning
    ----------------
    Full advection-diffusion-source PDE for radar reflectivity:
        ∂R/∂t + v·∇R - κ∇²R = s
    We assume zero source (s = 0), so the residual measures how well the
    predicted frames + learned flow satisfy 2D radar transport.

    The flow is **not** supervised by any ground-truth optical flow; it is
    learned purely by minimising this residual jointly with the pixel L1
    loss.  This is the same self-supervised flow estimation trick used by
    Nowcast3D (Chen et al., 2025) for 3D radar.

    Sign convention
    ---------------
    * v_x > 0 means rightward pixel motion
    * Sobel ∂R/∂x  has the standard convention "pixel value increases
      to the right →  derivative > 0"
    * Hence the advection term v·∇R is *positive* when the model expects R
      to *increase* at this pixel because high-R material is being advected
      *into* the pixel from the upstream direction.
    """
    B, T, C, H, W = pred.shape
    if T < 2:
        return pred.new_zeros(())
    assert flow.shape == (B, T - 1, 2, H, W), (
        f"flow shape mismatch: got {tuple(flow.shape)}, expected ({B}, {T - 1}, 2, {H}, {W})"
    )

    # ∂R/∂t  (forward difference)
    dr_dt = pred[:, 1:] - pred[:, :-1]  # (B, T-1, C, H, W)

    # ∇R via Sobel — applied to R[t] (the "from" side of the forward diff).
    R = pred[:, :-1].reshape(B * (T - 1) * C, 1, H, W)
    sx = _SOBEL_X.to(device=pred.device, dtype=pred.dtype)
    sy = sx.transpose(-1, -2).contiguous()
    grad_x = F.conv2d(R, sx, padding=1).reshape(B, T - 1, C, H, W)
    grad_y = F.conv2d(R, sy, padding=1).reshape(B, T - 1, C, H, W)

    # v · ∇R  — flow has C=1 (precip is scalar), broadcast across C if needed.
    v_x = flow[:, :, 0:1]  # (B, T-1, 1, H, W)
    v_y = flow[:, :, 1:2]
    advection = v_x * grad_x + v_y * grad_y  # (B, T-1, C, H, W)

    # ∇²R via 3x3 Laplacian
    kernel = _LAPLACIAN_STENCIL.to(device=pred.device, dtype=pred.dtype)
    lap = F.conv2d(R, kernel, padding=1).reshape(B, T - 1, C, H, W)

    residual = dr_dt + advection - kappa * lap
    return residual.abs().mean()


# ---------------------------------------------------------------------------
# Combined helper
# ---------------------------------------------------------------------------


def physics_loss_bundle(
    pred: torch.Tensor,
    *,
    lambda_mass: float = 0.0,
    lambda_smooth: float = 0.0,
    lambda_diffusion: float = 0.0,
    diffusion_kappa: float = 0.01,
    flow: torch.Tensor | None = None,
    lambda_advdiff: float = 0.0,
    advdiff_kappa: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine the 4 physics-informed losses with their lambda weights.

    Args
    ----
    pred             : ``(B, T, 1, H, W)`` — predicted future radar frames
    lambda_mass      : weight for mass conservation loss
    lambda_smooth    : weight for temporal smoothness loss
    lambda_diffusion : weight for diffusion-only residual loss (PI-JEPA v1)
    diffusion_kappa  : diffusion coefficient for the diffusion residual
    flow             : ``(B, T-1, 2, H, W)`` learnable flow field, required
                       when ``lambda_advdiff > 0`` (PI-JEPA v2).  When None,
                       advection-diffusion residual is skipped.
    lambda_advdiff   : weight for full advection-diffusion residual loss
                       (PI-JEPA v2; mutually exclusive with diffusion-only)
    advdiff_kappa    : diffusion coefficient for the advection-diffusion residual

    Returns
    -------
    (total_phys_loss, dict of individual components for logging)
    Each individual component is *unweighted* in the dict so wandb plots
    show the raw magnitudes, while ``total_phys_loss`` is the weighted sum.

    Note
    ----
    PI-JEPA v1 uses ``lambda_diffusion > 0, lambda_advdiff = 0``  (no flow)
    PI-JEPA v2 uses ``lambda_advdiff > 0, lambda_diffusion = 0``  (with flow)
    Setting both > 0 is allowed but redundant — they share the κ∇²R term.
    """
    parts = {
        "mass": mass_conservation_loss(pred) if lambda_mass > 0 else pred.new_zeros(()),
        "smooth": temporal_smoothness_loss(pred) if lambda_smooth > 0 else pred.new_zeros(()),
        "diffusion": diffusion_residual(pred, kappa=diffusion_kappa)
        if lambda_diffusion > 0
        else pred.new_zeros(()),
        "advdiff": (
            advection_diffusion_residual(pred, flow, kappa=advdiff_kappa)
            if lambda_advdiff > 0 and flow is not None
            else pred.new_zeros(())
        ),
    }
    total = (
        lambda_mass * parts["mass"]
        + lambda_smooth * parts["smooth"]
        + lambda_diffusion * parts["diffusion"]
        + lambda_advdiff * parts["advdiff"]
    )
    return total, parts


__all__ = [
    "mass_conservation_loss",
    "temporal_smoothness_loss",
    "diffusion_residual",
    "advection_diffusion_residual",
    "physics_loss_bundle",
]
