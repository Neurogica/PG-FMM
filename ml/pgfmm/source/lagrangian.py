"""Learned Lagrangian source model for precipitation nowcasting.

The model predicts a physically interpretable source forecast from observed
radar frames:

    R_{t+1} = warp(R_t, v_t) + s_t

where ``v_t`` is a learned dense velocity field and ``s_t`` is a learned
growth/decay source term.  The resulting rollout is intended to be the
``x_1`` endpoint for a source-to-target diffusion bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LagrangianSourceConfig:
    T_in: int = 5
    T_out: int = 20
    img_channels: int = 1
    img_size: int = 128
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_blocks: int = 2
    max_displacement: float = 8.0
    source_scale: float = 0.25


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class LagrangianSourceNet(nn.Module):
    """Predict flow/source fields and roll out a source forecast.

    Input shape is ``(B, T_in, C, H, W)`` in ``[0, 1]``.  Output forecast shape is
    ``(B, T_out, C, H, W)``; the auxiliary dict contains ``flow`` and ``source``.
    """

    def __init__(self, cfg: LagrangianSourceConfig):
        super().__init__()
        self.cfg = cfg
        in_ch = cfg.T_in * cfg.img_channels
        widths = [cfg.base_channels * m for m in cfg.channel_mult]

        self.stem = ConvBlock(in_ch, widths[0])
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = widths[0]
        for width in widths[1:]:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(ch, width, 3, stride=2, padding=1),
                    ConvBlock(width, width),
                )
            )
            ch = width

        for width in reversed(widths[:-1]):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(ch + width, width, 3, padding=1),
                    ConvBlock(width, width),
                )
            )
            ch = width

        out_ch = cfg.T_out * (2 + cfg.img_channels)
        self.head = nn.Conv2d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        if cond.ndim != 5:
            raise ValueError(f"expected cond shape (B,T,C,H,W), got {tuple(cond.shape)}")
        B, T, C, H, W = cond.shape
        if (cfg.T_in, cfg.img_channels) != (T, C):
            raise ValueError(f"expected T,C={(cfg.T_in, cfg.img_channels)}, got {(T, C)}")

        x = cond.flatten(1, 2)
        skips = []
        h = self.stem(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)

        for up in self.ups:
            skip = skips.pop(-2)
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(torch.cat([h, skip], dim=1))

        raw = self.head(h)
        raw = raw.view(B, cfg.T_out, 2 + cfg.img_channels, H, W)
        flow = cfg.max_displacement * torch.tanh(raw[:, :, :2])
        source = cfg.source_scale * torch.tanh(raw[:, :, 2:])
        pred = rollout_lagrangian(cond[:, -1], flow, source)
        return pred, {"flow": flow, "source": source}


def rollout_lagrangian(
    last_frame: torch.Tensor,
    flow: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Roll out ``R_{t+1}=warp(R_t,v_t)+s_t``.

    Args:
        last_frame: ``(B, C, H, W)``.
        flow: ``(B, T, 2, H, W)`` in pixel units.
        source: ``(B, T, C, H, W)`` additive source/sink term.
    """
    B, T, _, H, W = flow.shape
    cur = last_frame
    outs = []
    for t in range(T):
        cur = warp_with_flow(cur, flow[:, t])
        cur = (cur + source[:, t]).clamp(0.0, 1.0)
        outs.append(cur)
    return torch.stack(outs, dim=1)


def warp_with_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-sample ``img`` using forward pixel displacement ``flow``.

    Positive ``flow[:,0]`` moves rain to the right; ``grid_sample`` needs source
    coordinates, so the normalized flow is subtracted from the base grid.
    """
    B, C, H, W = img.shape
    if flow.shape != (B, 2, H, W):
        raise ValueError(
            f"flow shape {tuple(flow.shape)} incompatible with image {tuple(img.shape)}"
        )
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=img.device, dtype=img.dtype),
        torch.linspace(-1.0, 1.0, W, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, H, W, 2)
    norm = torch.empty(B, H, W, 2, device=img.device, dtype=img.dtype)
    norm[..., 0] = 2.0 * flow[:, 0] / max(W - 1, 1)
    norm[..., 1] = 2.0 * flow[:, 1] / max(H - 1, 1)
    grid = base - norm
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def flow_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    dx = flow[..., :, 1:] - flow[..., :, :-1]
    dy = flow[..., 1:, :] - flow[..., :-1, :]
    dt = flow[:, 1:] - flow[:, :-1] if flow.shape[1] > 1 else flow.new_zeros(())
    return dx.abs().mean() + dy.abs().mean() + dt.abs().mean()


def source_sparsity_loss(source: torch.Tensor) -> torch.Tensor:
    return source.abs().mean()


def soft_csi_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    thresholds: tuple[float, ...] = (74.0, 133.0, 160.0, 181.0),
    pixel_scale: float = 255.0,
    sharpness: float = 8.0,
) -> torch.Tensor:
    pred_scaled = pred * pixel_scale
    gt_scaled = gt.detach() * pixel_scale
    losses = []
    for thr in thresholds:
        pred_soft = torch.sigmoid((pred_scaled - float(thr)) / sharpness)
        gt_bin = (gt_scaled >= float(thr)).float()
        tp = (pred_soft * gt_bin).sum(dim=(-1, -2, -3))
        fp = (pred_soft * (1.0 - gt_bin)).sum(dim=(-1, -2, -3))
        fn = ((1.0 - pred_soft) * gt_bin).sum(dim=(-1, -2, -3))
        csi = (tp + 1e-6) / (tp + fp + fn + 1e-6)
        losses.append(1.0 - csi.mean())
    return torch.stack(losses).mean()


def lagrangian_source_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    aux: dict[str, torch.Tensor],
    *,
    lambda_l1: float = 1.0,
    lambda_mse: float = 0.25,
    lambda_soft_csi: float = 0.02,
    lambda_flow_smooth: float = 0.01,
    lambda_source_sparse: float = 0.001,
    thresholds: tuple[float, ...] = (74.0, 133.0, 160.0, 181.0),
    pixel_scale: float = 255.0,
    soft_csi_sharpness: float = 8.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight = 1.0 + 2.0 * gt.detach()
    l1 = (weight * (pred - gt).abs()).mean()
    mse = F.mse_loss(pred, gt)
    csi = soft_csi_loss(
        pred,
        gt,
        thresholds=thresholds,
        pixel_scale=pixel_scale,
        sharpness=soft_csi_sharpness,
    )
    flow_smooth = flow_smoothness_loss(aux["flow"])
    source_sparse = source_sparsity_loss(aux["source"])
    total = (
        lambda_l1 * l1
        + lambda_mse * mse
        + lambda_soft_csi * csi
        + lambda_flow_smooth * flow_smooth
        + lambda_source_sparse * source_sparse
    )
    parts = {
        "l1": l1.detach(),
        "mse": mse.detach(),
        "soft_csi": csi.detach(),
        "flow_smooth": flow_smooth.detach(),
        "source_sparse": source_sparse.detach(),
    }
    return total, parts


__all__ = [
    "LagrangianSourceConfig",
    "LagrangianSourceNet",
    "flow_smoothness_loss",
    "lagrangian_source_loss",
    "rollout_lagrangian",
    "soft_csi_loss",
    "source_sparsity_loss",
    "warp_with_flow",
]
