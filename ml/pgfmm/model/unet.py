"""Conditional 2D U-Net used by the SB nowcasting model.

We treat time as a *channel* dimension and condition on the past frames by
concatenation.  The backbone is ``diffusers.UNet2DModel`` so we inherit
group-norm, attention blocks, residual connections, and time-step embedding
out of the box.

Tensor shapes
-------------
* ``x``     : ``(B, T_out * C, H, W)``  — the bridge state at step ``t``
* ``cond``  : ``(B, T_in  * C, H, W)``  — past frames stacked into channels
* ``step``  : ``(B,)``                  — discrete bridge index (long tensor)
* ``noise_levels`` : 1-D LUT mapping integer step → continuous time fed to the
  U-Net's sinusoidal embedding (``≃`` what I²SB does).

The wrapper performs ``cat([x, cond], dim=1)`` if a condition is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers.models import UNet2DModel


@dataclass
class NowcastUNetConfig:
    T_in: int = 5
    T_out: int = 20
    img_channels: int = 1
    bridge_channels: int = 1
    img_size: int = 128
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 4, 8)
    attention_resolutions: tuple[int, ...] = (16, 8)
    num_res_blocks: int = 2
    dropout: float = 0.0
    interval: int = 1000  # number of bridge steps for the time embedding LUT
    cond_x1: bool = True
    cond_extra_T: int = 0  # # of extra T-frames to concat as condition
    # (e.g. T_out for an AlphaPre forecast). 0 = disabled.
    time_cond_channels: int = 0  # optional constant maps, e.g. flow-map (t, r)
    self_cond_channels: int = 0  # RIN-style self-conditioning channels (= T_out * img_channels
    # for a per-pixel coarse prediction passed as conditioning).
    use_residual_gate: bool = False
    gate_init_bias: float = -1.10  # sigmoid(-1.10) ~= 0.25 (matches calibrated residual)


class NowcastUNet(nn.Module):
    """A diffusers UNet2DModel wired for SB nowcasting.

    The wrapper:
      1. concatenates ``cond`` (past frames) onto ``x`` along the channel dim,
      2. translates integer ``step`` to a continuous timestep via ``noise_levels``,
      3. returns the U-Net's epsilon-style output reshaped back to ``(B, T_out*C, H, W)``.

    The output is treated as the I²SB regression target ``label = (x_t - x_0) / σ_fwd[t]``.
    """

    def __init__(self, cfg: NowcastUNetConfig):
        super().__init__()
        self.cfg = cfg

        in_ch = cfg.T_out * cfg.bridge_channels
        cond_ch = (cfg.T_in * cfg.img_channels) if cfg.cond_x1 else 0
        extra_ch = cfg.cond_extra_T * cfg.img_channels  # e.g. AlphaPre forecast
        time_cond_ch = cfg.time_cond_channels
        self_cond_ch = cfg.self_cond_channels
        out_ch = cfg.T_out * cfg.bridge_channels

        # diffusers down/up block selection: 'AttnDownBlock2D' uses self-attention.
        # We mirror I²SB / guided-diffusion: attention only at the smaller spatial
        # resolutions specified by ``attention_resolutions``.
        block_out = tuple(cfg.base_channels * m for m in cfg.channel_mult)
        n_levels = len(block_out)

        # Compute spatial resolution at each U-Net level (each step downsamples by 2).
        spatial = [cfg.img_size // (2**i) for i in range(n_levels)]
        attn = set(cfg.attention_resolutions)
        down_types = tuple("AttnDownBlock2D" if (s in attn) else "DownBlock2D" for s in spatial)
        up_types = tuple("AttnUpBlock2D" if (s in attn) else "UpBlock2D" for s in reversed(spatial))

        # GroupNorm requires num_channels % num_groups == 0 at every level.
        # diffusers' default 32 groups breaks for smoke-test sizes; pick the
        # largest divisor ≤ 32 that works for the smallest block.
        smallest = min(block_out)
        for g in (32, 16, 8, 4, 2, 1):
            if smallest % g == 0:
                norm_groups = g
                break
        else:
            norm_groups = 1

        self.unet = UNet2DModel(
            sample_size=cfg.img_size,
            in_channels=in_ch + cond_ch + extra_ch + time_cond_ch + self_cond_ch,
            out_channels=out_ch,
            layers_per_block=cfg.num_res_blocks,
            block_out_channels=block_out,
            down_block_types=down_types,
            up_block_types=up_types,
            attention_head_dim=8,
            norm_num_groups=norm_groups,
            dropout=cfg.dropout,
            time_embedding_type="positional",
        )
        total_in_ch = in_ch + cond_ch + extra_ch + time_cond_ch + self_cond_ch
        if cfg.use_residual_gate:
            self.gate_head = nn.Sequential(
                nn.Conv2d(total_in_ch, cfg.base_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(cfg.base_channels, out_ch, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.constant_(self.gate_head[-1].bias, cfg.gate_init_bias)
        else:
            self.gate_head = None

        # Step → continuous timestep LUT (paper: t0=1e-4, T=1.0).
        # We pass the SCALED (× interval) value to UNet2DModel; the embedding
        # there expects a number in roughly the same magnitude as the diffusion
        # step.  This matches I²SB's `noise_levels[step] * interval`.
        noise_levels = torch.linspace(1e-4, 1.0, cfg.interval) * cfg.interval
        self.register_buffer("noise_levels", noise_levels, persistent=False)

    # ------------------------------------------------------------------
    @property
    def in_channels(self) -> int:
        return int(self.unet.config.in_channels)

    @property
    def out_channels(self) -> int:
        return int(self.unet.config.out_channels)

    # ------------------------------------------------------------------
    def _prepare_input(
        self,
        x: torch.Tensor,  # (B, T_out, C, H, W) or (B, T_out*C, H, W)
        cond: torch.Tensor | None = None,  # (B, T_in, C, H, W) or (B, T_in*C, H, W)
        cond_extra: torch.Tensor | None = None,  # (B, T_extra, C, H, W) or flat (e.g. AlphaPre)
        time_cond: torch.Tensor | None = None,  # (B, time_cond_channels, H, W)
        self_cond: torch.Tensor | None = None,  # (B, T_out, C, H, W) or (B, T_out*C, H, W)
    ) -> tuple[torch.Tensor, bool]:
        cfg = self.cfg
        x_was_5d = x.ndim == 5
        if x_was_5d:
            x = x.flatten(1, 2)
        if cond is not None and cond.ndim == 5:
            cond = cond.flatten(1, 2)
        if cond_extra is not None and cond_extra.ndim == 5:
            cond_extra = cond_extra.flatten(1, 2)
        if time_cond is not None and time_cond.ndim == 5:
            time_cond = time_cond.flatten(1, 2)
        if self_cond is not None and self_cond.ndim == 5:
            self_cond = self_cond.flatten(1, 2)

        parts: list[torch.Tensor] = [x]
        if cfg.cond_x1:
            assert cond is not None, "cond_x1=True but cond is None"
            parts.append(cond)
        if cfg.cond_extra_T > 0:
            assert cond_extra is not None, f"cond_extra_T={cfg.cond_extra_T} but cond_extra is None"
            assert cond_extra.shape[1] == cfg.cond_extra_T * cfg.img_channels, (
                f"cond_extra channels {cond_extra.shape[1]} != "
                f"cond_extra_T * img_channels = {cfg.cond_extra_T * cfg.img_channels}"
            )
            parts.append(cond_extra)
        if cfg.time_cond_channels > 0:
            assert time_cond is not None, (
                f"time_cond_channels={cfg.time_cond_channels} but time_cond is None"
            )
            assert time_cond.shape[1] == cfg.time_cond_channels, (
                f"time_cond channels {time_cond.shape[1]} != "
                f"time_cond_channels = {cfg.time_cond_channels}"
            )
            parts.append(time_cond.to(device=x.device, dtype=x.dtype))
        if cfg.self_cond_channels > 0:
            if self_cond is None:
                # RIN-style warm-up: zeros denote "no prior coarse prediction".
                self_cond = x.new_zeros(
                    (x.shape[0], cfg.self_cond_channels, x.shape[-2], x.shape[-1])
                )
            assert self_cond.shape[1] == cfg.self_cond_channels, (
                f"self_cond channels {self_cond.shape[1]} != "
                f"self_cond_channels = {cfg.self_cond_channels}"
            )
            parts.append(self_cond.to(device=x.device, dtype=x.dtype))
        x_in = torch.cat(parts, dim=1) if len(parts) > 1 else x
        return x_in, x_was_5d

    def forward(
        self,
        x: torch.Tensor,  # (B, T_out, C, H, W) or (B, T_out*C, H, W)
        step: torch.Tensor,  # (B,) long
        cond: torch.Tensor | None = None,  # (B, T_in, C, H, W) or (B, T_in*C, H, W)
        cond_extra: torch.Tensor | None = None,  # (B, T_extra, C, H, W) or flat (e.g. AlphaPre)
        time_cond: torch.Tensor | None = None,  # (B, time_cond_channels, H, W)
        self_cond: torch.Tensor | None = None,  # (B, T_out, C, H, W) or flat (RIN-style)
    ) -> torch.Tensor:  # (B, T_out, C, H, W) (matches x layout)
        cfg = self.cfg
        x_in, x_was_5d = self._prepare_input(
            x,
            cond=cond,
            cond_extra=cond_extra,
            time_cond=time_cond,
            self_cond=self_cond,
        )

        # I²SB passes integer bridge indices, while DDBM passes the official
        # continuous preconditioned timestep (1000 * 0.25 * log sigma).
        # Keep the old LUT path for integer tensors so existing checkpoints are
        # unchanged.
        if step.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            t = self.noise_levels[step].to(x.dtype)
        else:
            t = step.to(device=x.device, dtype=x.dtype)
        out = self.unet(x_in, timestep=t).sample

        if x_was_5d:
            out = out.unflatten(1, (cfg.T_out, cfg.bridge_channels))
        return out

    def gate(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        cond_extra: torch.Tensor | None = None,
        time_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return reliability gate in ``[0, 1]`` for residual corrections."""
        if self.gate_head is None:
            raise RuntimeError("gate() called but use_residual_gate=False")
        cfg = self.cfg
        if time_cond is None and cfg.time_cond_channels > 0:
            h, w = x.shape[-2:]
            time_cond = x.new_zeros((x.shape[0], cfg.time_cond_channels, h, w))
        x_in, x_was_5d = self._prepare_input(
            x,
            cond=cond,
            cond_extra=cond_extra,
            time_cond=time_cond,
        )
        logits = self.gate_head(x_in)
        gate = torch.sigmoid(logits)
        if x_was_5d:
            gate = gate.unflatten(1, (cfg.T_out, cfg.bridge_channels))
        return gate
