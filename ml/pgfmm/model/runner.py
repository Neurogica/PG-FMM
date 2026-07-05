"""High-level glue: build everything and expose ``train_step`` / ``sample``.

Mirrors I²SB's ``Runner`` but stripped down for single-GPU + nowcasting:
* no DDP boilerplate (we let ``accelerate`` handle distribution if needed)
* no FID/ResNet eval (we'll plug in CSI / CRPS metrics in ``src/metrics/``)
* single ``cond`` mode = past-frames concatenation
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..source import LagrangianSourceConfig, LagrangianSourceNet
from .ddbm import DDBMBridge, DDBMConfig
from .diffusion import I2SBDiffusion
from .prior import build_prior
from .schedule import make_symmetric_beta_schedule, space_indices
from .unet import NowcastUNet, NowcastUNetConfig


@dataclass
class PGFMMConfig:
    # data layout
    T_in: int = 5
    T_out: int = 20
    img_channels: int = 1
    bridge_channels: int = 1
    img_size: int = 128

    # SB schedule
    bridge_type: str = "flow_map"  # i2sb | ddbm | flow_map
    interval: int = 1000
    beta_max: float = 0.3
    ot_ode: bool = False  # True → deterministic OT-flow variant

    # DDBM / DBIM-family bridge settings.  Defaults mirror official DDBM's VP
    # image-translation setup (ext_repos/DDBM/args.sh).
    ddbm_pred_mode: str = "vp"
    ddbm_sigma_data: float = 0.5
    ddbm_sigma_min: float = 1.0e-4
    ddbm_sigma_max: float = 1.0
    ddbm_beta_d: float = 2.0
    ddbm_beta_min: float = 0.1
    ddbm_cov_xy: float = 0.0
    ddbm_rho: float = 7.0
    ddbm_weight_schedule: str = "bridge_karras"
    ddbm_churn_step_ratio: float = 0.0
    ddbm_guidance: float = 1.0
    ddbm_sampler: str = "dbim"
    ddbm_eta: float = 0.0
    flow_map_noise_scale: float = 1.0
    flow_map_min_delta: float = 0.05
    flow_map_direct_prob: float = 0.25
    flow_map_consistency_weight: float = 0.0
    flow_map_consistency_start_step: int = 0
    flow_map_consistency_ramp_steps: int = 0
    flow_map_consistency_detach_midpoint: bool = False
    # Energy-Score Flow-Map training (Gneiting & Raftery 2007's proper
    # scoring rule applied to direct-endpoint samples).  When enabled, the
    # direct-endpoint loss replaces MSE with
    #     S = mean_k ||X_k - y|| - 0.5 * mean_{k!=k'} ||X_k - X_k'||
    # which simultaneously rewards per-sample accuracy and sample
    # diversity, breaking the sharpness <-> accuracy tradeoff of plain MSE.
    flow_map_energy_score_weight: float = 0.0
    flow_map_energy_score_samples: int = 2
    # RIN-style self-conditioning for Flow-Map Matching.  Enables a single
    # UNet to act as its own deterministic backbone: a warm-up forward pass
    # produces a coarse prediction, which is then fed as additional input to
    # a second, refinement forward pass.  At training, with probability
    # ``flow_map_self_cond_prob`` the model is shown a stop-grad coarse
    # prediction; otherwise it sees zeros.  Drives a hybrid of mean-style
    # (location-accurate) and generative (sharp) behaviour without any
    # external backbone.  Off by default for backward compatibility.
    flow_map_self_conditioning: bool = False
    flow_map_self_cond_prob: float = 0.5
    # Lead-Time-Adaptive Stochasticity (LTAS).  Modulates the per-frame
    # Gaussian noise scale so that near-deterministic short lead times use
    # smaller sigma and highly uncertain long lead times use full sigma.
    # The empirically observed AlphaPre residual variance grows
    # monotonically with lead time on SEVIR; using a matched per-frame
    # noise scale lets the flow map be sharp at short lead and generative
    # at long lead without changing the architecture.  Disabled by
    # default for backward compatibility.
    flow_map_lead_time_stochasticity: bool = False
    flow_map_noise_scale_min: float = 0.3
    flow_map_noise_scale_max: float = 1.0
    # Spectral Sharpness Anchor (SSA).  Adds a high-pass FFT MSE between
    # the GT and the final forecast (backbone + predicted residual) on
    # direct-endpoint samples.  Targets the conditional-mean blur that
    # residual flow-map models inherit from the deterministic backbone
    # (visible as low SSIM / blurred long-lead frames in v1-v15).  The
    # cutoff is expressed as a radial frequency index on the H x W
    # spectrum (e.g. 16 keeps the upper ~92% of frequencies for a 128
    # image).  Disabled by default.
    lambda_ssa: float = 0.0
    ssa_cutoff: int = 16
    # Spectral (PSD) Consistency.  Matches the radially-averaged power
    # spectral density (PSD) of the generated final forecast to the GT
    # on direct-endpoint (r=0) samples.  Unlike SSA -- which only
    # high-pass filters and L2-matches in the image domain -- this term
    # operates on the full 1-D radial energy spectrum (a standard
    # meteorological verification diagnostic) and therefore calibrates
    # sharpness across *all* scales rather than restoring a single
    # high-frequency band.  It is the nowcasting-specific contribution
    # for the standalone Flow-Map model: precipitation fields have a
    # characteristic power-law spectrum that conditional-mean / blurred
    # generators systematically under-shoot at small scales.  Matching in
    # log-space handles the large dynamic range of the spectrum.
    # Disabled by default.
    lambda_psd: float = 0.0

    # Bridge endpoint construction
    prior_kind: str = "last_frame"  # last_frame | mean_past | linear_extrapolation

    # Network
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 4, 8)
    attention_resolutions: tuple[int, ...] = (16, 8)
    num_res_blocks: int = 2
    dropout: float = 0.0
    cond_x1: bool = True  # past frames as concat condition
    # When True AND prior_kind in {'external', 'external_residual'}, the externally
    # supplied forecast (e.g. AlphaPre prediction) is also concatenated to the
    # network's condition channel.  Without this, residual-mode nets see no
    # backbone signal and cannot model the residual structure (we observed this
    # empirically: residual SB without it ≡ vanilla SB).
    cond_extra_x1_external: bool = False
    cond_extra_T_external: int = 0  # optional extra cached priors, e.g. flow warp

    # ---- Physics-informed advection prior (NowcastNet-style evolution) ----
    # When True, a LagrangianSourceNet rolls out the physically-consistent
    # forecast R_{t+1}=warp(R_t,v_t)+s_t from the past frames and the result
    # is concatenated to the generative head's condition (via cond_extra).
    # The deterministic advective prior carries the predictable motion skill
    # while the flow-map head supplies calibrated stochastic detail, which
    # decouples the skill <-> sharpness tradeoff.  The prior occupies the
    # FIRST ``T_out`` slices of cond_extra (cond_extra_T_external is bumped to
    # T_out automatically when enabled).
    motion_prior: bool = False
    motion_prior_base_channels: int = 96
    motion_prior_channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    motion_prior_num_blocks: int = 2
    motion_prior_max_displacement: float = 8.0
    motion_prior_source_scale: float = 0.25
    motion_prior_ckpt: str = ""  # optional pretrained LagrangianSourceNet
    motion_prior_freeze: bool = True  # freeze the prior (fixed physics anchor)
    use_residual_gate: bool = False
    gate_init_bias: float = -1.10
    lambda_gate_l1: float = 0.0

    # Sampling
    nfe: int = 20  # default function evals at sampling time
    residual_scale: float = 1.0  # test-time scale for external_residual
    residual_normalize: bool = False
    residual_transform: str = "pixel"  # pixel | fft_ri | fft_low_detail
    residual_low_freq: int = 20
    residual_norm_min: float = 0.25
    residual_norm_max: float = 2.0
    residual_norm_disagreement: float = 4.0
    residual_posterior_shrinkage: bool = False
    residual_bridge_var: float = 0.0025
    residual_target_posterior_shrinkage: bool = False
    residual_posterior_strength: float = 1.0
    residual_intensity_var_weight: float = 0.0
    residual_intensity_var_power: float = 2.0
    # When True, the second T_out slice of cond_extra_external is treated as a
    # learned per-pixel uncertainty estimate U_phi(past, AlphaPre) ~= |y - y_AP|
    # and used as sigma_bb in BRC instead of the (AlphaPre - motion)^2 proxy.
    use_learned_backbone_uncertainty: bool = False

    # Optional small physics regulariser.  All zero keeps vanilla SB/DDBM.
    lambda_mass: float = 0.0
    lambda_smooth: float = 0.0
    lambda_diffusion: float = 0.0
    diffusion_kappa: float = 0.01
    lambda_final_l1: float = 0.0
    lambda_false_positive: float = 0.0
    lambda_soft_csi: float = 0.0
    soft_csi_sharpness: float = 8.0
    skill_pixel_scale: float = 255.0
    skill_thresholds: tuple[float, ...] = (74.0, 133.0, 160.0, 181.0)
    # Anchor losses that preserve AlphaPre's calibration where it is already
    # correct.  ``lambda_preservation`` penalises adding residual on pixels
    # whose backbone forecast already matches the GT within
    # ``preservation_tolerance`` (in normalised intensity).
    # ``lambda_calibration_mse`` is a small pixel-space MSE between the final
    # forecast and the GT so the gated residual does not drift away from the
    # backbone's low-FAR / low-MSE regime.
    lambda_preservation: float = 0.0
    preservation_tolerance: float = 0.04
    lambda_calibration_mse: float = 0.0

    def unet_cfg(self) -> NowcastUNetConfig:
        cond_extra_T = self._cond_extra_T()
        return NowcastUNetConfig(
            T_in=self.T_in,
            T_out=self.T_out,
            img_channels=self.img_channels,
            bridge_channels=self.bridge_channels,
            img_size=self.img_size,
            base_channels=self.base_channels,
            channel_mult=tuple(self.channel_mult),
            attention_resolutions=tuple(self.attention_resolutions),
            num_res_blocks=self.num_res_blocks,
            dropout=self.dropout,
            interval=self.interval,
            cond_x1=self.cond_x1,
            cond_extra_T=cond_extra_T,
            time_cond_channels=2 if self.bridge_type == "flow_map" else 0,
            self_cond_channels=(
                self.T_out * self.img_channels
                if self.flow_map_self_conditioning and self.bridge_type == "flow_map"
                else 0
            ),
            use_residual_gate=self.use_residual_gate,
            gate_init_bias=self.gate_init_bias,
        )

    def _uses_extra_cond(self) -> bool:
        # x1_external is fed to the UNet as conditioning whenever the user
        # asks for it.  The prior_kind only controls whether the external
        # backbone is added back to the sample at inference time; using it
        # purely as a network input (no add-back) is a valid configuration
        # for generative nowcasters that want backbone *guidance* without
        # the residual-anchor blur transmission of v10 / DiffCast / FlowCast.
        return bool(self.cond_extra_x1_external)

    def _cond_extra_T(self) -> int:
        total = self.T_out if self._uses_extra_cond() else 0
        return total + self.cond_extra_T_external


class PGFMMRunner(nn.Module):
    """All trainable state in one nn.Module so ``accelerate.prepare(runner)`` works."""

    def __init__(self, cfg: PGFMMConfig, device: torch.device | str = "cpu"):
        super().__init__()
        self.cfg = cfg

        # Physics-informed advection prior reserves T_out extra condition
        # channels.  This must happen BEFORE ``unet_cfg()`` is read below.
        if cfg.motion_prior and cfg.cond_extra_T_external < cfg.T_out:
            cfg.cond_extra_T_external = cfg.T_out

        if cfg.bridge_type == "i2sb":
            betas = make_symmetric_beta_schedule(cfg.interval, beta_max=cfg.beta_max)
            self.diffusion = I2SBDiffusion(betas, device="cpu")
        elif cfg.bridge_type == "ddbm":
            self.diffusion = DDBMBridge(
                DDBMConfig(
                    pred_mode=cfg.ddbm_pred_mode,
                    sigma_data=cfg.ddbm_sigma_data,
                    sigma_min=cfg.ddbm_sigma_min,
                    sigma_max=cfg.ddbm_sigma_max,
                    beta_d=cfg.ddbm_beta_d,
                    beta_min=cfg.ddbm_beta_min,
                    cov_xy=cfg.ddbm_cov_xy,
                    rho=cfg.ddbm_rho,
                    weight_schedule=cfg.ddbm_weight_schedule,
                    churn_step_ratio=cfg.ddbm_churn_step_ratio,
                    guidance=cfg.ddbm_guidance,
                    sampler=cfg.ddbm_sampler,
                    eta=cfg.ddbm_eta,
                )
            )
        elif cfg.bridge_type == "flow_map":
            # Flow Map Matching directly learns the two-time solution operator
            # Phi_{t->r}(x_t | cond), so no diffusion/SB helper is required.
            self.diffusion = None
        else:
            raise ValueError(f"unknown bridge_type={cfg.bridge_type!r}")
        self.net = NowcastUNet(cfg.unet_cfg())

        # ---- Physics-informed advection prior (optional) ----
        self.motion_prior = None
        if cfg.motion_prior:
            self.motion_prior = LagrangianSourceNet(
                LagrangianSourceConfig(
                    T_in=cfg.T_in,
                    T_out=cfg.T_out,
                    img_channels=cfg.img_channels,
                    img_size=cfg.img_size,
                    base_channels=cfg.motion_prior_base_channels,
                    channel_mult=tuple(cfg.motion_prior_channel_mult),
                    num_blocks=cfg.motion_prior_num_blocks,
                    max_displacement=cfg.motion_prior_max_displacement,
                    source_scale=cfg.motion_prior_source_scale,
                )
            )
            if cfg.motion_prior_ckpt:
                self._load_motion_prior_ckpt(cfg.motion_prior_ckpt)
            if cfg.motion_prior_freeze:
                for p in self.motion_prior.parameters():
                    p.requires_grad_(False)
                self.motion_prior.eval()

        # Propagate device once at construction; later ``.to(...)`` calls
        # (e.g. accelerator.prepare) work as for any nn.Module.
        if device is not None:
            self.to(torch.device(device))

    @property
    def device(self) -> torch.device:
        """Always reflects where the parameters currently live."""
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Boundary pair ``(x_0, x_1, cond)`` for one batch
    # ------------------------------------------------------------------
    def split_boundary(
        self,
        frames: torch.Tensor,
        x1_external: torch.Tensor | None = None,
        cond_extra_external: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``frames`` shape ``(B, T_in+T_out, C, H, W)`` → ``(x_0, x_1, cond)``.

        * ``x_0`` = the bridge **target** distribution
        * ``x_1`` = the bridge **prior** distribution
        * ``cond`` = past frames (used as concat condition)  ``(B, T_in, C, H, W)``

        Behaviour depends on ``cfg.prior_kind``:

        ``last_frame`` / ``mean_past`` / ``linear_extrapolation``
            x_0 = future ground truth, x_1 built from past frames.

        ``external``
            x_0 = future ground truth, x_1 = caller-provided forecast
            (e.g. AlphaPre).  Bridge transports forecast → ground truth.

        ``external_residual``  (DiffCast-style)
            x_0 = (future ground truth − caller-provided forecast) = residual
            x_1 = zeros (residual prior).  Bridge transports zero → residual.
            The caller-provided forecast is added back at sampling time so
            the deterministic backbone's MSE quality is preserved.
        """
        cfg = self.cfg
        T = cfg.T_in + cfg.T_out
        assert frames.shape[1] == T, f"got T={frames.shape[1]}, expected {T}"
        cond = frames[:, : cfg.T_in].contiguous()
        gt = frames[:, cfg.T_in :].contiguous()

        if cfg.prior_kind in ("external", "external_residual"):
            if x1_external is None:
                raise ValueError(f"prior_kind={cfg.prior_kind!r} requires x1_external")
            x1_external = x1_external.to(gt.device)
            assert x1_external.shape == gt.shape, (
                f"x1_external shape {tuple(x1_external.shape)} != gt shape {tuple(gt.shape)}"
            )

        if cfg.prior_kind == "external":
            x_0 = self._encode_residual(gt)
            x_1 = self._encode_residual(x1_external)
        elif cfg.prior_kind == "external_residual":
            residual = gt - x1_external  # residual r = y − ŷ
            residual = residual / self._residual_norm_scale(
                x1_external,
                cond_extra_external,
                like=residual,
            )
            if cfg.residual_target_posterior_shrinkage:
                residual = residual * self._residual_posterior_weight(
                    x1_external,
                    cond_extra_external,
                    like=residual,
                    force=True,
                )
            x_0 = self._encode_residual(residual)
            x_1 = torch.zeros_like(x_0)  # residual prior is 0
        else:
            x_0 = gt
            x_1 = build_prior(cfg.prior_kind, cond, cfg.T_out)
        return x_0, x_1, cond

    # ------------------------------------------------------------------
    # Single training step (compatible with accelerate `accelerator.backward`)
    # ------------------------------------------------------------------
    def train_step(
        self,
        frames: torch.Tensor,
        x1_external: torch.Tensor | None = None,
        cond_extra_external: torch.Tensor | None = None,
        global_step: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Returns a dict with at least ``total_loss``."""
        cfg = self.cfg
        if self.motion_prior is not None and cond_extra_external is None:
            cond_extra_external = self._resolve_motion_prior(
                frames[:, : cfg.T_in].contiguous(), None
            )
        x_0, x_1, cond = self.split_boundary(
            frames,
            x1_external=x1_external,
            cond_extra_external=cond_extra_external,
        )
        B = x_0.shape[0]

        if cfg.bridge_type == "ddbm":
            return self._train_step_ddbm(
                x_0,
                x_1,
                cond,
                x1_external=x1_external,
                cond_extra_external=cond_extra_external,
            )
        if cfg.bridge_type == "flow_map":
            return self._train_step_flow_map(
                x_0,
                cond,
                x1_external=x1_external,
                cond_extra_external=cond_extra_external,
                global_step=global_step,
            )

        # x_t flattened to (B, T_out*C, H, W) so the bridge math is shape-agnostic
        x_0f = x_0.flatten(1, 2)
        x_1f = x_1.flatten(1, 2)

        # uniformly random bridge step per sample
        step = torch.randint(0, cfg.interval, (B,), device=x_0f.device)

        # forward bridge sample (Eq.11)
        xt = self.diffusion.q_sample(step, x_0f, x_1f, ot_ode=cfg.ot_ode)
        # regression target (Eq.12)
        label = self.diffusion.compute_label(step, x_0f, xt)

        cond_extra = self._build_cond_extra(x1_external, cond_extra_external)

        # network expects 5-D ((B, T_out, C, H, W)); we keep it flat and let the
        # wrapper reshape internally when needed
        pred = self.net(
            xt.unflatten(1, (cfg.T_out, cfg.bridge_channels)),
            step,
            cond=cond,
            cond_extra=cond_extra,
        ).flatten(1, 2)

        loss = F.mse_loss(pred, label)
        total_loss = loss
        log = {"total_loss": total_loss, "mse": loss.detach()}

        if self._uses_physics():
            pred_x0 = self.diffusion.compute_pred_x0(step, xt, pred, clamp=None)
            pred_x0 = pred_x0.unflatten(1, (cfg.T_out, cfg.bridge_channels))
            final_pred = self._to_final_forecast(
                pred_x0,
                x1_external,
                cond_extra_external,
            )
            phys, parts = self._physics_loss_bundle(
                final_pred.clamp(0.0, 1.0),
                lambda_mass=cfg.lambda_mass,
                lambda_smooth=cfg.lambda_smooth,
                lambda_diffusion=cfg.lambda_diffusion,
                diffusion_kappa=cfg.diffusion_kappa,
            )
            total_loss = total_loss + phys
            log = {
                "total_loss": total_loss,
                "mse": loss.detach(),
                "physics": phys.detach(),
                **{f"phys_{k}": v.detach() for k, v in parts.items()},
            }
        return log

    def _uses_physics(self) -> bool:
        cfg = self.cfg
        return cfg.lambda_mass > 0 or cfg.lambda_smooth > 0 or cfg.lambda_diffusion > 0

    def _load_motion_prior_ckpt(self, path: str) -> None:
        """Initialise the advection prior from a pretrained LagrangianSourceNet."""
        import os

        if not os.path.exists(path):
            print(f"[motion_prior] ckpt not found: {path} (random init)")
            return
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "ema" in ck:
            state = {
                k[len("ema_model.") :]: v
                for k, v in ck["ema"].items()
                if k.startswith("ema_model.")
            }
        else:
            state = ck.get("model", ck)
        missing, unexpected = self.motion_prior.load_state_dict(state, strict=False)
        print(
            f"[motion_prior] loaded {path} (missing={len(missing)}, unexpected={len(unexpected)})"
        )

    def _resolve_motion_prior(
        self,
        cond: torch.Tensor,
        cond_extra_external: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Compute the advection prior from ``cond`` and feed it as cond_extra.

        Only synthesises the prior when the model owns a motion module and the
        caller did not already provide an explicit ``cond_extra_external``.
        Runs in fp32 (grid_sample / warp are unstable under autocast) and is
        detached when the prior is frozen.
        """
        if self.motion_prior is None or cond_extra_external is not None:
            return cond_extra_external
        cond = cond.to(self.device)
        frozen = bool(self.cfg.motion_prior_freeze)
        grad_ctx = torch.no_grad() if frozen else contextlib.nullcontext()
        if frozen:
            self.motion_prior.eval()
        with grad_ctx, torch.autocast(device_type=self.device.type, enabled=False):
            prior, _ = self.motion_prior(cond.float())
        prior = prior.to(cond.dtype)
        return prior.detach() if frozen else prior

    def _build_cond_extra(
        self,
        x1_external: torch.Tensor | None,
        cond_extra_external: torch.Tensor | None,
    ) -> torch.Tensor | None:
        parts = []
        if self.cfg._uses_extra_cond():
            assert x1_external is not None
            parts.append(x1_external)
        if self.cfg.cond_extra_T_external > 0:
            assert cond_extra_external is not None, (
                f"cond_extra_T_external={self.cfg.cond_extra_T_external} but no extra condition was provided"
            )
            parts.append(cond_extra_external.to(parts[0].device if parts else self.device))
        if not parts:
            return None
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def _flow_map_time_cond(
        self,
        t: torch.Tensor,
        r: torch.Tensor,
        *,
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Constant input planes carrying the source and destination times.

        The timestep embedding receives ``t``; these extra planes make the
        destination ``r`` explicit, turning the network into a two-time map
        approximator rather than an instantaneous velocity model.
        """
        h, w = like.shape[-2:]
        t_map = t.to(device=like.device, dtype=like.dtype).view(-1, 1, 1, 1).expand(-1, 1, h, w)
        r_map = r.to(device=like.device, dtype=like.dtype).view(-1, 1, 1, 1).expand(-1, 1, h, w)
        return torch.cat([t_map, r_map], dim=1)

    def _flow_map_model_time(self, t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        return (t.to(device=like.device, dtype=like.dtype) * float(self.cfg.interval)).clamp_min(
            1.0e-4
        )

    def _ssa_high_pass(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Radial high-pass filter in 2-D Fourier space.

        Used by the Spectral Sharpness Anchor (SSA).  Given a tensor with
        a trailing ``(H, W)`` spatial dimension we zero out any frequency
        whose normalised radius is below ``cutoff / max(H, W)`` and return
        the real part of the inverse transform.  This isolates the
        high-frequency component of the final forecast that residual
        backbones (AlphaPre) tend to attenuate due to their conditional-
        mean MSE training objective.
        """
        if cutoff <= 0:
            return x
        h, w = x.shape[-2], x.shape[-1]
        device = x.device
        dtype = x.dtype
        fy = torch.fft.fftfreq(h, device=device).abs()
        fx = torch.fft.fftfreq(w, device=device).abs()
        radius = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2)
        norm_cutoff = float(cutoff) / float(max(h, w))
        mask = (radius >= norm_cutoff).to(dtype)
        spec = torch.fft.fft2(x.to(torch.float32))
        filtered = spec * mask.to(spec.dtype)
        out = torch.fft.ifft2(filtered).real.to(dtype)
        return out

    def _radial_psd(self, x: torch.Tensor) -> torch.Tensor:
        """Radially-averaged power spectral density of a 2-D field.

        Used by the Spectral (PSD) Consistency loss.  Given a tensor with
        trailing ``(H, W)`` spatial dims, returns the isotropic power
        spectrum ``(..., n_bins)`` obtained by averaging ``|FFT|^2`` over
        annuli of constant radial wavenumber.  ``n_bins`` is ``max(H, W)
        // 2`` (Nyquist).  Radial bin indices and per-bin counts are
        cached per ``(H, W, device)`` since they are geometry-only.
        """
        h, w = x.shape[-2], x.shape[-1]
        cache = getattr(self, "_psd_bin_cache", None)
        if cache is None or cache[0] != (h, w, x.device):
            fy = torch.fft.fftfreq(h, device=x.device)
            fx = torch.fft.fftfreq(w, device=x.device)
            radius = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2)
            n_bins = max(h, w) // 2
            r_norm = radius / radius.max().clamp_min(1.0e-12)
            bins = torch.clamp((r_norm * (n_bins - 1)).round().long(), 0, n_bins - 1).reshape(-1)
            counts = torch.bincount(bins, minlength=n_bins).clamp_min(1).float()
            cache = ((h, w, x.device), bins, counts, n_bins)
            self._psd_bin_cache = cache
        _, bins, counts, n_bins = cache
        spec = torch.fft.fft2(x.to(torch.float32))
        power = spec.real**2 + spec.imag**2  # (..., H, W)
        flat = power.reshape(*power.shape[:-2], h * w)
        out = power.new_zeros(*power.shape[:-2], n_bins)
        out.index_add_(-1, bins, flat)
        return out / counts

    def _flow_map_noise_scale(self, *, like: torch.Tensor) -> torch.Tensor | float:
        """Per-frame Gaussian noise scale for Flow-Map Matching.

        With ``flow_map_lead_time_stochasticity = False`` this returns the
        scalar ``flow_map_noise_scale`` (legacy behaviour).  When enabled
        (LTAS, Lead-Time-Adaptive Stochasticity) we modulate the per-frame
        sigma linearly from ``noise_scale_min`` (near-deterministic short
        lead) to ``noise_scale_max`` (fully generative long lead).  The
        returned tensor has shape ``(1, T_out * bridge_channels, 1, 1)``
        so it broadcasts against the flattened ``(B, T_out*C, H, W)``
        residual / noise tensors used everywhere in the Flow Map paths.
        """
        cfg = self.cfg
        if not cfg.flow_map_lead_time_stochasticity:
            return float(cfg.flow_map_noise_scale)
        T = int(cfg.T_out)
        C = int(cfg.bridge_channels)
        device = like.device
        dtype = like.dtype
        if T <= 1:
            sigma_per_frame = torch.full(
                (1,),
                float(cfg.flow_map_noise_scale_max),
                device=device,
                dtype=dtype,
            )
        else:
            sigma_per_frame = torch.linspace(
                float(cfg.flow_map_noise_scale_min),
                float(cfg.flow_map_noise_scale_max),
                T,
                device=device,
                dtype=dtype,
            )
        sigma = sigma_per_frame.view(1, T, 1, 1, 1).expand(1, T, C, 1, 1)
        return sigma.reshape(1, T * C, 1, 1)

    def _residual_norm_scale(
        self,
        x1_external: torch.Tensor | None,
        cond_extra_external: torch.Tensor | None,
        *,
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Condition-dependent residual scale for heteroscedastic residual bridges.

        The first extra cache is a deterministic motion/advection prior in the
        AlphaPre protocol runs.  Disagreement between AlphaPre and that prior is
        a threshold-free proxy for where residual uncertainty is large.
        """
        cfg = self.cfg
        scale = like.new_ones(like.shape)
        if not cfg.residual_normalize:
            return scale
        if x1_external is not None and cond_extra_external is not None:
            motion = cond_extra_external.to(like.device)[:, : cfg.T_out]
            disagreement = (x1_external.to(like.device) - motion).abs()
            scale = cfg.residual_norm_min + cfg.residual_norm_disagreement * disagreement
        else:
            scale = scale * cfg.residual_norm_min
        return scale.clamp(cfg.residual_norm_min, cfg.residual_norm_max)

    def _residual_posterior_weight(
        self,
        x1_external: torch.Tensor | None,
        cond_extra_external: torch.Tensor | None,
        *,
        like: torch.Tensor,
        force: bool = False,
    ) -> torch.Tensor:
        """Posterior residual trust from source-vs-motion uncertainty.

        When AlphaPre and the motion prior agree, the source endpoint is treated
        as high precision and the residual correction is shrunk.  When they
        disagree, source uncertainty is larger and the bridge correction is
        allowed to contribute more.
        """
        cfg = self.cfg
        if not (cfg.residual_posterior_shrinkage or force):
            return like.new_ones(like.shape)
        if x1_external is None or cond_extra_external is None:
            return like.new_ones(like.shape)
        source = x1_external.to(like.device)
        extra = cond_extra_external.to(like.device)
        motion = extra[:, : cfg.T_out]
        if cfg.use_learned_backbone_uncertainty:
            assert extra.shape[1] >= 2 * cfg.T_out, (
                "use_learned_backbone_uncertainty requires cond_extra to "
                f"contain at least 2*T_out={2 * cfg.T_out} frames; got "
                f"{extra.shape[1]}"
            )
            uncertainty = extra[:, cfg.T_out : 2 * cfg.T_out]
            source_var = uncertainty.pow(2)
        else:
            source_var = (source - motion).pow(2)
        intensity_w = float(cfg.residual_intensity_var_weight)
        if intensity_w > 0.0:
            # Strong rain has intrinsically larger deterministic forecast
            # uncertainty.  This is a smooth heteroscedastic variance term, not
            # a thresholded CSI/FAR rule.
            intensity = 0.5 * (source.abs() + motion.abs())
            source_var = source_var + intensity_w * intensity.pow(
                float(cfg.residual_intensity_var_power)
            )
        bridge_var = like.new_tensor(float(cfg.residual_bridge_var)).clamp_min(1.0e-8)
        weight = source_var / (source_var + bridge_var)
        if force:
            return weight
        strength = float(cfg.residual_posterior_strength)
        if strength >= 1.0:
            return weight
        if strength <= 0.0:
            return like.new_ones(like.shape)
        return (1.0 - strength) + strength * weight

    def _encode_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """Map pixel residuals to the bridge state space."""
        if self.cfg.residual_transform == "pixel":
            return residual
        if self.cfg.residual_transform == "fft_ri":
            z = torch.fft.fft2(residual.float(), dim=(-2, -1), norm="ortho")
            return torch.cat([z.real, z.imag], dim=2).to(residual.dtype)
        if self.cfg.residual_transform == "fft_low_detail":
            z = torch.fft.fft2(residual.float(), dim=(-2, -1), norm="ortho")
            z_low = z * self._low_frequency_mask(residual)
            low_pixel = torch.fft.ifft2(z_low, dim=(-2, -1), norm="ortho").real
            detail = residual.float() - low_pixel
            return torch.cat([z_low.real, z_low.imag, detail], dim=2).to(residual.dtype)
        raise ValueError(f"unknown residual_transform={self.cfg.residual_transform!r}")

    def _decode_residual(self, residual_state: torch.Tensor) -> torch.Tensor:
        """Map bridge state residuals back to pixel residuals."""
        if self.cfg.residual_transform == "pixel":
            return residual_state
        if self.cfg.residual_transform == "fft_ri":
            if residual_state.shape[2] != 2 * self.cfg.img_channels:
                raise ValueError(
                    "fft_ri residual state must have 2 * img_channels channels, "
                    f"got {residual_state.shape[2]}"
                )
            real, imag = residual_state.chunk(2, dim=2)
            z = torch.complex(real.float(), imag.float())
            return torch.fft.ifft2(z, dim=(-2, -1), norm="ortho").real.to(residual_state.dtype)
        if self.cfg.residual_transform == "fft_low_detail":
            if residual_state.shape[2] != 3 * self.cfg.img_channels:
                raise ValueError(
                    "fft_low_detail residual state must have 3 * img_channels channels, "
                    f"got {residual_state.shape[2]}"
                )
            real, imag, detail = torch.chunk(residual_state, 3, dim=2)
            z_low = torch.complex(real.float(), imag.float())
            z_low = z_low * self._low_frequency_mask(residual_state)
            low_pixel = torch.fft.ifft2(z_low, dim=(-2, -1), norm="ortho").real
            return (low_pixel + detail.float()).to(residual_state.dtype)
        raise ValueError(f"unknown residual_transform={self.cfg.residual_transform!r}")

    def _low_frequency_mask(self, x: torch.Tensor) -> torch.Tensor:
        """AlphaPre-style square low-frequency mask in full FFT coordinates."""
        h, w = x.shape[-2:]
        k = min(int(self.cfg.residual_low_freq), h, w)
        mask = x.new_zeros((1, 1, 1, h, w), dtype=torch.float32)
        mask[..., :k, :k] = 1.0
        mask[..., -k:, :k] = 1.0
        return mask.to(device=x.device)

    @staticmethod
    def _physics_loss_bundle(*args, **kwargs):
        # Lazy import avoids src.jepa.__init__ -> latent_sb -> src.bridge.runner
        # circular imports when the bridge package is imported by itself.
        from pgfmm.losses import physics_loss_bundle

        return physics_loss_bundle(*args, **kwargs)

    def _to_final_forecast(
        self,
        pred_x0: torch.Tensor,
        x1_external: torch.Tensor | None,
        cond_extra_external: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map bridge-space x0 prediction to pixel forecast space."""
        if self.cfg.prior_kind == "external":
            return self._decode_residual(pred_x0)
        if self.cfg.prior_kind == "external_residual":
            assert x1_external is not None
            residual = self._decode_residual(pred_x0)
            scale = self._residual_norm_scale(
                x1_external,
                cond_extra_external,
                like=residual,
            )
            correction = self.cfg.residual_scale * residual * scale
            correction = correction * self._residual_posterior_weight(
                x1_external,
                cond_extra_external,
                like=correction,
            )
            return correction + x1_external.to(pred_x0.device)
        return pred_x0

    def _apply_residual_gate(
        self,
        residual: torch.Tensor,
        cond: torch.Tensor,
        cond_extra: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.cfg.use_residual_gate:
            return residual, None
        gate = self.net.gate(residual, cond=cond, cond_extra=cond_extra)
        return gate * residual, gate

    def _skill_loss_bundle(
        self,
        final_pred: torch.Tensor,
        gt: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        alphapre: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        parts: dict[str, torch.Tensor] = {}
        zero = final_pred.new_zeros(())
        total = zero

        def _weighted_mean(values: torch.Tensor) -> torch.Tensor:
            if sample_weight is None:
                return values.mean()
            w = sample_weight.to(device=values.device, dtype=values.dtype)
            while w.ndim < values.ndim:
                w = w.view(*w.shape, 1)
            denom = (torch.ones_like(values) * w).sum().clamp_min(1.0)
            return (values * w).sum() / denom

        if cfg.lambda_final_l1 > 0:
            # Intensity-aware L1: extreme pixels matter more for CSI thresholds.
            weight = 1.0 + 2.0 * gt.detach()
            l1 = _weighted_mean(weight * (final_pred - gt).abs())
            parts["skill_l1"] = l1
            total = total + cfg.lambda_final_l1 * l1
        else:
            parts["skill_l1"] = zero

        if cfg.lambda_false_positive > 0:
            fp_terms = []
            for thr in cfg.skill_thresholds:
                thr_norm = float(thr) / cfg.skill_pixel_scale
                false_region = (gt.detach() < thr_norm).float()
                fp_terms.append(_weighted_mean(torch.relu(final_pred - thr_norm) * false_region))
            fp = torch.stack(fp_terms).mean()
            parts["skill_fp"] = fp
            total = total + cfg.lambda_false_positive * fp
        else:
            parts["skill_fp"] = zero
        if cfg.lambda_soft_csi > 0:
            csi_losses = []
            pred_scaled = final_pred * cfg.skill_pixel_scale
            gt_scaled = gt.detach() * cfg.skill_pixel_scale
            for thr in cfg.skill_thresholds:
                pred_soft = torch.sigmoid((pred_scaled - float(thr)) / cfg.soft_csi_sharpness)
                gt_bin = (gt_scaled >= float(thr)).float()
                tp = (pred_soft * gt_bin).sum(dim=(-1, -2, -3))
                fp = (pred_soft * (1.0 - gt_bin)).sum(dim=(-1, -2, -3))
                fn = ((1.0 - pred_soft) * gt_bin).sum(dim=(-1, -2, -3))
                soft_csi = (tp + 1e-6) / (tp + fp + fn + 1e-6)
                if sample_weight is None:
                    csi_losses.append(1.0 - soft_csi.mean())
                else:
                    w = sample_weight.to(device=soft_csi.device, dtype=soft_csi.dtype)
                    csi_losses.append(1.0 - (soft_csi * w).sum() / w.sum().clamp_min(1.0))
            soft_csi_loss = torch.stack(csi_losses).mean()
            parts["skill_soft_csi"] = soft_csi_loss
            total = total + cfg.lambda_soft_csi * soft_csi_loss
        else:
            parts["skill_soft_csi"] = zero

        if cfg.lambda_preservation > 0 and alphapre is not None:
            tol = float(cfg.preservation_tolerance)
            already_good = ((alphapre.detach() - gt.detach()).abs() < tol).float()
            residual_added = final_pred - alphapre.detach()
            denom = already_good.sum().clamp_min(1.0)
            pres = (residual_added.abs() * already_good).sum() / denom
            parts["skill_preservation"] = pres
            total = total + cfg.lambda_preservation * pres
        else:
            parts["skill_preservation"] = zero

        if cfg.lambda_calibration_mse > 0:
            cal = _weighted_mean((final_pred - gt) ** 2)
            parts["skill_calibration_mse"] = cal
            total = total + cfg.lambda_calibration_mse * cal
        else:
            parts["skill_calibration_mse"] = zero
        return total, parts

    def _train_step_ddbm(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        cond: torch.Tensor,
        x1_external: torch.Tensor | None = None,
        cond_extra_external: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        x_0f = x_0.flatten(1, 2)
        x_1f = x_1.flatten(1, 2)
        sigmas = self.diffusion.sample_sigmas(x_0f.shape[0], x_0f.device)
        cond_extra = self._build_cond_extra(x1_external, cond_extra_external)

        def model_fn(x_t: torch.Tensor, t_cont: torch.Tensor) -> torch.Tensor:
            return self.net(
                x_t.unflatten(1, (cfg.T_out, cfg.bridge_channels)),
                t_cont,
                cond=cond,
                cond_extra=cond_extra,
            ).flatten(1, 2)

        loss, denoised, parts = self.diffusion.training_losses(
            model_fn, x_0f, x_1f, sigmas, return_xt=cfg.use_residual_gate
        )
        parts.pop("_x_t", None)
        parts.pop("_sigmas", None)
        total_loss = loss
        log = {"total_loss": total_loss, **parts}
        pred_x0 = denoised.unflatten(1, (cfg.T_out, cfg.bridge_channels))
        gated_pred_x0, gate = self._apply_residual_gate(pred_x0, cond, cond_extra)
        final_pred = self._to_final_forecast(
            gated_pred_x0,
            x1_external,
            cond_extra_external,
        ).clamp(0.0, 1.0)
        if cfg.prior_kind in ("external", "external_residual"):
            gt = self._to_final_forecast(x_0, x1_external, cond_extra_external)
        else:
            gt = x_0
        alphapre_anchor = (
            x1_external.clamp(0.0, 1.0)
            if x1_external is not None and cfg.prior_kind in ("external", "external_residual")
            else None
        )
        skill_loss, skill_parts = self._skill_loss_bundle(
            final_pred,
            gt.clamp(0.0, 1.0),
            alphapre=alphapre_anchor,
        )
        gate_loss = pred_x0.new_zeros(())
        if gate is not None and cfg.lambda_gate_l1 > 0:
            gate_loss = gate.mean()
            total_loss = total_loss + cfg.lambda_gate_l1 * gate_loss
        if skill_loss.requires_grad or skill_loss.item() != 0.0:
            total_loss = total_loss + skill_loss
            log = {
                "total_loss": total_loss,
                **parts,
                "skill": skill_loss.detach(),
                **{k: v.detach() for k, v in skill_parts.items()},
            }
            if gate is not None:
                log["gate_mean"] = gate.detach().mean()
                log["gate_l1"] = gate_loss.detach()
        elif gate is not None:
            log["total_loss"] = total_loss
            log["gate_mean"] = gate.detach().mean()
            log["gate_l1"] = gate_loss.detach()
        if self._uses_physics():
            phys, phys_parts = self._physics_loss_bundle(
                final_pred.clamp(0.0, 1.0),
                lambda_mass=cfg.lambda_mass,
                lambda_smooth=cfg.lambda_smooth,
                lambda_diffusion=cfg.lambda_diffusion,
                diffusion_kappa=cfg.diffusion_kappa,
            )
            total_loss = total_loss + phys
            log = {
                "total_loss": total_loss,
                **parts,
                "skill": skill_loss.detach(),
                **{k: v.detach() for k, v in skill_parts.items()},
                "physics": phys.detach(),
                **{f"phys_{k}": v.detach() for k, v in phys_parts.items()},
            }
            if gate is not None:
                log["gate_mean"] = gate.detach().mean()
                log["gate_l1"] = gate_loss.detach()
        return log

    def _train_step_flow_map(
        self,
        x_0: torch.Tensor,
        cond: torch.Tensor,
        x1_external: torch.Tensor | None = None,
        cond_extra_external: torch.Tensor | None = None,
        global_step: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Direct Flow Map Matching for AlphaPre-guided residual nowcasting.

        Instead of learning an instantaneous velocity field and integrating it,
        this objective learns the two-time solution operator
        ``Phi_{t->r}(x_t, t, r | cond)``.  For a source endpoint ``x_1`` and
        target residual ``x_0`` we sample a deterministic interpolant
        ``x_s = (1-s) x_0 + s x_1`` and train the network to map ``x_t`` directly
        to ``x_r`` for randomly sampled ``0 <= r < t <= 1``.  Sampling can then
        use one large shortcut or a few shorter shortcuts with the same model.
        """
        cfg = self.cfg
        x_0f = x_0.flatten(1, 2)
        b = x_0f.shape[0]
        noise_scale = self._flow_map_noise_scale(like=x_0f)
        x_1f = torch.randn_like(x_0f) * noise_scale

        min_delta = max(1.0e-4, min(float(cfg.flow_map_min_delta), 0.95))
        t = min_delta + (1.0 - min_delta) * torch.rand(b, device=x_0f.device)
        # Most samples learn arbitrary shortcuts; some are forced to map directly
        # to the data endpoint so one-step inference is trained explicitly.
        r = torch.rand(b, device=x_0f.device) * (t - min_delta)
        direct_mask = torch.rand(b, device=x_0f.device) < float(cfg.flow_map_direct_prob)
        r = torch.where(direct_mask, torch.zeros_like(r), r)

        view = (b,) + (1,) * (x_0f.ndim - 1)
        t_b = t.view(view)
        r_b = r.view(view)
        x_t = (1.0 - t_b) * x_0f + t_b * x_1f
        x_r = (1.0 - r_b) * x_0f + r_b * x_1f

        cond_extra = self._build_cond_extra(x1_external, cond_extra_external)
        time_cond = self._flow_map_time_cond(t, r, like=x_t)

        # ----- RIN-style self-conditioning warm-up ---------------------
        # For samples drawn with probability ``flow_map_self_cond_prob`` we
        # first run a no-grad forward at (t, r=0) with self_cond=zeros to
        # produce a "coarse" prediction y_hat, then run the main forward
        # with self_cond = stop_grad(y_hat).  The other samples use
        # self_cond = zeros so the model also learns to predict from
        # scratch (matches inference's warm-up pass).
        self_cond = None
        if cfg.flow_map_self_conditioning:
            self_cond = torch.zeros(
                b,
                cfg.T_out * cfg.img_channels,
                *x_t.shape[-2:],
                device=x_t.device,
                dtype=x_t.dtype,
            )
            use_self_cond = torch.rand(b, device=x_t.device) < float(cfg.flow_map_self_cond_prob)
            if int(use_self_cond.sum().item()) > 0:
                with torch.no_grad():
                    warm_t = t
                    warm_r = torch.zeros_like(r)
                    warm_time_cond = self._flow_map_time_cond(
                        warm_t,
                        warm_r,
                        like=x_t,
                    )
                    warm_pred = self.net(
                        x_t.unflatten(1, (cfg.T_out, cfg.bridge_channels)),
                        self._flow_map_model_time(warm_t, like=x_t),
                        cond=cond,
                        cond_extra=cond_extra,
                        time_cond=warm_time_cond,
                        self_cond=self_cond,
                    ).detach()
                # For self-conditioning, the model expects a per-pixel
                # coarse prediction in (B, T_out * img_channels, H, W).
                # In the residual / fft regime bridge_channels may differ
                # from img_channels; we only support pixel-space
                # self-conditioning in v13, where they match.
                assert cfg.bridge_channels == cfg.img_channels, (
                    "flow_map_self_conditioning requires bridge_channels == "
                    "img_channels (pixel-space output)."
                )
                warm_pred_flat = warm_pred.flatten(1, 2)
                mask = use_self_cond.view(-1, 1, 1, 1)
                self_cond = torch.where(mask, warm_pred_flat, self_cond)

        pred = self.net(
            x_t.unflatten(1, (cfg.T_out, cfg.bridge_channels)),
            self._flow_map_model_time(t, like=x_t),
            cond=cond,
            cond_extra=cond_extra,
            time_cond=time_cond,
            self_cond=self_cond,
        ).flatten(1, 2)

        loss = F.mse_loss(pred, x_r)
        total_loss = loss
        consistency = pred.new_zeros(())
        submap_loss = pred.new_zeros(())
        consistency_weight = self._flow_map_consistency_weight(
            global_step,
            device=pred.device,
            dtype=pred.dtype,
        )
        if float(consistency_weight.item()) > 0.0:
            s = r + torch.rand(b, device=x_0f.device) * (t - r)
            s_b = s.view(view)
            x_s = (1.0 - s_b) * x_0f + s_b * x_1f
            pred_ts = self.net(
                x_t.unflatten(1, (cfg.T_out, cfg.bridge_channels)),
                self._flow_map_model_time(t, like=x_t),
                cond=cond,
                cond_extra=cond_extra,
                time_cond=self._flow_map_time_cond(t, s, like=x_t),
            ).flatten(1, 2)
            pred_ts_for_comp = (
                pred_ts.detach() if cfg.flow_map_consistency_detach_midpoint else pred_ts
            )
            pred_sr = self.net(
                pred_ts_for_comp.unflatten(1, (cfg.T_out, cfg.bridge_channels)),
                self._flow_map_model_time(s, like=x_t),
                cond=cond,
                cond_extra=cond_extra,
                time_cond=self._flow_map_time_cond(s, r, like=x_t),
            ).flatten(1, 2)
            # Flow maps are solution operators, so composing two shorter maps
            # should match the single longer map.  The supervised x_s target
            # prevents collapse; this term enforces operator consistency.
            submap_loss = 0.5 * F.mse_loss(pred_ts, x_s) + 0.5 * F.mse_loss(pred_sr, x_r)
            consistency = F.mse_loss(pred_sr, pred.detach())
            total_loss = total_loss + consistency_weight * (submap_loss + consistency)

        # ----- Spectral Sharpness Anchor (SSA) -------------------------
        # On direct-endpoint samples (r=0) the network's prediction IS the
        # predicted residual in bridge space.  Decoding + adding the
        # AlphaPre backbone gives the final pixel-space forecast, and we
        # match its high-frequency Fourier content to the GT's.  This is
        # the third paper contribution -- it directly attacks the
        # conditional-mean blur that residual flow-map models inherit
        # from MSE-trained deterministic backbones, complementing RFM
        # (operator regression) and CC (operator composition).
        ssa_loss = pred.new_zeros(())
        if (
            float(cfg.lambda_ssa) > 0.0
            and cfg.prior_kind == "external_residual"
            and x1_external is not None
        ):
            direct_count_ssa = int(direct_mask.sum().item())
            if direct_count_ssa > 0:
                pred_d = pred[direct_mask].unflatten(1, (cfg.T_out, cfg.bridge_channels))
                x1_d = x1_external.to(pred.device)[direct_mask]
                cond_extra_d = (
                    cond_extra_external.to(pred.device)[direct_mask]
                    if cond_extra_external is not None
                    else None
                )
                gt_d = self._to_final_forecast(
                    x_0[direct_mask]
                    if x_0.ndim == 5
                    else x_0f[direct_mask].unflatten(1, (cfg.T_out, cfg.bridge_channels)),
                    x1_d,
                    cond_extra_d,
                )
                final_pred_d = self._to_final_forecast(
                    pred_d,
                    x1_d,
                    cond_extra_d,
                )
                hp_pred = self._ssa_high_pass(final_pred_d, int(cfg.ssa_cutoff))
                hp_gt = self._ssa_high_pass(gt_d, int(cfg.ssa_cutoff))
                ssa_loss = F.mse_loss(hp_pred, hp_gt)
                total_loss = total_loss + float(cfg.lambda_ssa) * ssa_loss

        # ----- Spectral (PSD) Consistency -------------------------------
        # On direct-endpoint (r=0) samples the prediction is the model's
        # one-step forecast.  We decode it to pixel space and match its
        # radially-averaged power spectrum to the GT's in log-space.  This
        # is the nowcasting-specific contribution: precipitation fields
        # follow a characteristic power-law energy spectrum, and matching
        # it explicitly calibrates sharpness across all scales (countering
        # the small-scale energy deficit of conditional-mean / blurred
        # generators).  Works for both standalone (last_frame) and
        # residual priors via ``_to_final_forecast``.
        psd_loss = pred.new_zeros(())
        if float(cfg.lambda_psd) > 0.0:
            direct_count_psd = int(direct_mask.sum().item())
            if direct_count_psd > 0:
                pred_d = pred[direct_mask].unflatten(1, (cfg.T_out, cfg.bridge_channels))
                x1_d = x1_external.to(pred.device)[direct_mask] if x1_external is not None else None
                cond_extra_d = (
                    cond_extra_external.to(pred.device)[direct_mask]
                    if cond_extra_external is not None
                    else None
                )
                gt_src = (
                    x_0[direct_mask]
                    if x_0.ndim == 5
                    else x_0f[direct_mask].unflatten(1, (cfg.T_out, cfg.bridge_channels))
                )
                gt_d_psd = self._to_final_forecast(gt_src, x1_d, cond_extra_d)
                final_pred_psd = self._to_final_forecast(pred_d, x1_d, cond_extra_d)
                psd_pred = self._radial_psd(final_pred_psd.flatten(1, 2))
                psd_gt = self._radial_psd(gt_d_psd.flatten(1, 2))
                eps = 1.0e-8
                psd_loss = F.mse_loss(torch.log(psd_pred + eps), torch.log(psd_gt + eps))
                total_loss = total_loss + float(cfg.lambda_psd) * psd_loss

        log = {
            "total_loss": total_loss,
            "flow_map_mse": loss.detach(),
            "flow_map_consistency": consistency.detach(),
            "flow_map_submap": submap_loss.detach(),
            "flow_map_consistency_weight": consistency_weight.detach(),
            "flow_map_ssa": ssa_loss.detach(),
            "flow_map_psd": psd_loss.detach(),
            "flow_map_t": t.detach().mean(),
            "flow_map_r": r.detach().mean(),
            "flow_map_direct_frac": direct_mask.float().detach().mean(),
        }

        direct_count = int(direct_mask.sum().item())

        # ----- Energy Score on direct endpoint samples -----------------
        # Proper scoring rule training (Gneiting & Raftery 2007).  For
        # each direct-mask example we draw K samples from the model with
        # fresh noise at the same (t, r=0) and compute
        #     S = mean_k ||X_k - y|| - 0.5 * mean_{k != k'} ||X_k - X_k'||
        # which simultaneously rewards per-sample accuracy (first term)
        # and sample diversity (second term).  The first existing forward
        # pass is reused as the first of the K samples; only K-1 extra
        # forward passes are needed.
        if (
            direct_count > 0
            and cfg.flow_map_energy_score_weight > 0.0
            and cfg.flow_map_energy_score_samples >= 2
        ):
            K = int(cfg.flow_map_energy_score_samples)
            t_d = t[direct_mask]
            r_d = r[direct_mask]
            x_0_d = x_0f[direct_mask]
            cond_d = cond[direct_mask]
            cond_extra_d = cond_extra[direct_mask] if cond_extra is not None else None
            t_d_b = t_d.view((-1,) + (1,) * (x_0_d.ndim - 1))

            # The first sample is just the existing prediction on the
            # direct-mask subset.  Detach? No -- we want the energy score
            # gradient to flow through every sample.
            samples = [pred[direct_mask]]

            for _ in range(K - 1):
                fresh_x1 = torch.randn_like(x_0_d) * self._flow_map_noise_scale(
                    like=x_0_d,
                )
                fresh_x_t = (1.0 - t_d_b) * x_0_d + t_d_b * fresh_x1
                fresh_time_cond = self._flow_map_time_cond(
                    t_d,
                    r_d,
                    like=fresh_x_t,
                )
                pred_k = self.net(
                    fresh_x_t.unflatten(1, (cfg.T_out, cfg.bridge_channels)),
                    self._flow_map_model_time(t_d, like=fresh_x_t),
                    cond=cond_d,
                    cond_extra=cond_extra_d,
                    time_cond=fresh_time_cond,
                ).flatten(1, 2)
                samples.append(pred_k)

            stacked = torch.stack(samples, dim=0)  # (K, N, D)
            gt_d = x_0_d.unsqueeze(0)  # (1, N, D)
            # Per-dimension RMSE-style normalisation so the magnitude is
            # comparable to MSE (otherwise the L2 norm scales as sqrt(D)
            # and the loss dominates the regression term).  This is still
            # a proper scoring rule -- normalising by a constant does not
            # change the optimum of the energy score.
            sample_to_gt = ((stacked - gt_d).flatten(2).pow(2).mean(-1).clamp_min(1e-12)).sqrt()
            term1 = sample_to_gt.mean(dim=0)  # (N,)
            diff = stacked.unsqueeze(0) - stacked.unsqueeze(1)  # (K,K,N,D)
            pair_dist = (diff.flatten(3).pow(2).mean(-1).clamp_min(1e-12)).sqrt()  # (K, K, N)
            K_total = K * (K - 1)
            term2 = 0.5 * pair_dist.sum(dim=(0, 1)) / max(K_total, 1)

            energy_score = (term1 - term2).mean()
            total_loss = total_loss + (float(cfg.flow_map_energy_score_weight) * energy_score)
            log["total_loss"] = total_loss
            log["flow_map_energy_score"] = energy_score.detach()
            log["flow_map_energy_score_term1"] = term1.detach().mean()
            log["flow_map_energy_score_term2"] = term2.detach().mean()

        # ----- Existing direct endpoint skill / BCA losses -------------
        if direct_count > 0 and (
            cfg.use_residual_gate
            or cfg.lambda_gate_l1 > 0.0
            or cfg.lambda_final_l1 > 0.0
            or cfg.lambda_false_positive > 0.0
            or cfg.lambda_soft_csi > 0.0
            or cfg.lambda_preservation > 0.0
            or cfg.lambda_calibration_mse > 0.0
        ):
            pred_endpoint = pred.unflatten(1, (cfg.T_out, cfg.bridge_channels))[direct_mask]
            cond_endpoint = cond[direct_mask]
            x1_endpoint = x1_external[direct_mask] if x1_external is not None else None
            extra_endpoint = (
                cond_extra_external[direct_mask] if cond_extra_external is not None else None
            )
            cond_extra_endpoint = cond_extra[direct_mask] if cond_extra is not None else None
            gated_endpoint, gate = self._apply_residual_gate(
                pred_endpoint,
                cond_endpoint,
                cond_extra_endpoint,
            )
            final_endpoint = self._to_final_forecast(
                gated_endpoint,
                x1_endpoint,
                extra_endpoint,
            ).clamp(0.0, 1.0)
            gt_endpoint = self._to_final_forecast(
                x_0[direct_mask],
                x1_endpoint,
                extra_endpoint,
            ).clamp(0.0, 1.0)
            alphapre_endpoint = x1_endpoint.clamp(0.0, 1.0) if x1_endpoint is not None else None
            skill_loss, skill_parts = self._skill_loss_bundle(
                final_endpoint,
                gt_endpoint,
                alphapre=alphapre_endpoint,
            )
            gate_loss = pred.new_zeros(())
            if gate is not None and cfg.lambda_gate_l1 > 0:
                gate_loss = gate.mean()
                total_loss = total_loss + cfg.lambda_gate_l1 * gate_loss
            if skill_loss.requires_grad or skill_loss.item() != 0.0:
                total_loss = total_loss + skill_loss
            log = {
                **log,
                "total_loss": total_loss,
                "flow_map_endpoint_skill": skill_loss.detach(),
                **{k: v.detach() for k, v in skill_parts.items()},
            }
            if gate is not None:
                log["gate_mean"] = gate.detach().mean()
                log["gate_l1"] = gate_loss.detach()

        if self._uses_physics():
            pred_state = pred.unflatten(1, (cfg.T_out, cfg.bridge_channels))
            final_pred = self._to_final_forecast(
                pred_state,
                x1_external,
                cond_extra_external,
            )
            phys, parts = self._physics_loss_bundle(
                final_pred.clamp(0.0, 1.0),
                lambda_mass=cfg.lambda_mass,
                lambda_smooth=cfg.lambda_smooth,
                lambda_diffusion=cfg.lambda_diffusion,
                diffusion_kappa=cfg.diffusion_kappa,
            )
            total_loss = total_loss + phys
            log = {
                "total_loss": total_loss,
                "flow_map_mse": loss.detach(),
                "physics": phys.detach(),
                **{f"phys_{k}": v.detach() for k, v in parts.items()},
            }
        return log

    def _flow_map_consistency_weight(
        self,
        global_step: int | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Scheduled composition-consistency weight for Flow Map training."""
        cfg = self.cfg
        target = float(cfg.flow_map_consistency_weight)
        if target <= 0.0:
            return torch.zeros((), device=device, dtype=dtype)
        if global_step is None:
            return torch.full((), target, device=device, dtype=dtype)
        start = max(0, int(cfg.flow_map_consistency_start_step))
        if global_step < start:
            return torch.zeros((), device=device, dtype=dtype)
        ramp = max(0, int(cfg.flow_map_consistency_ramp_steps))
        scale = 1.0 if ramp <= 0 else min(1.0, max(0.0, (global_step - start) / float(ramp)))
        return torch.full((), target * scale, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Inference: full reverse-time sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        cond_frames: torch.Tensor,  # (B, T_in, C, H, W)
        x1_external: torch.Tensor | None = None,  # (B, T_out, C, H, W)
        cond_extra_external: torch.Tensor | None = None,
        nfe: int | None = None,
        log_count: int = 1,
        ot_ode: bool | None = None,
        clamp: tuple[float, float] | None = (0.0, 1.0),
        verbose: bool = True,
    ) -> torch.Tensor:
        """Generate ``T_out`` future frames given ``cond_frames``.

        Behaviour depends on ``cfg.prior_kind``:

        ``last_frame`` / ``mean_past`` / ``linear_extrapolation``
            x_1 built from past frames; bridge yields the prediction directly.

        ``external``
            x_1 = ``x1_external`` (e.g. AlphaPre forecast); bridge transports
            forecast → ground truth and returns the result directly.

        ``external_residual``  (DiffCast-style, **most robust for nowcasting**)
            x_1 = zeros, bridge yields the residual r̂; we add ``x1_external``
            back to obtain the final forecast (=  ŷ + r̂).  This preserves
            the backbone's MSE-good behaviour and lets SB only model the
            stochastic high-frequency residual.

        Returns ``(B, T_out, C, H, W)`` clamped to ``clamp`` (default ``[0, 1]``).
        """
        cfg = self.cfg
        nfe = nfe or cfg.nfe
        ot_ode = cfg.ot_ode if ot_ode is None else ot_ode

        device = self.device
        cond = cond_frames.to(device)

        # ---- physics-informed advection prior as conditioning ----
        if self.motion_prior is not None and cond_extra_external is None:
            cond_extra_external = self._resolve_motion_prior(cond, None)

        # ---- build the bridge endpoint x_1 ----
        if cfg.prior_kind == "external":
            if x1_external is None:
                raise ValueError("prior_kind='external' requires x1_external")
            x1_external = x1_external.to(device)
            x_1 = self._encode_residual(x1_external)
            # Spectral bridge states are unconstrained; clamp only after iFFT.
            inner_clamp = clamp if cfg.residual_transform == "pixel" else None
        elif cfg.prior_kind == "external_residual":
            if x1_external is None:
                raise ValueError("prior_kind='external_residual' requires x1_external")
            x1_external = x1_external.to(device)
            x_1 = x1_external.new_zeros(
                x1_external.shape[0],
                cfg.T_out,
                cfg.bridge_channels,
                x1_external.shape[-2],
                x1_external.shape[-1],
            )
            # During SB sampling the predicted x_0 lives in residual-space and
            # can be negative; only clamp at the very end after adding ŷ.
            inner_clamp = None
        else:
            x_1 = build_prior(cfg.prior_kind, cond_frames, cfg.T_out).to(device)
            inner_clamp = clamp

        x_1f = x_1.flatten(1, 2)
        cond_extra = self._build_cond_extra(x1_external, cond_extra_external)

        if cfg.bridge_type == "flow_map":
            # Flow-Map Matching is intrinsically generative; the initial state
            # is always a fresh Gaussian sample regardless of prior_kind.  The
            # prior_kind only controls whether a deterministic backbone is
            # added back to the final prediction afterwards.
            B = x_1f.shape[0]
            H, W = x_1f.shape[-2:]
            init_buffer = torch.randn(
                B,
                cfg.T_out * cfg.bridge_channels,
                H,
                W,
                device=device,
                dtype=x_1f.dtype,
            )
            current = init_buffer * self._flow_map_noise_scale(like=init_buffer)

            # ----- RIN-style self-conditioning warm-up at inference -----
            # Pass 1 (with zeros) gives the model its own coarse prediction;
            # Pass 2 (the main flow-map iteration) consumes it as self_cond.
            # We compute pass 1 as a single direct-endpoint forward at t=1,
            # r=0 (the model is trained to handle this configuration), and
            # then reuse the resulting coarse map for every step of pass 2.
            self_cond = None
            if cfg.flow_map_self_conditioning:
                warm_t = torch.full((B,), 1.0, device=device, dtype=x_1f.dtype)
                warm_r = torch.zeros_like(warm_t)
                warm_time_cond = self._flow_map_time_cond(
                    warm_t,
                    warm_r,
                    like=current,
                )
                warm_self_cond = torch.zeros(
                    B,
                    cfg.T_out * cfg.img_channels,
                    H,
                    W,
                    device=device,
                    dtype=x_1f.dtype,
                )
                warm_pred = self.net(
                    current.unflatten(1, (cfg.T_out, cfg.bridge_channels)),
                    self._flow_map_model_time(warm_t, like=current),
                    cond=cond,
                    cond_extra=cond_extra,
                    time_cond=warm_time_cond,
                    self_cond=warm_self_cond,
                ).flatten(1, 2)
                self_cond = warm_pred

            times = torch.linspace(1.0, 0.0, int(nfe) + 1, device=device)
            for i in range(int(nfe)):
                t = times[i].expand(current.shape[0])
                r = times[i + 1].expand(current.shape[0])
                time_cond = self._flow_map_time_cond(t, r, like=current)
                current = self.net(
                    current.unflatten(1, (cfg.T_out, cfg.bridge_channels)),
                    self._flow_map_model_time(t, like=current),
                    cond=cond,
                    cond_extra=cond_extra,
                    time_cond=time_cond,
                    self_cond=self_cond,
                ).flatten(1, 2)
            final = current.unflatten(1, (cfg.T_out, cfg.bridge_channels))
            if cfg.prior_kind in ("external", "external_residual"):
                final, _ = self._apply_residual_gate(final.to(device), cond, cond_extra)
                final = self._to_final_forecast(final.to(device), x1_external, cond_extra_external)
                if clamp is not None:
                    final = final.clamp(*clamp)
            elif clamp is not None:
                final = final.clamp(*clamp)
            return final

        if cfg.bridge_type == "ddbm":

            def model_fn(x_t: torch.Tensor, t_cont: torch.Tensor) -> torch.Tensor:
                return self.net(
                    x_t.unflatten(1, (cfg.T_out, cfg.bridge_channels)),
                    t_cont,
                    cond=cond,
                    cond_extra=cond_extra,
                ).flatten(1, 2)

            final = self.diffusion.sample(
                model_fn,
                x_1f,
                steps=nfe,
                clip_denoised=(inner_clamp is not None),
                progress=verbose,
            )
            final = final.unflatten(1, (cfg.T_out, cfg.bridge_channels))
            if cfg.prior_kind in ("external", "external_residual"):
                final, _ = self._apply_residual_gate(final.to(device), cond, cond_extra)
                final = self._to_final_forecast(final, x1_external, cond_extra_external)
                if clamp is not None:
                    final = final.clamp(*clamp)
            return final

        # Sub-grid of bridge steps for NFE-step sampling (must include 0 & N-1)
        steps = space_indices(cfg.interval, nfe + 1)
        assert steps[0] == 0 and steps[-1] == cfg.interval - 1

        def pred_x0_fn(xt: torch.Tensor, step_int: int) -> torch.Tensor:
            step = torch.full((xt.shape[0],), step_int, device=xt.device, dtype=torch.long)
            net_out = self.net(
                xt.unflatten(1, (cfg.T_out, cfg.bridge_channels)),
                step,
                cond=cond,
                cond_extra=cond_extra,
            ).flatten(1, 2)
            return self.diffusion.compute_pred_x0(step, xt, net_out, clamp=inner_clamp)

        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        if 0 not in log_steps:
            log_steps = [0] + log_steps[1:]

        xs, _ = self.diffusion.ddpm_sampling(
            steps,
            pred_x0_fn,
            x_1f,
            mask=None,
            ot_ode=ot_ode,
            log_steps=log_steps,
            verbose=verbose,
        )
        final = xs[:, 0]  # (B, T_out*C, H, W)
        final = final.unflatten(1, (cfg.T_out, cfg.bridge_channels))

        # External-residual mode: add the deterministic backbone forecast back.
        if cfg.prior_kind in ("external", "external_residual"):
            final = self._to_final_forecast(final.to(device), x1_external, cond_extra_external)
            if clamp is not None:
                final = final.clamp(*clamp)

        return final
