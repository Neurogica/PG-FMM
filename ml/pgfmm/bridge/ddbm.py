"""DDBM-style diffusion bridge math for nowcasting.

This module is a narrow, auditable port of the official DDBM implementation:

    ext_repos/DDBM/ddbm/karras_diffusion.py

We keep only the pieces needed by our existing nowcasting runner:

* bridge forward sample and preconditioned denoising from ``KarrasDenoiser``
* bridge-karras loss weighting
* deterministic Heun sampler used by DDBM sampling scripts

All data plumbing, conditioning, ``external_residual`` semantics, EMA, metrics,
and AlphaPre-compatible protocol stay in our existing pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Append singleton dims until ``x.ndim == target_dims``."""
    while x.ndim < target_dims:
        x = x[..., None]
    return x


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Mean over all non-batch dimensions."""
    return x.mean(dim=tuple(range(1, x.ndim)))


def append_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.new_zeros([1])])


def vp_logsnr(t: torch.Tensor | float, beta_d: float, beta_min: float) -> torch.Tensor:
    t = torch.as_tensor(t)
    return -torch.log((0.5 * beta_d * (t**2) + beta_min * t).exp() - 1)


def vp_logs(t: torch.Tensor | float, beta_d: float, beta_min: float) -> torch.Tensor:
    t = torch.as_tensor(t)
    return -0.25 * t**2 * beta_d - 0.5 * t * beta_min


@dataclass
class DDBMConfig:
    pred_mode: str = "vp"
    sigma_data: float = 0.5
    sigma_min: float = 1.0e-4
    sigma_max: float = 1.0
    beta_d: float = 2.0
    beta_min: float = 0.1
    cov_xy: float = 0.0
    rho: float = 7.0
    weight_schedule: str = "bridge_karras"
    churn_step_ratio: float = 0.0
    guidance: float = 1.0
    sampler: str = "dbim"  # heun | dbim
    eta: float = 0.0


class DDBMBridge(nn.Module):
    """Official DDBM bridge equations wrapped as an ``nn.Module``.

    The wrapped model predicts the DDBM network output. ``denoise`` then applies
    the official preconditioning:

        denoised = c_out(t) * model(c_in(t) * x_t, t) + c_skip(t) * x_t

    where ``denoised`` is the estimate of the target endpoint ``x_0``.
    """

    def __init__(self, cfg: DDBMConfig | None = None):
        super().__init__()
        self.cfg = cfg or DDBMConfig()
        self.num_timesteps = 40  # official DDBM logging convention

    @property
    def pred_mode(self) -> str:
        return self.cfg.pred_mode

    @property
    def sigma_min(self) -> float:
        return self.cfg.sigma_min

    @property
    def sigma_max(self) -> float:
        return self.cfg.sigma_max

    @property
    def beta_d(self) -> float:
        return self.cfg.beta_d

    @property
    def beta_min(self) -> float:
        return self.cfg.beta_min

    def sample_sigmas(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Official ``real-uniform`` sigma sampler."""
        return (
            torch.rand(batch_size, device=device) * (self.cfg.sigma_max - self.cfg.sigma_min)
            + self.cfg.sigma_min
        )

    def get_snr(self, sigmas: torch.Tensor) -> torch.Tensor:
        if self.cfg.pred_mode.startswith("vp"):
            return vp_logsnr(sigmas, self.cfg.beta_d, self.cfg.beta_min).exp()
        return sigmas**-2

    def get_weightings(self, sigma: torch.Tensor) -> torch.Tensor:
        snrs = self.get_snr(sigma)
        if self.cfg.weight_schedule.startswith("bridge_karras") and self.cfg.pred_mode in {
            "vp",
            "ve",
        }:
            _, c_out, _ = self.get_bridge_scalings(sigma)
            return 1.0 / c_out.clamp_min(1e-12) ** 2
        if self.cfg.weight_schedule == "snr":
            return snrs
        if self.cfg.weight_schedule == "snr+1":
            return snrs + 1
        if self.cfg.weight_schedule == "karras":
            return snrs + 1.0 / self.cfg.sigma_data**2
        if self.cfg.weight_schedule.startswith("bridge_karras"):
            if self.cfg.pred_mode == "ve":
                smax2 = self.cfg.sigma_max**2
                A = (
                    sigma**4 / self.cfg.sigma_max**4 * self.cfg.sigma_data**2
                    + (1 - sigma**2 / smax2) ** 2 * self.cfg.sigma_data**2
                    + 2 * sigma**2 / smax2 * (1 - sigma**2 / smax2) * self.cfg.cov_xy
                    + sigma**2 * (1 - sigma**2 / smax2)
                )
                denom = (sigma / self.cfg.sigma_max) ** 4 * (
                    self.cfg.sigma_data**4 - self.cfg.cov_xy**2
                ) + self.cfg.sigma_data**2 * sigma**2 * (1 - sigma**2 / smax2)
                return A / denom
            if self.cfg.pred_mode == "vp":
                logsnr_t = vp_logsnr(sigma, self.cfg.beta_d, self.cfg.beta_min)
                logsnr_T = vp_logsnr(1, self.cfg.beta_d, self.cfg.beta_min).to(sigma.device)
                logs_t = vp_logs(sigma, self.cfg.beta_d, self.cfg.beta_min)
                logs_T = vp_logs(1, self.cfg.beta_d, self.cfg.beta_min).to(sigma.device)
                a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
                b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()
                A = (
                    a_t**2 * self.cfg.sigma_data**2
                    + b_t**2 * self.cfg.sigma_data**2
                    + 2 * a_t * b_t * self.cfg.cov_xy
                    + c_t
                )
                denom = (
                    a_t**2 * (self.cfg.sigma_data**4 - self.cfg.cov_xy**2)
                    + self.cfg.sigma_data**2 * c_t
                )
                return A / denom
            if self.cfg.pred_mode in {"vp_simple", "ve_simple"}:
                return torch.ones_like(snrs)
        if self.cfg.weight_schedule == "uniform":
            return torch.ones_like(snrs)
        raise NotImplementedError(self.cfg.weight_schedule)

    def get_bridge_scalings(
        self, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.pred_mode in {"vp", "ve"}:
            a_t, b_t, c_t = self.get_abc(sigma)
            A = (
                a_t**2 * self.cfg.sigma_data**2
                + b_t**2 * self.cfg.sigma_data**2
                + 2 * a_t * b_t * self.cfg.cov_xy
                + c_t**2
            )
            c_in = 1 / A.sqrt()
            c_skip = (b_t * self.cfg.sigma_data**2 + a_t * self.cfg.cov_xy) / A
            c_out = (
                a_t**2 * (self.cfg.sigma_data**4 - self.cfg.cov_xy**2)
                + self.cfg.sigma_data**2 * c_t**2
            ).sqrt() * c_in
            return c_skip, c_out, c_in

        if self.cfg.pred_mode == "ve":
            smax2 = self.cfg.sigma_max**2
            A = (
                sigma**4 / self.cfg.sigma_max**4 * self.cfg.sigma_data**2
                + (1 - sigma**2 / smax2) ** 2 * self.cfg.sigma_data**2
                + 2 * sigma**2 / smax2 * (1 - sigma**2 / smax2) * self.cfg.cov_xy
                + sigma**2 * (1 - sigma**2 / smax2)
            )
            c_in = 1 / A.sqrt()
            c_skip = (
                (1 - sigma**2 / smax2) * self.cfg.sigma_data**2 + sigma**2 / smax2 * self.cfg.cov_xy
            ) / A
            c_out = (
                (sigma / self.cfg.sigma_max) ** 4 * (self.cfg.sigma_data**4 - self.cfg.cov_xy**2)
                + self.cfg.sigma_data**2 * sigma**2 * (1 - sigma**2 / smax2)
            ).sqrt() * c_in
            return c_skip, c_out, c_in

        if self.cfg.pred_mode == "vp":
            logsnr_t = vp_logsnr(sigma, self.cfg.beta_d, self.cfg.beta_min)
            logsnr_T = vp_logsnr(1, self.cfg.beta_d, self.cfg.beta_min).to(sigma.device)
            logs_t = vp_logs(sigma, self.cfg.beta_d, self.cfg.beta_min)
            logs_T = vp_logs(1, self.cfg.beta_d, self.cfg.beta_min).to(sigma.device)
            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()
            A = (
                a_t**2 * self.cfg.sigma_data**2
                + b_t**2 * self.cfg.sigma_data**2
                + 2 * a_t * b_t * self.cfg.cov_xy
                + c_t
            )
            c_in = 1 / A.sqrt()
            c_skip = (b_t * self.cfg.sigma_data**2 + a_t * self.cfg.cov_xy) / A
            c_out = (
                a_t**2 * (self.cfg.sigma_data**4 - self.cfg.cov_xy**2)
                + self.cfg.sigma_data**2 * c_t
            ).sqrt() * c_in
            return c_skip, c_out, c_in

        if self.cfg.pred_mode in {"ve_simple", "vp_simple"}:
            return torch.zeros_like(sigma), torch.ones_like(sigma), torch.ones_like(sigma)

        raise NotImplementedError(self.cfg.pred_mode)

    def get_alpha_rho(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """DBIM official noise-schedule coefficients.

        Ported from ``ext_repos/DBIM/ddbm/karras_diffusion.py``:
        ``VPNoiseSchedule.get_alpha_rho`` / ``VENoiseSchedule.get_alpha_rho``.
        """
        t = t.to(torch.float64)
        if self.cfg.pred_mode.startswith("vp"):
            alpha_t = torch.exp(-0.5 * self.cfg.beta_min * t - 0.25 * self.cfg.beta_d * t**2)
            alpha_T = torch.exp(
                torch.as_tensor(
                    -0.5 * self.cfg.beta_min * self.cfg.sigma_max
                    - 0.25 * self.cfg.beta_d * self.cfg.sigma_max**2,
                    device=t.device,
                    dtype=t.dtype,
                )
            )
            rho_t = (torch.exp(self.cfg.beta_min * t + 0.5 * self.cfg.beta_d * t**2) - 1).sqrt()
            rho_T = (
                torch.exp(
                    torch.as_tensor(
                        self.cfg.beta_min * self.cfg.sigma_max
                        + 0.5 * self.cfg.beta_d * self.cfg.sigma_max**2,
                        device=t.device,
                        dtype=t.dtype,
                    )
                )
                - 1
            ).sqrt()
        elif self.cfg.pred_mode.startswith("ve"):
            alpha_t = torch.ones_like(t)
            alpha_T = torch.ones((), device=t.device, dtype=t.dtype)
            rho_t = t
            rho_T = torch.as_tensor(self.cfg.sigma_max, device=t.device, dtype=t.dtype)
        else:
            raise NotImplementedError(self.cfg.pred_mode)
        alpha_bar_t = alpha_t / alpha_T
        rho_bar_t = (rho_T**2 - rho_t**2).clamp_min(0).sqrt()
        return (
            alpha_t.to(torch.float32),
            alpha_bar_t.to(torch.float32),
            rho_t.to(torch.float32),
            rho_bar_t.to(torch.float32),
        )

    def get_abc(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        if self.cfg.pred_mode.startswith("vp"):
            rho_T = (
                torch.exp(
                    torch.as_tensor(
                        self.cfg.beta_min * self.cfg.sigma_max
                        + 0.5 * self.cfg.beta_d * self.cfg.sigma_max**2,
                        device=t.device,
                    )
                )
                - 1
            ).sqrt()
        else:
            rho_T = torch.as_tensor(self.cfg.sigma_max, device=t.device)
        a_t = (alpha_bar_t * rho_t**2) / rho_T**2
        b_t = (alpha_t * rho_bar_t**2) / rho_T**2
        c_t = (alpha_t * rho_bar_t * rho_t) / rho_T
        return a_t, b_t, c_t

    def q_sample(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        sigmas: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Official DDBM bridge sample ``q(x_t | x_0, x_T)``."""
        if noise is None:
            noise = torch.randn_like(x0)
        a_t, b_t, c_t = [
            append_dims(item, x0.ndim)
            for item in self.get_abc(sigmas.clamp(max=self.cfg.sigma_max))
        ]
        return a_t * xT + b_t * x0 + c_t * noise

    def denoise(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_t: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_skip, c_out, c_in = [append_dims(x, x_t.ndim) for x in self.get_bridge_scalings(sigmas)]
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = model_fn(c_in * x_t, rescaled_t)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised

    def training_losses(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_start: torch.Tensor,
        xT: torch.Tensor,
        sigmas: torch.Tensor,
        return_xt: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x_t = self.q_sample(x_start, xT, sigmas)
        _, denoised = self.denoise(model_fn, x_t, sigmas)
        weights = append_dims(self.get_weightings(sigmas), x_start.ndim)
        xs_mse = mean_flat((denoised - x_start) ** 2)
        mse = mean_flat(weights * (denoised - x_start) ** 2)
        parts = {"mse": mse.mean().detach(), "xs_mse": xs_mse.mean().detach()}
        if return_xt:
            parts["_x_t"] = x_t.detach()
            parts["_sigmas"] = sigmas.detach()
        return mse.mean(), denoised, parts

    def get_sigmas_karras(self, n: int, device: torch.device) -> torch.Tensor:
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.cfg.sigma_min ** (1 / self.cfg.rho)
        max_inv_rho = (self.cfg.sigma_max - 1e-4) ** (1 / self.cfg.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.cfg.rho
        return append_zero(sigmas)

    def get_sigmas_uniform(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(self.cfg.sigma_max - 1.0e-3, self.cfg.sigma_min, n + 1, device=device)

    def _to_d(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        denoised: torch.Tensor,
        xT: torch.Tensor,
    ) -> torch.Tensor:
        grad_pxtlx0 = (denoised - x) / append_dims(sigma**2, x.ndim)
        grad_pxTlxt = (xT - x) / (
            append_dims(torch.ones_like(sigma) * self.cfg.sigma_max**2, x.ndim)
            - append_dims(sigma**2, x.ndim)
        )
        gt2 = 2 * sigma
        return -0.5 * gt2 * (grad_pxtlx0 - self.cfg.guidance * grad_pxTlxt)

    def _get_d_vp(
        self,
        x: torch.Tensor,
        denoised: torch.Tensor,
        xT: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        beta_d = self.cfg.beta_d
        beta_min = self.cfg.beta_min
        sigma_max = self.cfg.sigma_max

        def vp_snr_sqrt_reciprocal(z):
            return (np.e ** (0.5 * beta_d * (z**2) + beta_min * z) - 1) ** 0.5

        def vp_snr_sqrt_reciprocal_deriv(z):
            return (
                0.5
                * (beta_min + beta_d * z)
                * (vp_snr_sqrt_reciprocal(z) + 1 / vp_snr_sqrt_reciprocal(z))
            )

        def s(z):
            return (1 + vp_snr_sqrt_reciprocal(z) ** 2).rsqrt()

        def s_deriv(z):
            return -vp_snr_sqrt_reciprocal(z) * vp_snr_sqrt_reciprocal_deriv(z) * (s(z) ** 3)

        def logs(z):
            return -0.25 * z**2 * beta_d - 0.5 * z * beta_min

        def std(z):
            return vp_snr_sqrt_reciprocal(z) * s(z)

        def logsnr(z):
            return -2 * torch.log(vp_snr_sqrt_reciprocal(z))

        logsnr_t = logsnr(t)
        logsnr_T = logsnr(torch.as_tensor(sigma_max, device=x.device))
        logs_t = logs(t)
        logs_T = logs(torch.as_tensor(sigma_max, device=x.device))
        std_t = std(t)
        sigma_t = vp_snr_sqrt_reciprocal(t)
        sigma_t_deriv = vp_snr_sqrt_reciprocal_deriv(t)

        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        mu_t = append_dims(a_t, x.ndim) * xT + append_dims(b_t, x.ndim) * denoised
        grad_logq = (
            -(x - mu_t)
            / append_dims(std_t**2, x.ndim)
            / append_dims(-torch.expm1(logsnr_T - logsnr_t), x.ndim)
        )
        grad_logpxTlxt = (
            -(x - append_dims(torch.exp(logs_t - logs_T), x.ndim) * xT)
            / append_dims(std_t**2, x.ndim)
            / append_dims(torch.expm1(logsnr_t - logsnr_T), x.ndim)
        )
        f = append_dims(s_deriv(t) * (-logs_t).exp(), x.ndim) * x
        gt2 = 2 * (logs_t).exp() ** 2 * sigma_t * sigma_t_deriv
        return f - append_dims(gt2, x.ndim) * (0.5 * grad_logq - self.cfg.guidance * grad_logpxTlxt)

    @torch.no_grad()
    def sample_dbim(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        xT: torch.Tensor,
        steps: int,
        eta: float | None = None,
        clip_denoised: bool = True,
        progress: bool = False,
    ) -> torch.Tensor:
        """DBIM first-order sampler.

        Ported from ``ext_repos/DBIM/ddbm/karras_diffusion.py::sample_dbim``.
        ``eta=0`` gives deterministic implicit sampling; ``eta>0`` injects
        stochasticity while preserving the same marginal bridge family.
        """
        eta = self.cfg.eta if eta is None else eta
        ts = self.get_sigmas_uniform(steps, device=xT.device)
        x = xT
        x_T = xT
        ones = x.new_ones([x.shape[0]])
        indices = range(len(ts) - 1)
        if progress:
            indices = tqdm(indices, desc="DBIM")

        def denoiser(x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            _, denoised = self.denoise(model_fn, x_t, sigma)
            return denoised.clamp(-1, 1) if clip_denoised else denoised

        x0_hat = denoiser(x, self.cfg.sigma_max * ones)
        noise = torch.randn_like(x0_hat)
        a_0, b_0, c_0 = [append_dims(item, x.ndim) for item in self.get_abc(ts[0] * ones)]
        x = a_0 * x_T + b_0 * x0_hat + c_0 * noise

        for i in indices:
            s = ts[i]
            t = ts[i + 1]
            x0_hat = denoiser(x, s * ones)
            a_s, b_s, c_s = [append_dims(item, x.ndim) for item in self.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x.ndim) for item in self.get_abc(t * ones)]
            _, _, rho_s, _ = [append_dims(item, x.ndim) for item in self.get_alpha_rho(s * ones)]
            alpha_t, _, rho_t, _ = [
                append_dims(item, x.ndim) for item in self.get_alpha_rho(t * ones)
            ]
            omega_st = (
                eta
                * (alpha_t * rho_t)
                * (1 - rho_t**2 / rho_s.clamp_min(1e-12) ** 2).clamp_min(0).sqrt()
            )
            tmp_var = (c_t**2 - omega_st**2).clamp_min(0).sqrt() / c_s.clamp_min(1e-12)
            coeff_xs = tmp_var
            coeff_x0_hat = b_t - tmp_var * b_s
            coeff_xT = a_t - tmp_var * a_s
            noise = torch.randn_like(x0_hat)
            is_last = i == len(ts) - 2
            x = (
                coeff_x0_hat * x0_hat
                + coeff_xT * x_T
                + coeff_xs * x
                + (0 if is_last else 1) * omega_st * noise
            )
        return x.clamp(-1, 1) if clip_denoised else x

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        xT: torch.Tensor,
        steps: int,
        clip_denoised: bool = True,
        progress: bool = False,
    ) -> torch.Tensor:
        if self.cfg.sampler == "heun":
            return self.sample_heun(model_fn, xT, steps, clip_denoised, progress)
        if self.cfg.sampler == "dbim":
            return self.sample_dbim(
                model_fn, xT, steps, clip_denoised=clip_denoised, progress=progress
            )
        raise NotImplementedError(self.cfg.sampler)

    @torch.no_grad()
    def sample_heun(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        xT: torch.Tensor,
        steps: int,
        clip_denoised: bool = True,
        progress: bool = False,
    ) -> torch.Tensor:
        """Official DDBM deterministic Heun sampler."""
        sigmas = self.get_sigmas_karras(steps, device=xT.device)
        x = xT
        x_T = xT
        s_in = x.new_ones([x.shape[0]])
        indices = range(len(sigmas) - 1)
        if progress:
            indices = tqdm(indices, desc="DDBM Heun")

        def denoiser(x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            _, denoised = self.denoise(model_fn, x_t, sigma)
            return denoised.clamp(-1, 1) if clip_denoised else denoised

        for i in indices:
            sigma_hat = sigmas[i]
            denoised = denoiser(x, sigma_hat * s_in)
            if self.cfg.pred_mode == "ve":
                d = self._to_d(x, sigma_hat * s_in, denoised, x_T)
            elif self.cfg.pred_mode.startswith("vp"):
                d = self._get_d_vp(x, denoised, x_T, sigma_hat * s_in)
            else:
                raise NotImplementedError(self.cfg.pred_mode)
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                x = x + d * dt
            else:
                x_2 = x + d * dt
                denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
                if self.cfg.pred_mode == "ve":
                    d_2 = self._to_d(x_2, sigmas[i + 1] * s_in, denoised_2, x_T)
                else:
                    d_2 = self._get_d_vp(x_2, denoised_2, x_T, sigmas[i + 1] * s_in)
                x = x + (d + d_2) / 2 * dt
        return x.clamp(-1, 1) if clip_denoised else x
