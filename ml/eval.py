"""Evaluate a trained PG-FMM checkpoint using AlphaPre's Evaluator.

Reusing AlphaPre's evaluator (from ``ext_repos/AlphaPre/utils/metrics.py``)
guarantees the numbers we report are directly comparable to AlphaPre /
DiffCast / DuoCast paper Tables (CSI / FAR / POD / HSS / SSIM / CRPS / LPIPS
with the same thresholds and the same pooling).

Examples
--------
# Quick val-loss check on 50 val batches
uv run python ml/eval.py \
    --ckpt experiments/sevir/Exps/pgfmm_sevir_v1_full/checkpoints/last.pt \
    --config experiments/sevir/Exps/pgfmm_sevir_v1_full/config.yaml \
    --split val --nfe 20 --batch_size 8 --max_batches 50 --use_ema

# Full SEVIR test eval (matches the AlphaPre baseline run)
uv run python ml/eval.py \
    --ckpt experiments/sevir/Exps/pgfmm_sevir_v1_full/checkpoints/best.pt \
    --config experiments/sevir/Exps/pgfmm_sevir_v1_full/config.yaml \
    --split test --nfe 20 --batch_size 8 --use_ema \
    2>&1 | tee logs/pgfmm_v1_full_eval.log
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent  # .../PG-FMM/ml
ROOT = HERE.parent  # .../PG-FMM
sys.path.insert(0, str(HERE))  # make `import pgfmm` importable

# Make sure lpips' alexnet weights land in a writable cache dir (the default
# ~/.cache/torch may be read-only on shared boxes).
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

# AlphaPre's evaluator uses lpips, whose current AlexNet wrapper still calls
# torchvision with the deprecated ``pretrained=`` argument.  Keep evaluation
# logs focused while preserving the official AlphaPre metric implementation.
warnings.filterwarnings(
    "ignore",
    message="The parameter 'pretrained' is deprecated since 0.13.*",
    category=UserWarning,
    module="torchvision.models._utils",
)
warnings.filterwarnings(
    "ignore",
    message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13.*",
    category=UserWarning,
    module="torchvision.models._utils",
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pgfmm.model import PGFMMConfig, PGFMMRunner  # noqa: E402
from pgfmm.data import PIXEL_SCALES, THRESHOLDS, dataset_kwargs_from_cfg, get_dataset  # noqa: E402
from pgfmm.data.paired import MultiCachePairedDataset, PairedDataset  # noqa: E402


# ---------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt", required=True, type=Path, help="Checkpoint .pt produced by train.py"
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="OmegaConf .yaml from the same training run (usually <Exps>/<note>/config.yaml)",
    )
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument(
        "--nfe", type=int, default=20, help="Number of function evaluations at sampling time"
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--max_batches", type=int, default=None, help="Stop after this many batches (debug only)"
    )
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--use_ema", action="store_true", help="Sample from the EMA-averaged weights")
    p.add_argument(
        "--ot_ode",
        action="store_true",
        help="Legacy deterministic-sampler flag; no-op for the flow-map sampler (kept for CLI compatibility)",
    )
    p.add_argument("--override", nargs="*", default=[], help="OmegaConf-style key=value overrides")
    p.add_argument(
        "--csv_out",
        type=Path,
        default=None,
        help="Append a one-row metric summary CSV for comparison tables",
    )
    p.add_argument("--run_name", default=None, help="Human-readable run name written to --csv_out")
    p.add_argument("--method", default=None, help="Method label written to --csv_out")
    p.add_argument(
        "--per_lead_out",
        type=Path,
        default=None,
        help="If set, dump per-lead-time CSI/HSS per threshold to "
        "this .npz (one (T_out,) vector per threshold). Used for "
        "the prior-type ablation lead-time figure.",
    )
    p.add_argument(
        "--ensemble",
        type=int,
        default=1,
        help="Number of stochastic samples to draw per input and "
        "average (Monte-Carlo integration over the flow-map's "
        "initial noise).  ensemble=1 disables averaging.",
    )
    p.add_argument(
        "--ensemble_seed",
        type=int,
        default=0,
        help="Base RNG seed for the ensemble; offset by member idx.",
    )
    p.add_argument(
        "--pmm_sweep",
        action="store_true",
        help="Score several mean<->PMM blend weights in a SINGLE "
        "sampling pass.  For each alpha in --pmm_weights the "
        "output is alpha*PMM + (1-alpha)*mean; one CSV row is "
        "written per alpha (method suffixed with -pmmADD).  "
        "alpha=0 is the plain mean, alpha=1 the full PMM.",
    )
    p.add_argument(
        "--pmm_weights",
        nargs="*",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Blend weights swept by --pmm_sweep.",
    )
    p.add_argument(
        "--pmm_weight",
        type=float,
        default=1.0,
        help="For --ensemble_agg pmm (no sweep): output is "
        "w*PMM + (1-w)*mean.  w=1 full PMM, w=0 plain mean.",
    )
    p.add_argument(
        "--pred_gain",
        type=float,
        default=1.0,
        help="Test-time intensity calibration: pred -> clamp(gain * "
        "pred ** gamma).  gain=gamma=1 is a no-op.",
    )
    p.add_argument(
        "--pred_gamma",
        type=float,
        default=1.0,
        help="Test-time gamma correction exponent (see --pred_gain).",
    )
    p.add_argument(
        "--cal_gains",
        nargs="*",
        type=float,
        default=None,
        help="With --pmm_sweep: also sweep these intensity gains in "
        "the SAME sampling pass (cartesian product with "
        "--cal_gammas and --pmm_weights).",
    )
    p.add_argument(
        "--cal_gammas",
        nargs="*",
        type=float,
        default=None,
        help="With --pmm_sweep: gamma grid swept alongside --cal_gains.",
    )
    p.add_argument(
        "--prior_only",
        action="store_true",
        help="Ablation T4: skip the generative head and evaluate the "
        "frozen motion-prior (Lagrangian) rollout directly. "
        "Requires a config with motion.enabled=true.",
    )
    p.add_argument(
        "--ensemble_agg",
        default="mean",
        choices=["mean", "pmm"],
        help="Ensemble aggregation. 'mean' is the plain ensemble "
        "average (smooths intensity peaks -> low high-threshold "
        "POD/CSI). 'pmm' is the Probability-Matched Mean (Ebert "
        "2001): keep the ensemble-mean spatial pattern but "
        "re-assign the pooled member intensity distribution by "
        "rank, restoring heavy-rain peaks while preserving the "
        "smooth light-rain placement.",
    )
    return p.parse_args()


def calibrate(pred: torch.Tensor, gain: float, gamma: float) -> torch.Tensor:
    """Test-time intensity calibration: clamp(gain * pred ** gamma) in [0, 1]."""
    if gamma != 1.0:
        pred = pred.clamp_min(0.0) ** gamma
    if gain != 1.0:
        pred = pred * gain
    return pred.clamp_(0.0, 1.0)


def probability_matched_mean(samples: torch.Tensor) -> torch.Tensor:
    """Probability-Matched Mean (Ebert 2001) over an ensemble.

    ``samples`` has shape ``(K, B, T, C, H, W)``.  For each (B, T, C) field we

      1. take the ensemble-mean spatial pattern (good placement, smooth),
      2. pool all ``K * H * W`` member values and subsample the sorted pool to
         ``H * W`` representative intensities (restores the member intensity
         distribution, i.e. the heavy-rain peaks the mean washes out),
      3. assign those intensities back to the grid by the rank order of the
         ensemble mean.

    Returns the PMM field of shape ``(B, T, C, H, W)``.
    """
    K, B, T, C, H, W = samples.shape
    n = H * W
    flat = samples.reshape(K, B, T, C, n)
    mean = flat.mean(dim=0)  # (B, T, C, n)

    # Representative intensity distribution: sort the pooled K*n values
    # (descending) and take every K-th, yielding exactly n values.
    pooled = flat.permute(1, 2, 3, 0, 4).reshape(B, T, C, K * n)
    pooled_sorted, _ = torch.sort(pooled, dim=-1, descending=True)
    representative = pooled_sorted[..., ::K][..., :n]  # (B, T, C, n)

    # Rank order of the mean (descending): order[..., j] is the flat pixel that
    # holds the j-th largest mean value.  Scatter the j-th largest pooled
    # intensity there.
    order = torch.argsort(mean, dim=-1, descending=True)  # (B, T, C, n)
    out = torch.empty_like(mean)
    out.scatter_(-1, order, representative)
    return out.reshape(B, T, C, H, W)


# ---------------------------------------------------------------------- helpers
def build_runner_from_cfg(cfg) -> PGFMMRunner:
    transport = cfg.transport
    flow_map = cfg.get("flow_map", {})
    physics = cfg.get("physics", {})
    skill = cfg.get("skill", {})
    gate = cfg.get("gate", {})
    residual = cfg.get("residual", {})
    motion = cfg.get("motion", {})
    residual_transform = residual.get("transform", "pixel")
    img_channels = cfg.dataset.get("img_channels", 1)
    default_state_channels = {
        "pixel": img_channels,
        "fft_ri": 2 * img_channels,
        "fft_low_detail": 3 * img_channels,
    }.get(residual_transform, img_channels)
    state_channels = residual.get(
        "state_channels",
        default_state_channels,
    )
    runner_cfg = PGFMMConfig(
        T_in=cfg.dataset.T_in,
        T_out=cfg.dataset.T_out,
        img_channels=1,
        state_channels=state_channels,
        img_size=cfg.dataset.img_size,
        interval=transport.interval,
        prior_kind=transport.prior_kind,
        flow_map_noise_scale=flow_map.get("noise_scale", 1.0),
        flow_map_min_delta=flow_map.get("min_delta", 0.05),
        flow_map_direct_prob=flow_map.get("direct_prob", 0.25),
        flow_map_consistency_weight=flow_map.get("consistency_weight", 0.0),
        flow_map_consistency_start_step=flow_map.get("consistency_start_step", 0),
        flow_map_consistency_ramp_steps=flow_map.get("consistency_ramp_steps", 0),
        flow_map_consistency_detach_midpoint=bool(
            flow_map.get("consistency_detach_midpoint", False)
        ),
        flow_map_energy_score_weight=flow_map.get("energy_score_weight", 0.0),
        flow_map_energy_score_samples=flow_map.get("energy_score_samples", 2),
        flow_map_self_conditioning=bool(flow_map.get("self_conditioning", False)),
        flow_map_self_cond_prob=flow_map.get("self_cond_prob", 0.5),
        flow_map_lead_time_stochasticity=bool(flow_map.get("lead_time_stochasticity", False)),
        flow_map_noise_scale_min=flow_map.get("noise_scale_min", 0.3),
        flow_map_noise_scale_max=flow_map.get("noise_scale_max", 1.0),
        lambda_ssa=flow_map.get("lambda_ssa", 0.0),
        ssa_cutoff=int(flow_map.get("ssa_cutoff", 16)),
        lambda_psd=flow_map.get("lambda_psd", 0.0),
        cond_x1=transport.cond_x1,
        cond_extra_x1_external=bool(transport.get("cond_extra_x1_external", False)),
        cond_extra_T_external=transport.get("cond_extra_T_external", 0),
        motion_prior=bool(motion.get("enabled", False)),
        motion_prior_base_channels=motion.get("base_channels", 96),
        motion_prior_channel_mult=tuple(motion.get("channel_mult", [1, 2, 4, 4])),
        motion_prior_num_blocks=motion.get("num_blocks", 2),
        motion_prior_max_displacement=motion.get("max_displacement", 8.0),
        motion_prior_source_scale=motion.get("source_scale", 0.25),
        motion_prior_ckpt=motion.get("ckpt", ""),
        motion_prior_freeze=bool(motion.get("freeze", True)),
        use_residual_gate=bool(gate.get("use_residual_gate", False)),
        gate_init_bias=gate.get("init_bias", -1.10),
        lambda_gate_l1=gate.get("lambda_gate_l1", 0.0),
        base_channels=cfg.network.base_channels,
        channel_mult=tuple(cfg.network.channel_mult),
        attention_resolutions=tuple(cfg.network.attention_resolutions),
        num_res_blocks=cfg.network.num_res_blocks,
        dropout=cfg.network.dropout,
        nfe=cfg.sample.nfe,
        residual_scale=cfg.sample.get("residual_scale", 1.0),
        residual_normalize=bool(residual.get("normalize", False)),
        residual_transform=residual_transform,
        residual_low_freq=residual.get("low_freq", 20),
        residual_norm_min=residual.get("norm_min", 0.25),
        residual_norm_max=residual.get("norm_max", 2.0),
        residual_norm_disagreement=residual.get("norm_disagreement", 4.0),
        residual_posterior_shrinkage=bool(residual.get("posterior_shrinkage", False)),
        residual_prior_var=residual.get("prior_var", 0.0025),
        residual_target_posterior_shrinkage=bool(residual.get("target_posterior_shrinkage", False)),
        residual_posterior_strength=residual.get("posterior_strength", 1.0),
        residual_intensity_var_weight=residual.get("intensity_var_weight", 0.0),
        residual_intensity_var_power=residual.get("intensity_var_power", 2.0),
        use_learned_backbone_uncertainty=bool(
            residual.get("use_learned_backbone_uncertainty", False)
        ),
        lambda_mass=physics.get("lambda_mass", 0.0),
        lambda_smooth=physics.get("lambda_smooth", 0.0),
        lambda_diffusion=physics.get("lambda_diffusion", 0.0),
        diffusion_kappa=physics.get("diffusion_kappa", 0.01),
        lambda_final_l1=skill.get("lambda_final_l1", 0.0),
        lambda_false_positive=skill.get("lambda_false_positive", 0.0),
        lambda_soft_csi=skill.get("lambda_soft_csi", 0.0),
        soft_csi_sharpness=skill.get("soft_csi_sharpness", 8.0),
        skill_pixel_scale=skill.get("pixel_scale", 255.0),
        skill_thresholds=tuple(skill.get("thresholds", [74.0, 133.0, 160.0, 181.0])),
        lambda_preservation=skill.get("lambda_preservation", 0.0),
        preservation_tolerance=skill.get("preservation_tolerance", 0.04),
        lambda_calibration_mse=skill.get("lambda_calibration_mse", 0.0),
    )
    return PGFMMRunner(runner_cfg)


def load_state_dict(runner: PGFMMRunner, ckpt: dict, use_ema: bool):
    """Load model state into runner.  Handles ema-pytorch's state-dict layout."""
    if use_ema and "ema" in ckpt:
        # ema-pytorch saves keys prefixed with "ema_model.<orig_key>" (and a few
        # bookkeeping keys "initted", "step", "online_model.*"; we just need
        # the ema_model.* slice).
        ema_state = ckpt["ema"]
        new_state = {}
        for k, v in ema_state.items():
            if k.startswith("ema_model."):
                new_state[k[len("ema_model.") :]] = v
        if not new_state:
            raise RuntimeError("ckpt['ema'] has no 'ema_model.*' keys")
        missing, unexpected = runner.load_state_dict(new_state, strict=False)
        print(
            f"[ckpt] loaded EMA: {len(new_state)} tensors  "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        runner.load_state_dict(ckpt["model"], strict=False)
        print(f"[ckpt] loaded model  (step={ckpt.get('step', '?')})")


def setup_alphapre_evaluator():
    """Import AlphaPre's Evaluator with our shared FFT shim already in place."""
    # We don't use the full baselines._common.setup() because it chdirs into
    # the AlphaPre repo, which we don't want here.
    sys.path.insert(0, str(ROOT / "ext_repos" / "AlphaPre"))
    from utils.metrics import Evaluator  # noqa: E402

    return Evaluator


def summarize_evaluator(evaluator, thresholds: list[float | int]) -> dict[str, float]:
    """Recompute AlphaPre-style scalar metrics from Evaluator internals.

    ``Evaluator.done()`` prints the full table but only returns ``{"csi": ...}``.
    For CSV comparisons we mirror its formulas here and flatten the result into
    stable columns.
    """
    summary: dict[str, float] = {}
    avg_csi, avg_far, avg_pod, avg_hss = [], [], [], []
    avg_csi44, avg_csi16 = [], []

    for threshold in thresholds:
        metrics = evaluator.metrics[threshold]
        hits = np.nan_to_num(np.array(metrics["hits"]))
        misses = np.nan_to_num(np.array(metrics["misses"]))
        falsealarms = np.nan_to_num(np.array(metrics["falsealarms"]))
        correctnegs = np.nan_to_num(np.array(metrics["correctnegs"]))

        mean_hits = np.mean(hits, axis=0)
        mean_misses = np.mean(misses, axis=0)
        mean_false = np.mean(falsealarms, axis=0)
        mean_correct = np.mean(correctnegs, axis=0)

        csi = np.nan_to_num(mean_hits / (mean_hits + mean_misses + mean_false))
        far = np.nan_to_num(mean_false / (mean_hits + mean_false))
        pod = np.nan_to_num(mean_hits / (mean_hits + mean_misses))
        hss = np.nan_to_num(
            2
            * (mean_hits * mean_correct - mean_misses * mean_false)
            / (
                (mean_hits + mean_misses) * (mean_misses + mean_correct)
                + (mean_hits + mean_false) * (mean_false + mean_correct)
            )
        )

        hits44 = np.array(metrics["hits44"])
        misses44 = np.array(metrics["misses44"])
        false44 = np.array(metrics["falsealarms44"])
        hits16 = np.array(metrics["hits16"])
        misses16 = np.array(metrics["misses16"])
        false16 = np.array(metrics["falsealarms16"])
        csi44 = np.nan_to_num(
            np.mean(hits44) / (np.mean(hits44) + np.mean(misses44) + np.mean(false44))
        )
        csi16 = np.nan_to_num(
            np.mean(hits16) / (np.mean(hits16) + np.mean(misses16) + np.mean(false16))
        )

        thr = str(int(threshold) if float(threshold).is_integer() else threshold)
        summary[f"csi@{thr}"] = float(np.mean(csi))
        summary[f"far@{thr}"] = float(np.mean(far))
        summary[f"pod@{thr}"] = float(np.mean(pod))
        summary[f"hss@{thr}"] = float(np.mean(hss))
        summary[f"csi_pool4@{thr}"] = float(csi44)
        summary[f"csi_pool16@{thr}"] = float(csi16)

        avg_csi.append(np.mean(csi))
        avg_far.append(np.mean(far))
        avg_pod.append(np.mean(pod))
        avg_hss.append(np.mean(hss))
        avg_csi44.append(csi44)
        avg_csi16.append(csi16)

    summary["avg_csi"] = float(np.nan_to_num(np.mean(avg_csi)))
    summary["avg_far"] = float(np.nan_to_num(np.mean(avg_far)))
    summary["avg_pod"] = float(np.nan_to_num(np.mean(avg_pod)))
    summary["avg_hss"] = float(np.nan_to_num(np.mean(avg_hss)))
    summary["avg_csi_pool4"] = float(np.nan_to_num(np.mean(avg_csi44)))
    summary["avg_csi_pool16"] = float(np.nan_to_num(np.mean(avg_csi16)))

    for key, values in evaluator.losses.items():
        arr = np.array(values)
        summary[key] = float(np.nan_to_num(np.mean(arr))) if arr.size else float("nan")
    return summary


def append_metrics_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    if path.exists():
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            existing = next(reader, None)
        if existing:
            fieldnames = list(dict.fromkeys([*existing, *fieldnames]))

    rows = []
    if path.exists():
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    rows.append({k: row.get(k, "") for k in fieldnames})

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------- main
def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.override)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}  ckpt={args.ckpt}  cfg={args.config}")

    # ------ runner & ckpt ------
    runner = build_runner_from_cfg(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    load_state_dict(runner, ckpt, use_ema=args.use_ema)
    runner.eval()
    n_params = sum(p.numel() for p in runner.parameters())
    print(f"[model] params={n_params / 1e6:.2f} M")

    # ------ data ------
    name = cfg.dataset.name
    ds_kw = dataset_kwargs_from_cfg(cfg.dataset)
    ds = get_dataset(name, split=args.split, img_size=cfg.dataset.img_size, **ds_kw)
    use_paired = cfg.transport.prior_kind in ("external", "external_residual") or bool(
        cfg.transport.get("cond_extra_x1_external", False)
    )
    if use_paired:
        cache_dir = ROOT / cfg.dataset.cache_dir
        cache_h5 = cache_dir / f"{args.split}.h5"
        extra_cache_dirs = [ROOT / p for p in cfg.dataset.get("extra_cache_dirs", [])]
        if extra_cache_dirs:
            ds = MultiCachePairedDataset(
                ds,
                cache_h5,
                [d / f"{args.split}.h5" for d in extra_cache_dirs],
            )
        else:
            ds = PairedDataset(ds, cache_h5)
        print(f"[data] paired with cached preds from {cache_h5}  backbone={ds.cache.backbone}")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    print(
        f"[data] {name}/{args.split}: {len(ds):,} samples  "
        f"({len(dl):,} batches @ batch={args.batch_size})"
    )

    # ------ evaluator ------
    Evaluator = setup_alphapre_evaluator()
    evaluator = Evaluator(
        seq_len=cfg.dataset.T_out,
        value_scale=PIXEL_SCALES[name],
        thresholds=THRESHOLDS[name],
    )
    print(f"[eval ] thresholds={THRESHOLDS[name]}  pixel_scale={PIXEL_SCALES[name]}")

    # ------ autocast ------
    if args.mixed_precision == "bf16":
        amp_kwargs = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    elif args.mixed_precision == "fp16":
        amp_kwargs = dict(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")
    else:
        amp_kwargs = dict(device_type="cuda", enabled=False)

    def write_row(ev, method, n, dt):
        if args.csv_out is None:
            return
        metrics = summarize_evaluator(ev, THRESHOLDS[name])
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "run_name": args.run_name or args.ckpt.parents[1].name,
            "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_commit": git_value(["rev-parse", "--short", "HEAD"]),
            "dataset": name,
            "split": args.split,
            "n_samples": n,
            "nfe": args.nfe,
            "use_ema": bool(args.use_ema),
            "ot_ode": bool(args.ot_ode),
            "ckpt": str(args.ckpt),
            "config": str(args.config),
            "cache_dir": str(cfg.dataset.get("cache_dir", "")),
            "transport_type": "flow_map",
            "prior_kind": str(cfg.transport.get("prior_kind", "")),
            "elapsed_sec": round(dt, 3),
            "samples_per_sec": round(n / max(dt, 1e-9), 6),
            **metrics,
        }
        append_metrics_csv(args.csv_out, row)
        print(f"[csv ] appended {method} to {args.csv_out}")

    # ------ PMM blend sweep: one sampling pass, many alpha operating points ----
    if args.pmm_sweep:
        if args.ensemble <= 1:
            raise ValueError("--pmm_sweep requires --ensemble > 1")
        cal_gains = args.cal_gains if args.cal_gains else [1.0]
        cal_gammas = args.cal_gammas if args.cal_gammas else [1.0]
        # Operating points = pmm_weight x gain x gamma, all from ONE sampling pass.
        op_points = [(a, g, gm) for a in args.pmm_weights for g in cal_gains for gm in cal_gammas]
        evals = {
            pt: Evaluator(
                seq_len=cfg.dataset.T_out,
                value_scale=PIXEL_SCALES[name],
                thresholds=THRESHOLDS[name],
            )
            for pt in op_points
        }
        n_done = 0
        t0 = time.time()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dl, desc=f"PMM sweep (NFE={args.nfe})")):
                if args.max_batches is not None and i >= args.max_batches:
                    break
                if use_paired:
                    if len(batch) == 3:
                        frames, x1_ext, cond_extra_ext = batch
                        cond_extra_ext = cond_extra_ext.to(device, non_blocking=True)
                    else:
                        frames, x1_ext = batch
                        cond_extra_ext = None
                    frames = frames.to(device, non_blocking=True)
                    x1_ext = x1_ext.to(device, non_blocking=True)
                else:
                    frames = batch.to(device, non_blocking=True)
                    x1_ext = None
                    cond_extra_ext = None
                cond = frames[:, : cfg.dataset.T_in].contiguous()
                gt = frames[:, cfg.dataset.T_in :].contiguous()
                with torch.amp.autocast(**amp_kwargs):
                    members = []
                    for m in range(args.ensemble):
                        torch.manual_seed(args.ensemble_seed + m * 9973 + i)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(args.ensemble_seed + m * 9973 + i)
                        members.append(
                            runner.sample(
                                cond,
                                x1_external=x1_ext,
                                cond_extra_external=cond_extra_ext,
                                nfe=args.nfe,
                                ot_ode=args.ot_ode,
                                verbose=False,
                            )
                            .float()
                            .clamp_(0.0, 1.0)
                        )
                stacked = torch.stack(members, dim=0)
                mean = stacked.mean(dim=0)
                pmm = probability_matched_mean(stacked)
                for (a, g, gm), ev in evals.items():
                    blend = (a * pmm + (1.0 - a) * mean).float().clamp_(0.0, 1.0)
                    blend = calibrate(blend, g, gm).to(gt.device)
                    ev.evaluate(gt, blend)
                n_done += gt.shape[0]
        dt = time.time() - t0
        print(f"\n[time] {n_done} samples in {dt:.1f}s  ({n_done / dt:.2f} samp/s)")
        base = args.method or cfg.get("method", "pgfmm")
        for (a, g, gm), ev in evals.items():
            ev.done(is_main_process=True)
            tag = f"{base}-pmm{int(round(a * 100)):03d}"
            if g != 1.0 or gm != 1.0:
                tag += f"-g{g:g}-gm{gm:g}"
            write_row(ev, tag, n_done, dt)
        return

    # ------ sampling loop ------
    n_done = 0
    t0 = time.time()
    pbar = tqdm(dl, desc=f"PG-FMM sample (NFE={args.nfe})")
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            if args.max_batches is not None and i >= args.max_batches:
                break
            if use_paired:
                if len(batch) == 3:
                    frames, x1_ext, cond_extra_ext = batch
                    cond_extra_ext = cond_extra_ext.to(device, non_blocking=True)
                else:
                    frames, x1_ext = batch
                    cond_extra_ext = None
                frames = frames.to(device, non_blocking=True)
                x1_ext = x1_ext.to(device, non_blocking=True)
            else:
                frames = batch.to(device, non_blocking=True)
                x1_ext = None
                cond_extra_ext = None
            cond = frames[:, : cfg.dataset.T_in].contiguous()
            gt = frames[:, cfg.dataset.T_in :].contiguous()

            with torch.amp.autocast(**amp_kwargs):
                if args.prior_only:
                    if runner.motion_prior is None:
                        raise ValueError("--prior_only requires a config with motion.enabled=true")
                    pred = runner._resolve_motion_prior(cond, None)
                    if pred is None:
                        raise RuntimeError("motion prior returned None")
                elif args.ensemble <= 1:
                    pred = runner.sample(
                        cond,
                        x1_external=x1_ext,
                        cond_extra_external=cond_extra_ext,
                        nfe=args.nfe,
                        ot_ode=args.ot_ode,
                        verbose=False,
                    )
                else:
                    accum = None
                    members = [] if args.ensemble_agg == "pmm" else None
                    for m in range(args.ensemble):
                        torch.manual_seed(args.ensemble_seed + m * 9973 + i)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(args.ensemble_seed + m * 9973 + i)
                        sample_m = (
                            runner.sample(
                                cond,
                                x1_external=x1_ext,
                                cond_extra_external=cond_extra_ext,
                                nfe=args.nfe,
                                ot_ode=args.ot_ode,
                                verbose=False,
                            )
                            .float()
                            .clamp_(0.0, 1.0)
                        )
                        if members is not None:
                            members.append(sample_m)
                        else:
                            accum = sample_m if accum is None else accum + sample_m
                    if members is not None:
                        stacked = torch.stack(members, dim=0)
                        pmm = probability_matched_mean(stacked)
                        w = float(args.pmm_weight)
                        pred = pmm if w >= 1.0 else w * pmm + (1.0 - w) * stacked.mean(dim=0)
                    else:
                        pred = accum / float(args.ensemble)
            pred = pred.float().clamp_(0.0, 1.0).to(gt.device)
            if args.pred_gain != 1.0 or args.pred_gamma != 1.0:
                pred = calibrate(pred, args.pred_gain, args.pred_gamma)

            # AlphaPre evaluator wants numpy (B, T, C, H, W) in [0, 1]
            evaluator.evaluate(gt, pred)
            n_done += pred.shape[0]

    dt = time.time() - t0
    print(f"\n[time] {n_done} samples in {dt:.1f}s  ({n_done / dt:.2f} samp/s)")

    # ------ summary (this prints AlphaPre-style table to stdout) ------
    res = evaluator.done(is_main_process=True)
    print(f"\n[done] {res}")

    if args.per_lead_out is not None:
        ths = THRESHOLDS[name]
        out = {
            "thresholds": np.array([float(t) for t in ths]),
            "seq_len": np.array(int(cfg.dataset.T_out)),
        }
        for threshold in ths:
            m = evaluator.metrics[threshold]
            h = np.nan_to_num(np.array(m["hits"]))
            mi = np.nan_to_num(np.array(m["misses"]))
            fa = np.nan_to_num(np.array(m["falsealarms"]))
            cn = np.nan_to_num(np.array(m["correctnegs"]))
            mh, mm, mf, mc = h.mean(0), mi.mean(0), fa.mean(0), cn.mean(0)
            csi = np.nan_to_num(mh / (mh + mm + mf))
            hss = np.nan_to_num(
                2 * (mh * mc - mm * mf) / ((mh + mm) * (mm + mc) + (mh + mf) * (mf + mc))
            )
            thr = str(int(threshold) if float(threshold).is_integer() else threshold)
            out[f"csi_per_lead@{thr}"] = np.asarray(csi)
            out[f"hss_per_lead@{thr}"] = np.asarray(hss)
        args.per_lead_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.per_lead_out, **out)
        print(f"[per-lead] saved per-lead-time CSI/HSS to {args.per_lead_out}")

    if args.csv_out is not None:
        metrics = summarize_evaluator(evaluator, THRESHOLDS[name])
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": args.method or cfg.get("method", "pgfmm"),
            "run_name": args.run_name or args.ckpt.parents[1].name,
            "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_commit": git_value(["rev-parse", "--short", "HEAD"]),
            "dataset": name,
            "split": args.split,
            "n_samples": n_done,
            "nfe": args.nfe,
            "use_ema": bool(args.use_ema),
            "ot_ode": bool(args.ot_ode),
            "ckpt": str(args.ckpt),
            "config": str(args.config),
            "cache_dir": str(cfg.dataset.get("cache_dir", "")),
            "transport_type": "flow_map",
            "prior_kind": str(cfg.transport.get("prior_kind", "")),
            "elapsed_sec": round(dt, 3),
            "samples_per_sec": round(n_done / max(dt, 1e-9), 6),
            **metrics,
        }
        append_metrics_csv(args.csv_out, row)
        print(f"[csv ] appended metrics to {args.csv_out}")


if __name__ == "__main__":
    main()
