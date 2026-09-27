"""Train the PG-FMM model (two frozen-then-generative stages).

Single-GPU friendly via HuggingFace `accelerate` (auto handles bf16, DDP later).

Examples
--------
# Stage 1 — Lagrangian advection prior (train first, then frozen)
python ml/train.py --config ml/configs/sevir/lagrangian_prior.yaml --note lagrangian_prior

# Stage 2 — Flow-Map Matching head (conditioned on the frozen prior)
python ml/train.py --config ml/configs/sevir/pgfmm.yaml --note pgfmm
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # .../PG-FMM/ml
ROOT = HERE.parent  # .../PG-FMM
sys.path.insert(0, str(HERE))  # make `import pgfmm` importable

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from ema_pytorch import EMA  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pgfmm.data import dataset_kwargs_from_cfg, get_dataset  # noqa: E402
from pgfmm.data.paired import MultiCachePairedDataset, PairedDataset  # noqa: E402
from pgfmm.model import PGFMMConfig, PGFMMRunner  # noqa: E402


# --------------------------------------------------------------------- helpers
def cycle(dl):
    while True:
        yield from dl


def cosine_lr(step, *, warmup, total, lr_max, lr_min=0.0):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(1.0, progress)))


def augment_flips(frames, p_h: float, p_v: float):
    """Per-sample random horizontal/vertical flips of a full (B,T,C,H,W) clip.

    The same flip is applied to every frame in a sample so the temporal
    dynamics stay consistent; advection/optical-flow is equivariant to flips,
    so this is physically valid radar augmentation (no preferred L/R or U/D).
    Cuts overfitting on small train sets (e.g. CIKM, 1k clips).
    """
    B = frames.shape[0]
    if p_h > 0.0:
        mask = torch.rand(B, device=frames.device) < p_h
        if mask.any():
            frames[mask] = torch.flip(frames[mask], dims=[-1])
    if p_v > 0.0:
        mask = torch.rand(B, device=frames.device) < p_v
        if mask.any():
            frames[mask] = torch.flip(frames[mask], dims=[-2])
    return frames


# --------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--note", default="run", help="experiment subdir suffix")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint (e.g. .../last.pt) to resume model, EMA, optimizer and step counter from",
    )
    p.add_argument("--override", nargs="*", default=[], help="OmegaConf-style key=value overrides")
    # ---- wandb integration ---------------------------------------------
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging via accelerate trackers",
    )
    p.add_argument(
        "--wandb-project",
        default="pgfmm",
        help="W&B project name (only used if --wandb)",
    )
    p.add_argument(
        "--wandb-entity", default=None, help="W&B team/user (default: uses your login default)"
    )
    p.add_argument(
        "--wandb-mode",
        default=None,
        choices=[None, "online", "offline", "disabled"],
        help="W&B run mode (overrides $WANDB_MODE)",
    )
    p.add_argument(
        "--wandb-tags", nargs="*", default=None, help="Optional W&B tags (space-separated list)"
    )
    return p.parse_args()


def build_runner(cfg) -> PGFMMRunner:
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



# ------------------------------------------------------------- Stage 1: prior
def train_prior(cfg, args) -> None:
    """Train the Stage-1 Lagrangian advection prior (configs/*/lagrangian_prior.yaml).

    Dispatched from ``main()`` for configs without a ``transport`` section.
    Saves ``runs/<dataset>/<note>/checkpoints/{best,last}.pt`` in the format
    expected by ``PGFMMRunner`` (``motion.ckpt``) and
    ``ml/scripts/cache_lagrangian_source.py``.
    """
    from pgfmm.source.lagrangian import (
        LagrangianSourceConfig,
        LagrangianSourceNet,
        lagrangian_source_loss,
    )

    exp_dir = Path(os.environ.get("PGFMM_OUT", ROOT / "runs")) / cfg.dataset.name / args.note
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, exp_dir / "config.yaml")

    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
    is_main = accelerator.is_main_process
    if args.wandb and is_main:
        accelerator.print("[note] wandb logging is not wired for Stage-1; ignoring --wandb")

    ds_kw = dataset_kwargs_from_cfg(cfg.dataset)
    train_ds = get_dataset(cfg.dataset.name, split="train", img_size=cfg.dataset.img_size, **ds_kw)
    val_ds = get_dataset(cfg.dataset.name, split="val", img_size=cfg.dataset.img_size, **ds_kw)
    persistent = cfg.train.num_workers > 0
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                          num_workers=cfg.train.num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=persistent)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers, pin_memory=True,
                        drop_last=False, persistent_workers=persistent)

    model = LagrangianSourceNet(LagrangianSourceConfig(
        T_in=cfg.dataset.T_in,
        T_out=cfg.dataset.T_out,
        img_size=cfg.dataset.img_size,
        base_channels=cfg.model.base_channels,
        channel_mult=tuple(cfg.model.channel_mult),
        num_blocks=cfg.model.num_blocks,
        max_displacement=cfg.model.max_displacement,
        source_scale=cfg.model.source_scale,
    ))
    if is_main:
        accelerator.print(f"[model] Lagrangian prior params="
                          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optim = AdamW(model.parameters(), lr=cfg.train.lr,
                  weight_decay=cfg.train.weight_decay, betas=(0.9, 0.999))
    model, optim, train_dl, val_dl = accelerator.prepare(model, optim, train_dl, val_dl)
    ema = EMA(model, beta=cfg.train.ema_rate, update_every=10).to(accelerator.device)

    def compute_loss(frames):
        cond = frames[:, : cfg.dataset.T_in]
        gt = frames[:, cfg.dataset.T_in:]
        pred, aux = model(cond)
        total, parts = lagrangian_source_loss(
            pred, gt, aux,
            lambda_l1=cfg.loss.lambda_l1,
            lambda_mse=cfg.loss.lambda_mse,
            lambda_soft_csi=cfg.loss.lambda_soft_csi,
            lambda_flow_smooth=cfg.loss.lambda_flow_smooth,
            lambda_source_sparse=cfg.loss.lambda_source_sparse,
            thresholds=tuple(cfg.loss.thresholds),
            pixel_scale=cfg.loss.pixel_scale,
            soft_csi_sharpness=cfg.loss.soft_csi_sharpness,
        )
        return total

    def save_ckpt(path, step):
        torch.save({"step": step,
                    "model": accelerator.get_state_dict(model),
                    "ema": ema.state_dict(),
                    "optim": optim.state_dict()}, path)
        accelerator.print(f"[ckpt] saved {path}")

    @torch.no_grad()
    def run_val(max_batches):
        model.eval()
        losses = []
        for i, frames in enumerate(val_dl):
            if i >= max_batches:
                break
            losses.append(compute_loss(frames.to(accelerator.device)).item())
        model.train()
        return sum(losses) / max(1, len(losses))

    train_iter = cycle(train_dl)
    best_val = float("inf")
    t0 = time.time()
    pbar = tqdm(range(cfg.train.total_steps), disable=not is_main, dynamic_ncols=True)
    for step in pbar:
        lr = cosine_lr(step, warmup=cfg.train.warmup_steps,
                       total=cfg.train.total_steps, lr_max=cfg.train.lr)
        for g in optim.param_groups:
            g["lr"] = lr
        frames = next(train_iter).to(accelerator.device, non_blocking=True)
        loss = compute_loss(frames)
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optim.step()
        optim.zero_grad()
        ema.update()

        if is_main and step % cfg.train.log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.1e}",
                             sps=f"{(step + 1) / (time.time() - t0):.2f}/s")
        if is_main and step > 0 and step % cfg.train.val_every == 0:
            val_loss = run_val(cfg.train.val_max_batches)
            accelerator.print(f"[val] step={step} loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(ckpt_dir / "best.pt", step)
        if is_main and step > 0 and step % cfg.train.ckpt_every == 0:
            save_ckpt(ckpt_dir / "last.pt", step)

    if is_main:
        save_ckpt(ckpt_dir / "last.pt", cfg.train.total_steps)
        if not (ckpt_dir / "best.pt").exists():
            save_ckpt(ckpt_dir / "best.pt", cfg.train.total_steps)
        accelerator.print(f"[done] Stage-1 prior saved under {ckpt_dir}")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.override)))
    set_seed(args.seed)

    if "transport" not in cfg:
        # Stage-1 config (Lagrangian advection prior) has no flow-map sections.
        return train_prior(cfg, args)

    exp_name = f"pgfmm_{cfg.dataset.name}_{args.note}"
    exp_dir = Path(os.environ.get("PGFMM_OUT", ROOT / "runs")) / cfg.dataset.name / exp_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, exp_dir / "config.yaml")

    # ---- wandb env setup (must happen BEFORE Accelerator.init_trackers) ---
    if args.wandb_mode is not None:
        os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.setdefault("WANDB_DIR", str(exp_dir))

    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        log_with="wandb" if args.wandb else None,
    )
    is_main = accelerator.is_main_process

    if args.wandb:
        wandb_init_kwargs: dict = {}
        if args.wandb_entity:
            wandb_init_kwargs["entity"] = args.wandb_entity
        if args.wandb_tags:
            wandb_init_kwargs["tags"] = list(args.wandb_tags)
        wandb_init_kwargs["name"] = exp_name
        wandb_init_kwargs["dir"] = str(exp_dir)
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_init_kwargs},
        )
        if is_main:
            accelerator.print(
                f"[wandb] project={args.wandb_project!r}  run={exp_name!r}  "
                f"mode={os.environ.get('WANDB_MODE', 'online')}"
            )

    # ---- data ---------------------------------------------------------
    ds_kw = dataset_kwargs_from_cfg(cfg.dataset)
    train_ds = get_dataset(cfg.dataset.name, split="train", img_size=cfg.dataset.img_size, **ds_kw)
    val_ds = get_dataset(cfg.dataset.name, split="val", img_size=cfg.dataset.img_size, **ds_kw)
    use_paired = cfg.transport.prior_kind in ("external", "external_residual") or bool(
        cfg.transport.get("cond_extra_x1_external", False)
    )
    if use_paired:
        # Paired mode: each item is (frames, precomputed_pred)
        cache_dir = ROOT / cfg.dataset.cache_dir  # e.g. data/sevir_pred/alphapre
        extra_cache_dirs = [ROOT / p for p in cfg.dataset.get("extra_cache_dirs", [])]
        if extra_cache_dirs:
            train_ds = MultiCachePairedDataset(
                train_ds,
                cache_dir / "train.h5",
                [d / "train.h5" for d in extra_cache_dirs],
            )
            val_ds = MultiCachePairedDataset(
                val_ds,
                cache_dir / "val.h5",
                [d / "val.h5" for d in extra_cache_dirs],
            )
        else:
            train_ds = PairedDataset(train_ds, cache_dir / "train.h5")
            val_ds = PairedDataset(val_ds, cache_dir / "val.h5")
        if accelerator.is_main_process:
            accelerator.print(
                f"[data] paired with cached preds from {cache_dir}/  "
                f"backbone={train_ds.cache.backbone}"
            )
    persistent = cfg.train.num_workers > 0
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent,
    )
    if is_main:
        accelerator.print(f"[data] train batches={len(train_dl):,}  val batches={len(val_dl):,}")

    # ---- model + optim ------------------------------------------------
    runner = build_runner(cfg)
    # Keep the training section available to validation after the runner
    # has been converted to a plain PGFMMConfig.  This avoids adding
    # non-model bookkeeping fields to the dataclass.
    runner._train_cfg = cfg.train
    n_params = sum(p.numel() for p in runner.parameters())
    if is_main:
        accelerator.print(f"[model] params = {n_params / 1e6:.2f} M")

    optim = AdamW(
        runner.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.999),
    )
    runner, optim, train_dl, val_dl = accelerator.prepare(runner, optim, train_dl, val_dl)

    ema = EMA(runner, beta=cfg.train.ema_rate, update_every=10).to(accelerator.device)

    # ---- optional resume ---------------------------------------------
    # Restores model, EMA, optimizer and the step counter so an interrupted
    # run continues exactly where it left off.  The cosine LR is a pure
    # function of the step, so no scheduler state is needed.
    start_step = 0
    best_val = float("inf")
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(runner).load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        start_step = int(ckpt.get("step", 0)) + 1
        if is_main:
            accelerator.print(f"[resume] loaded {args.resume} -> continuing from step {start_step}")
        # Seed best_val with the resumed model so best.pt is only overwritten
        # by a genuine improvement (the pre-resume best.pt is preserved until
        # then).
        if is_main:
            best_val = run_validation(
                runner,
                val_dl,
                accelerator,
                max_batches=cfg.train.val_max_batches,
                use_paired=use_paired,
            )
            accelerator.print(f"[resume] seeded best_val={best_val:.4f}")

    # ---- training loop ------------------------------------------------
    train_iter = cycle(train_dl)
    pbar = tqdm(
        range(start_step, cfg.train.total_steps),
        initial=start_step,
        total=cfg.train.total_steps,
        disable=not is_main,
        dynamic_ncols=True,
    )
    t0 = time.time()
    # Early stopping: number of consecutive val checks without improvement
    # before we stop training.  0 (default) disables early stopping.
    early_stop_patience = int(cfg.train.get("early_stop_patience", 0))
    early_stop_min_delta = float(cfg.train.get("early_stop_min_delta", 0.0))
    aug_cfg = cfg.train.get("augment", {})
    aug_p_h = float(aug_cfg.get("flip_h", 0.0)) if aug_cfg else 0.0
    aug_p_v = float(aug_cfg.get("flip_v", 0.0)) if aug_cfg else 0.0
    if is_main and (aug_p_h > 0.0 or aug_p_v > 0.0):
        accelerator.print(f"[aug ] flip_h={aug_p_h}  flip_v={aug_p_v}")
    no_improve_count = 0
    early_stop_triggered = False
    last_step = -1

    for step in pbar:
        last_step = step
        runner.train()
        # cosine lr w/ warmup
        lr_total_steps = int(cfg.train.get("lr_total_steps", cfg.train.total_steps))
        lr = cosine_lr(
            step, warmup=cfg.train.warmup_steps, total=lr_total_steps, lr_max=cfg.train.lr
        )
        for g in optim.param_groups:
            g["lr"] = lr

        batch = next(train_iter)
        if use_paired:
            if len(batch) == 3:
                frames, x1_ext, cond_extra_ext = batch
                cond_extra_ext = cond_extra_ext.to(accelerator.device, non_blocking=True)
            else:
                frames, x1_ext = batch
                cond_extra_ext = None
            frames = frames.to(accelerator.device, non_blocking=True)
            x1_ext = x1_ext.to(accelerator.device, non_blocking=True)
        else:
            frames, x1_ext, cond_extra_ext = (
                batch.to(accelerator.device, non_blocking=True),
                None,
                None,
            )
            if aug_p_h > 0.0 or aug_p_v > 0.0:
                frames = augment_flips(frames, aug_p_h, aug_p_v)
        out = accelerator.unwrap_model(runner).train_step(
            frames,
            x1_external=x1_ext,
            cond_extra_external=cond_extra_ext,
            global_step=step,
        )
        loss = out["total_loss"]

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(runner.parameters(), cfg.train.grad_clip)
        optim.step()
        optim.zero_grad()
        ema.update()

        if is_main and step % cfg.train.log_every == 0:
            sps = (step - start_step + 1) / (time.time() - t0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.1e}", sps=f"{sps:.2f}/s")
            if args.wandb:
                log_items = {
                    "train/total_loss": loss.item(),
                    "train/lr": lr,
                    "train/steps_per_sec": sps,
                }
                for k, v in out.items():
                    if k == "total_loss":
                        continue
                    if torch.is_tensor(v) and v.numel() == 1:
                        log_items[f"train/{k}"] = v.item()
                    elif isinstance(v, (float, int)):
                        log_items[f"train/{k}"] = float(v)
                accelerator.log(log_items, step=step)

        if is_main and step > 0 and step % cfg.train.val_every == 0:
            val_loss = run_validation(
                runner,
                val_dl,
                accelerator,
                max_batches=cfg.train.val_max_batches,
                use_paired=use_paired,
            )
            accelerator.print(f"[val ] step={step}  loss={val_loss:.4f}")
            if args.wandb:
                accelerator.log(
                    {"val/total_loss": val_loss, "val/best": min(best_val, val_loss)}, step=step
                )
            if val_loss < best_val - early_stop_min_delta:
                best_val = val_loss
                no_improve_count = 0
                _save_ckpt(ckpt_dir / "best.pt", runner, ema, optim, step, accelerator)
            else:
                no_improve_count += 1
                if early_stop_patience > 0:
                    accelerator.print(
                        f"[val ] no improvement: {no_improve_count}/{early_stop_patience}"
                        f"  (best={best_val:.4f}, current={val_loss:.4f})"
                    )
                    if no_improve_count >= early_stop_patience:
                        accelerator.print(
                            f"[stop] early stopping at step {step}  "
                            f"(no val improvement for {early_stop_patience} consecutive checks)"
                        )
                        early_stop_triggered = True

        if is_main and step > 0 and step % cfg.train.ckpt_every == 0:
            _save_ckpt(ckpt_dir / "last.pt", runner, ema, optim, step, accelerator)
            if bool(cfg.train.get("save_milestones", False)):
                ms_dir = ckpt_dir / "milestones"
                ms_dir.mkdir(parents=True, exist_ok=True)
                _save_ckpt_light(ms_dir / f"step_{step}.pt", runner, ema, step, accelerator)

        if early_stop_triggered:
            break

    if is_main:
        _save_ckpt(ckpt_dir / "last.pt", runner, ema, optim, last_step, accelerator)
        accelerator.print(f"[done] saved final ckpt to {ckpt_dir}/last.pt")
    if args.wandb:
        accelerator.end_training()


def _save_ckpt(path, runner, ema, optim, step, accelerator):
    state = {
        "step": step,
        "model": accelerator.get_state_dict(runner),
        "ema": ema.state_dict(),
        "optim": optim.state_dict(),
    }
    torch.save(state, path)
    accelerator.print(f"[ckpt] saved {path}")


def _save_ckpt_light(path, runner, ema, step, accelerator):
    """Milestone checkpoint without optimizer state (model + ema only).

    Used for post-hoc val-CSI based model selection (AlphaPre protocol),
    keeps each file small enough to retain ~20 milestones cheaply.
    """
    state = {
        "step": step,
        "model": accelerator.get_state_dict(runner),
        "ema": ema.state_dict(),
    }
    torch.save(state, path)
    accelerator.print(f"[ckpt-ms] saved {path}")


@torch.no_grad()
def run_validation(runner, val_dl, accelerator, max_batches=20, use_paired=False):
    """Deterministic validation loss for early-stopping.

    The stochastic Flow Map ``train_step`` MSE is biased toward the
    trivial "predict the conditional mean" solution and is extremely
    noisy across val checks because every batch draws a fresh Gaussian
    noise and a fresh random (t, r) pair.  Empirically this caused
    v15 / v16 to lock ``best.pt`` to a step (7,500 / 74,400) where the
    model had not even started learning the residual structure.

    We fix this by:
      * fixing the RNG seed for the val pass (reproducible across
        checks),
      * forcing every val example to be a direct-endpoint sample
        (r=0) so the loss measures pure residual regression rather
        than interpolant fitting, and
      * reporting a configurable direct-endpoint metric.  The default is
        flow_map_mse for backward compatibility, but residual nowcasting
        runs can select ``train.val_metric=calibration_mse`` so
        checkpoints are chosen by final forecast risk rather than by the
        flow-map-space residual basis.
    """
    cfg = accelerator.unwrap_model(runner).cfg
    train_cfg = getattr(accelerator.unwrap_model(runner), "_train_cfg", None)
    is_flow_map = True
    val_metric = "flow_map_mse"
    if train_cfg is not None:
        val_metric = str(train_cfg.get("val_metric", val_metric))

    if is_flow_map:
        orig_direct_prob = float(cfg.flow_map_direct_prob)
        orig_cc_weight = float(cfg.flow_map_consistency_weight)
        orig_ssa = float(getattr(cfg, "lambda_ssa", 0.0))
        orig_psd = float(getattr(cfg, "lambda_psd", 0.0))
        cfg.flow_map_direct_prob = 1.0
        cfg.flow_map_consistency_weight = 0.0
        if hasattr(cfg, "lambda_ssa"):
            cfg.lambda_ssa = 0.0
        if hasattr(cfg, "lambda_psd"):
            cfg.lambda_psd = 0.0

    runner.eval()
    losses = []
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    rng_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(20260528)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260528)

    try:
        for i, batch in enumerate(val_dl):
            if i >= max_batches:
                break
            if use_paired:
                if len(batch) == 3:
                    frames, x1_ext, cond_extra_ext = batch
                    cond_extra_ext = cond_extra_ext.to(accelerator.device, non_blocking=True)
                else:
                    frames, x1_ext = batch
                    cond_extra_ext = None
                frames = frames.to(accelerator.device, non_blocking=True)
                x1_ext = x1_ext.to(accelerator.device, non_blocking=True)
            else:
                frames = batch.to(accelerator.device, non_blocking=True)
                x1_ext = None
                cond_extra_ext = None
            out = accelerator.unwrap_model(runner).train_step(
                frames,
                x1_external=x1_ext,
                cond_extra_external=cond_extra_ext,
            )
            if is_flow_map and val_metric == "calibration_mse":
                losses.append(out["skill_calibration_mse"].item())
            elif is_flow_map and val_metric == "total_loss":
                losses.append(out["total_loss"].item())
            elif is_flow_map and "flow_map_mse" in out:
                losses.append(out["flow_map_mse"].item())
            else:
                losses.append(out["total_loss"].item())
    finally:
        if is_flow_map:
            cfg.flow_map_direct_prob = orig_direct_prob
            cfg.flow_map_consistency_weight = orig_cc_weight
            if hasattr(cfg, "lambda_ssa"):
                cfg.lambda_ssa = orig_ssa
            if hasattr(cfg, "lambda_psd"):
                cfg.lambda_psd = orig_psd
        torch.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    return sum(losses) / max(1, len(losses))


if __name__ == "__main__":
    main()
