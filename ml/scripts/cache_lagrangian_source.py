#!/usr/bin/env python3
"""Cache learned Lagrangian source forecasts for the diffusion bridge.

Output layout matches ``scripts/cache_predictions.py`` so the existing
``PairedDataset`` can feed it to ``prior_kind: external`` unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pgfmm.data import PIXEL_SCALES, dataset_kwargs_from_cfg, get_dataset  # noqa: E402
from pgfmm.source import LagrangianSourceConfig, LagrangianSourceNet  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset", default="sevir", choices=["sevir"])
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--T_in", type=int, default=5)
    p.add_argument("--T_out", type=int, default=20)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--out_dir", type=Path, default=Path("data/sevir_pred/lagrangian_source_v1"))
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def build_model(cfg) -> LagrangianSourceNet:
    source_cfg = LagrangianSourceConfig(
        T_in=cfg.dataset.T_in,
        T_out=cfg.dataset.T_out,
        img_channels=1,
        img_size=cfg.dataset.img_size,
        base_channels=cfg.model.base_channels,
        channel_mult=tuple(cfg.model.channel_mult),
        num_blocks=cfg.model.get("num_blocks", 2),
        max_displacement=cfg.model.max_displacement,
        source_scale=cfg.model.source_scale,
    )
    return LagrangianSourceNet(source_cfg)


def load_state(model: LagrangianSourceNet, ckpt_path: Path, use_ema: bool) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if use_ema and "ema" in ckpt:
        ema_state = ckpt["ema"]
        sd = {k[len("ema_model.") :]: v for k, v in ema_state.items() if k.startswith("ema_model.")}
        if not sd:
            raise RuntimeError("EMA state has no ema_model.* weights")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[ckpt] loaded EMA from {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[ckpt] loaded model from {ckpt_path} step={ckpt.get('step', '?')}")


@torch.no_grad()
def cache_split(model: LagrangianSourceNet, split: str, args, out_path: Path) -> None:
    cfg = OmegaConf.load(args.config)
    ds_kw = dataset_kwargs_from_cfg(cfg.dataset)
    img_size = int(cfg.dataset.get("img_size", args.img_size))
    ds = get_dataset(args.dataset, split=split, img_size=img_size, **ds_kw)
    if args.max_samples is not None:
        ds = Subset(ds, range(min(args.max_samples, len(ds))))
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if not args.overwrite:
            print(f"[skip] {out_path} exists; pass --overwrite to recompute")
            return
        out_path.unlink()

    device = next(model.parameters()).device
    pixel_scale = PIXEL_SCALES[args.dataset]
    n = len(ds)
    T_in = int(cfg.dataset.T_in)
    T_out = int(cfg.dataset.T_out)
    H = W = img_size
    if args.mixed_precision == "bf16":
        amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    elif args.mixed_precision == "fp16":
        amp = dict(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")
    else:
        amp = dict(device_type="cuda", enabled=False)

    t0 = time.time()
    with h5py.File(out_path, "w") as f:
        preds_ds = f.create_dataset(
            "preds",
            shape=(n, T_out, 1, H, W),
            dtype="uint16",
            chunks=(1, args.T_out, 1, H, W),
            compression="lzf",
        )
        sample_idx = f.create_dataset("sample_idx", shape=(n,), dtype="int32")
        f.attrs["pixel_scale"] = float(pixel_scale)
        f.attrs["backbone"] = "learned_lagrangian_source"
        f.attrs["src_ckpt"] = str(args.ckpt)
        f.attrs["dataset"] = args.dataset
        f.attrs["split"] = split
        f.attrs["T_in"] = T_in
        f.attrs["T_out"] = T_out
        f.attrs["img_size"] = img_size

        idx = 0
        for frames in tqdm(dl, desc=f"lagrangian source {split}"):
            frames = frames.to(device, non_blocking=True)
            cond = frames[:, :T_in].contiguous()
            with torch.amp.autocast(**amp):
                pred, _ = model(cond)
            pred = pred.float().clamp_(0.0, 1.0).cpu().numpy()
            pred_u16 = np.round(pred * pixel_scale).astype(np.uint16)
            B = pred_u16.shape[0]
            preds_ds[idx : idx + B] = pred_u16
            sample_idx[idx : idx + B] = np.arange(idx, idx + B, dtype=np.int32)
            idx += B
    dt = time.time() - t0
    print(
        f"[done] {split}: wrote {idx:,} forecasts to {out_path} ({dt:.1f}s, {idx / dt:.1f} samp/s)"
    )


def main():
    args = parse_args()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    cfg = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    load_state(model, args.ckpt, args.use_ema)
    print(f"[env] device={device} out={args.out_dir}")

    for split in args.splits:
        cache_split(model, split, args, args.out_dir / f"{split}.h5")


if __name__ == "__main__":
    main()
