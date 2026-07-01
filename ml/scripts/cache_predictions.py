#!/usr/bin/env python3
"""Pre-compute deterministic-backbone predictions and cache them to h5.

Used to feed AlphaPre / DiffCast / SimVP / etc. predictions as the bridge
endpoint ``x_1`` for our SB-residual training (Idea C / DiffCast-style).

Cached layout (one file per split):

    data/sevir_pred/<backbone>/<split>.h5
        preds   : (N, T_out, C, H, W) uint16   = round(pred_float * pixel_scale)
        meta    : { 'pixel_scale': PIXEL_SCALES[name],
                    'backbone'   : 'alphapre',
                    'src_ckpt'   : '<path>',
                    'sample_idx' : (N,) int32   = the dataset.__getitem__ index
                                                    that produced each pred }

We index by ``sample_idx`` so that downstream code can pair predictions with
``Dataset[idx]`` exactly, even after dataset reshuffles or stride changes.

Usage
-----
# AlphaPre on SEVIR (train + val + test, ~30 min on Blackwell)
uv run python scripts/cache_predictions.py \
    --backbone alphapre \
    --ckpt    ext_repos/AlphaPre/resources/AlphaPre_sevir128.pt \
    --dataset sevir --img_size 128 --T_in 5 --T_out 20 \
    --splits  train val test \
    --batch_size 16 --num_workers 8 --mixed_precision bf16 \
    --out_dir data/sevir_pred/alphapre_official
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Make sure transient caches land in writable spots on shared boxes.
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pgfmm.data import PIXEL_SCALES, get_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=["alphapre", "diffcast", "simvp"])
    p.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="Backbone checkpoint .pt (the published one or our re-trained)",
    )
    p.add_argument("--dataset", required=True, choices=["sevir", "meteo", "shanghai", "cikm"])
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--T_in", type=int, default=5)
    p.add_argument("--T_out", type=int, default=20)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--out_dir", required=True, type=Path)
    return p.parse_args()


# ----------------------------------------------------------------------
# Backbone factories
# ----------------------------------------------------------------------
def build_alphapre(
    ckpt_path: Path, img_size: int, T_in: int, T_out: int, img_channels: int = 1
) -> torch.nn.Module:
    """Build the official AlphaPre model and load its checkpoint."""
    # Resolve the ckpt path BEFORE setup(), which chdirs into ext_repos/AlphaPre.
    ckpt_abs = Path(ckpt_path).expanduser().resolve()
    if not ckpt_abs.exists():
        raise FileNotFoundError(
            f"AlphaPre checkpoint not found: {ckpt_abs}\n"
            f"  Hint: download from "
            f"https://drive.google.com/file/d/1hzT2-biQhWuKTER8w1yoQx5Zh0nMYl80/view\n"
            f"  uv run gdown 1hzT2-biQhWuKTER8w1yoQx5Zh0nMYl80 -O {ckpt_abs}"
        )

    # Reuse the baselines wrapper to inject sys.path & FFT shim, but DON'T
    # patch DATAPATH (we don't use AlphaPre's loader here).
    from experiments.baselines._common import setup

    setup("AlphaPre")

    # After setup() ext_repos/AlphaPre is on sys.path[0]
    from models.alphapre import get_model  # noqa: E402

    model = get_model(
        input_shape=(img_size, img_size),
        T_in=T_in,
        T_out=T_out,
        img_channels=img_channels,
        dim=64,
        n_layers=3,
        spec_num=20,
        pha_weight=0.01,
        anet_weight=0.1,
        amp_weight=0.01,
        aweight_stop_steps=5000,
    )

    ckpt = torch.load(ckpt_abs, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    print(
        f"[backbone] loaded AlphaPre from {ckpt_abs}  "
        f"(step={ckpt.get('step', '?')}, "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M)"
    )
    return model


def build_backbone(name: str, **kw) -> torch.nn.Module:
    if name == "alphapre":
        return build_alphapre(**kw)
    raise NotImplementedError(f"backbone={name!r} not yet supported (PRs welcome)")


# ----------------------------------------------------------------------
# Cache one split
# ----------------------------------------------------------------------
@torch.no_grad()
def cache_split(model, name: str, split: str, args, out_path: Path):
    ds = get_dataset(args.dataset, split=split, img_size=args.img_size)
    print(f"\n[{split}] dataset has {len(ds):,} samples")

    # Important: shuffle=False so index-to-prediction mapping is stable.
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_scale = PIXEL_SCALES[args.dataset]
    H = W = args.img_size

    if args.mixed_precision == "bf16":
        amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    elif args.mixed_precision == "fp16":
        amp = dict(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")
    else:
        amp = dict(device_type="cuda", enabled=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"  [skip] {out_path} already exists; remove to recompute")
        return

    # Pre-allocate; uint16 stores 0..pixel_scale exactly when pixel_scale<=65535.
    n = len(ds)
    with h5py.File(out_path, "w") as f:
        preds_ds = f.create_dataset(
            "preds",
            shape=(n, args.T_out, 1, H, W),
            dtype="uint16",
            chunks=(1, args.T_out, 1, H, W),
            compression="lzf",
        )
        sample_idx = f.create_dataset(
            "sample_idx",
            shape=(n,),
            dtype="int32",
        )
        f.attrs["pixel_scale"] = float(pixel_scale)
        f.attrs["backbone"] = "alphapre"
        f.attrs["src_ckpt"] = str(args.ckpt)
        f.attrs["dataset"] = args.dataset
        f.attrs["split"] = split
        f.attrs["T_in"] = args.T_in
        f.attrs["T_out"] = args.T_out
        f.attrs["img_size"] = args.img_size

        idx = 0
        t0 = time.time()
        for frames in tqdm(dl, desc=f"cache {split}"):
            frames = frames.to(device, non_blocking=True)
            cond = frames[:, : args.T_in].contiguous()
            with torch.amp.autocast(**amp):
                pred, _ = model.predict(cond, compute_loss=False)
            # pred: (B, T_out, C, H, W) float in [0, 1]
            pred = pred.float().clamp_(0.0, 1.0).cpu().numpy()
            pred_u16 = np.round(pred * pixel_scale).astype(np.uint16)
            B = pred_u16.shape[0]
            preds_ds[idx : idx + B] = pred_u16
            sample_idx[idx : idx + B] = np.arange(idx, idx + B, dtype=np.int32)
            idx += B

        dt = time.time() - t0
        print(
            f"  wrote {idx:,} preds to {out_path}  "
            f"({dt:.1f}s, {idx / dt:.1f} samp/s, {out_path.stat().st_size / 1e9:.2f} GB)"
        )


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    # build_backbone() may chdir() inside setup(); resolve any relative paths up-front.
    args.ckpt = Path(args.ckpt).expanduser().resolve()
    args.out_dir = Path(args.out_dir).expanduser().resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}  backbone={args.backbone}  ckpt={args.ckpt}")
    print(f"[out] {args.out_dir}")

    model = (
        build_backbone(
            args.backbone,
            ckpt_path=args.ckpt,
            img_size=args.img_size,
            T_in=args.T_in,
            T_out=args.T_out,
        )
        .to(device)
        .eval()
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        out = args.out_dir / f"{split}.h5"
        cache_split(model, args.backbone, split, args, out)

    print("\nDone.  Cached files:")
    for p in sorted(args.out_dir.glob("*.h5")):
        print(f"  {p}  ({p.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
