# `ml/` — PG-FMM method code

Installable package `pgfmm` plus training/evaluation entrypoints.

## Install

```bash
# from the repo root
uv sync            # or: pip install -e .
```

Core deps: PyTorch, diffusers, accelerate, ema-pytorch, omegaconf, einops, h5py, scipy.
Evaluation metrics reuse the **AlphaPre** benchmark `Evaluator` (for numbers directly
comparable to prior work); clone it once into `ext_repos/AlphaPre/` and install `lpips`:

```bash
git clone https://github.com/... ext_repos/AlphaPre    # AlphaPre (CVPR'25) evaluator
uv sync --extra eval
```

## Data

Set the data root (defaults to `../data`) and follow [`../data/README.md`](../data/README.md):

```bash
export PGFMM_DATA_ROOT=/path/to/data
```

## Train

PG-FMM is two frozen-then-generative stages. Train the Lagrangian prior first, then the
flow-map head (its config points at the frozen prior checkpoint via `motion.ckpt`).

```bash
# Stage 1 — Lagrangian advection prior (per dataset)
python ml/train.py --config ml/configs/sevir/lagrangian_prior.yaml --note lagrangian_prior

# Stage 2 — Flow-Map Matching head (conditioned on the frozen prior)
python ml/train.py --config ml/configs/sevir/pgfmm.yaml --note pgfmm
```

Outputs go to `runs/<dataset>/<note>/` (override with `PGFMM_OUT`). Datasets:
`sevir`, `meteo`, `cikm`, `shanghai`.

## Evaluate

```bash
python ml/eval.py \
  --ckpt runs/sevir/pgfmm/checkpoints/last.pt \
  --config ml/configs/sevir/pgfmm.yaml \
  --split test --ensemble 16 --ensemble_agg pmm --nfe 4 --use_ema \
  --csv_out results/sevir.csv
```

Reports CSI (per threshold + mean), HSS, SSIM, MSE, CRPS, LPIPS under the AlphaPre protocol.

## Package map

| Module | Contents |
|---|---|
| `pgfmm.bridge` | Flow-Map Matching runner (`SBNowcastRunner`), U-Net, samplers, schedules |
| `pgfmm.source.lagrangian` | Lagrangian advection prior (motion + source, semi-Lagrangian rollout) |
| `pgfmm.data` | SEVIR / MeteoNet / CIKM / Shanghai loaders, cached-prediction pairing |
| `pgfmm.losses` | Physics regularizers (mass conservation, smoothness, advection–diffusion) |
| `pgfmm.metrics` | Table-3-style probabilistic metrics |
