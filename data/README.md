# `data/` — dataset layout

No data is committed. Point `PGFMM_DATA_ROOT` here (default) and arrange the
four radar benchmarks as below.

```
data/
├── sevir/                      # SEVIR-VIL
│   ├── CATALOG.csv
│   └── data/vil/<year>/*.h5
└── diffcast/                   # MeteoNet / Shanghai / CIKM (DiffCast bundle)
    ├── meteo_radar.h5
    ├── shanghai.h5
    └── cikm.h5
```

## Sources

| Dataset | Frames (in→out) | Pixel scale | Source |
|---|---|---|---|
| SEVIR-VIL | 5 → 20 (5-min) | 0–255 VIL | [SEVIR](https://sevir.mit.edu/) (NEXRAD VIL) |
| MeteoNet | 5 → 20 | ÷90 | [DiffCast](https://github.com/DeminYu98/DiffCast) bundle |
| Shanghai | 5 → 20 | ÷90 | DiffCast bundle |
| CIKM | 5 → 10 | ÷80 | DiffCast bundle |

We evaluate under the **AlphaPre** protocol (test window and thresholds) so numbers are
directly comparable to prior work. Thresholds: SEVIR `{16,74,133,160,181,219}`,
MeteoNet `{12,18,24,32}`, Shanghai/CIKM `{20,30,35,40}`.

## Cached predictions (optional)

`ml/eval.py` and the flow-map conditioning can use cached baseline/prior rollouts under
`data/<dataset>_pred/<name>/{train,val,test}.h5` — build them with
`ml/scripts/cache_lagrangian_source.py` (frozen prior) and `ml/scripts/cache_predictions.py`.
Format: `preds (N, T_out, 1, H, W) uint16` + `sample_idx (N,)`.
