"""Central registry that maps a dataset name to its loader + metadata."""

from __future__ import annotations
import os
from pathlib import Path

# Project root resolved once at import time.
_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3]  # PG-FMM repo root
DATA_ROOT = Path(os.environ.get('PGFMM_DATA_ROOT', ROOT / 'data'))


# Dataset paths.  All of MeteoNet/Shanghai/CIKM are single .h5 files
# downloaded from the DiffCast GoogleDrive bundle.
# SEVIR is a directory with CATALOG.csv + data/vil/<year>/*.h5.
DATASET_PATHS = {
    'sevir':    DATA_ROOT / 'sevir',
    'meteo':    DATA_ROOT / 'diffcast' / 'meteo_radar.h5',
    'shanghai': DATA_ROOT / 'diffcast' / 'shanghai.h5',
    'cikm':     DATA_ROOT / 'diffcast' / 'cikm.h5',
}

# Evaluator ``value_scale`` passed to AlphaPre's Evaluator.float2int (NOT the loader
# divisor).  Loaders still use /255 for Shanghai & CIKM h5 uint8; thresholds
# {20,30,35,40} are applied after ``pred * value_scale`` (see AlphaPre
# dataset_shanghai.PIXEL_SCALE=90, dataset_cikm.PIXEL_SCALE=80).
PIXEL_SCALES = {
    'sevir':    255.0,
    'meteo':    90.0,
    'shanghai': 90.0,    # AlphaPre dataset_shanghai.py
    'cikm':     80.0,    # AlphaPre dataset_cikm.py (DiffCast uses 90; paper Table 1 = AlphaPre)
}

# Intensity thresholds for CSI / POD / FAR computation (after rescaling
# back via PIXEL_SCALES).  Match AlphaPre / DiffCast paper tables.
THRESHOLDS = {
    'sevir':    [16, 74, 133, 160, 181, 219],   # SEVIR VIL paper convention
    'meteo':    [12, 18, 24, 32],
    'shanghai': [20, 30, 35, 40],
    'cikm':     [20, 30, 35, 40],
}

# Total frame count per sample.  Default split: 5 input -> 20 output (15 for CIKM).
SEQ_LENS = {
    'sevir':    25,   # 5 + 20
    'meteo':    25,
    'shanghai': 25,
    'cikm':     15,   # 5 + 10  (CIKM only has 15 frames per sample)
}

# Shanghai has no official val split; reserve this tail fraction of train
# for model selection so best.pt is never chosen on the test set.
SHANGHAI_VAL_FRACTION = 0.1

# Spatial split for input/output (in_seq, out_seq)
INOUT_SPLIT = {
    'sevir':    (5, 20),
    'meteo':    (5, 20),
    'shanghai': (5, 20),
    'cikm':     (5, 10),
}


def dataset_kwargs_from_cfg(ds_cfg) -> dict:
    """Extract optional SEVIR sampling kwargs from an OmegaConf ``dataset`` block."""
    kw: dict = {}
    if ds_cfg.get('stride') is not None:
        kw['stride'] = int(ds_cfg.stride)
    if ds_cfg.get('seq_len') is not None:
        kw['seq_len'] = int(ds_cfg.seq_len)
    elif ds_cfg.get('T_in') is not None and ds_cfg.get('T_out') is not None:
        kw['seq_len'] = int(ds_cfg.T_in) + int(ds_cfg.T_out)
    if ds_cfg.get('eval_tail_multiple') is not None:
        kw['eval_tail_multiple'] = int(ds_cfg.eval_tail_multiple)
    if ds_cfg.get('samples_per_event') is not None:
        kw['samples_per_event'] = int(ds_cfg.samples_per_event)
    return kw


def get_dataset(name: str, split: str = 'train', img_size: int = 128, **kw):
    """Build a torch Dataset for *name* / *split*.

    Args
    ----
    name      : 'sevir' | 'meteo' | 'shanghai' | 'cikm'
    split     : 'train' | 'val' | 'test'
    img_size  : output spatial resolution after Resize/CenterCrop (default 128)

    Returns
    -------
    torch.utils.data.Dataset whose __getitem__ returns a (T, 1, img_size, img_size)
    float tensor in [0, 1].
    """
    name = name.lower()
    path = DATASET_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(
            f'Dataset {name!r} not found at {path}. '
            f'Run scripts/download_data.sh first.'
        )

    if name == 'sevir':
        from .sevir import build_sevir
        return build_sevir(path, split=split, img_size=img_size, **kw)
    if name == 'meteo':
        from .diffcast_h5 import Meteo
        return Meteo(str(path), img_size=img_size, type=split)
    if name == 'shanghai':
        from .diffcast_h5 import Shanghai
        # Shanghai ships only train/test.  The upstream loader aliases
        # 'val' -> 'test', which would select best.pt on the *test* set
        # (a leak).  Instead we carve a deterministic holdout from the tail
        # of the training split for validation/model-selection, leaving the
        # test split untouched for the final single eval.
        if split in ('train', 'val'):
            from torch.utils.data import Subset
            full_train = Shanghai(str(path), img_size=img_size, type='train')
            n = len(full_train)
            n_val = max(1, int(round(n * SHANGHAI_VAL_FRACTION)))
            val_indices = list(range(n - n_val, n))
            train_indices = list(range(0, n - n_val))
            return Subset(full_train, val_indices if split == 'val' else train_indices)
        return Shanghai(str(path), img_size=img_size, type='test')
    if name == 'cikm':
        from .diffcast_h5 import CIKM
        # CIKM uses 'valid' as the split name internally.
        cikm_split = 'valid' if split in ('val', 'valid') else split
        return CIKM(str(path), img_size=img_size, type=cikm_split)
    raise KeyError(name)
