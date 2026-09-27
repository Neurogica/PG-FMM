"""Minimal SEVIR (VIL) loader compatible with AlphaPre / DiffCast convention.

We do **not** depend on the upstream `eie-sevir` python package; we just read
the official h5 files directly using h5py + the CATALOG.csv index.

Each VIL h5 holds a `vil` dataset of shape (N, 384, 384, 49) uint8, where
*N* = number of events in that h5, indexed by `file_index` in CATALOG.csv.
Each event is 49 frames spaced 5 min apart over 4 hours.

For our 5->20 setting we need T=25 frames (= 25 x 5min = 2 hours).  By
default we follow AlphaPre's official SEVIR protocol: filter out missing VIL
events, remove duplicated event IDs, then sample sub-sequences of length 25
with stride=13.

    train : time_utc <= 2019-01-01
    val   : 2019-01-01 < time_utc <= 2019-06-01
    test  : 2019-06-01 < time_utc <= 2019-12-31
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


SEVIR_VIL_FRAMES_PER_EVENT = 49
SEVIR_VIL_HW = 384


def _parse_date(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def _read_catalog_vil(catalog_csv: Path) -> List[dict]:
    rows = []
    with open(catalog_csv) as f:
        for row in csv.DictReader(f):
            if row['img_type'] != 'vil':
                continue
            rows.append(row)
    return rows


class SEVIRVIL(Dataset):
    """Minimal SEVIR-VIL dataset returning (T, 1, img_size, img_size) float in [0,1].

    Sub-sequence sampling: each event of 49 frames yields:

        1 + (49 - seq_len) // stride

    samples by default, exactly matching AlphaPre's `SEVIRDataLoader`.
    For seq_len=25 and stride=13 this gives starts [0, 13].

    Args
    ----
    sevir_dir         : directory with CATALOG.csv and data/vil/<year>/*.h5
    split             : 'train' | 'val' | 'test'
    img_size          : output H=W (default 128)
    seq_len           : T (default 25)
    stride            : AlphaPre-style sub-sequence stride (default 13)
    samples_per_event : legacy override; if set, uses evenly spaced starts
                        instead of AlphaPre stride sampling.
    """

    ALPHAPRE_DATE_RANGES = {
        'train': (None, dt.datetime(2019, 1, 1)),
        'val':   (dt.datetime(2019, 1, 1), dt.datetime(2019, 6, 1)),
        'test':  (dt.datetime(2019, 6, 1), dt.datetime(2019, 12, 31)),
    }

    def __init__(self,
                 sevir_dir: str | Path,
                 split: str = 'train',
                 img_size: int = 128,
                 seq_len: int = 25,
                 stride: int = 13,
                 samples_per_event: int | None = None,
                 eval_tail_multiple: int = 16,
                 require_pct_missing_zero: bool = True):
        super().__init__()
        assert split in self.ALPHAPRE_DATE_RANGES
        self.sevir_dir = Path(sevir_dir)
        self.split = split
        self.img_size = img_size
        self.seq_len = seq_len
        self.stride = stride
        self.samples_per_event = samples_per_event
        self.eval_tail_multiple = eval_tail_multiple
        self.require_pct_missing_zero = require_pct_missing_zero
        self.transform = transforms.Resize((img_size, img_size), antialias=True)

        catalog = self.sevir_dir / 'CATALOG.csv'
        rows = _read_catalog_vil(catalog)
        d0, d1 = self.ALPHAPRE_DATE_RANGES[split]
        rows = list(self._dedupe_like_alphapre(rows))

        # Filter by date and check h5 exists.  AlphaPre uses start_date <
        # time_utc <= end_date, with start_date=None for train.
        kept: List[Tuple[Path, int]] = []
        seen_files = set()
        missing_files = set()
        for r in rows:
            t = _parse_date(r['time_utc'])
            after_start = True if d0 is None else t > d0
            before_end = True if d1 is None else t <= d1
            if not (after_start and before_end):
                continue
            if require_pct_missing_zero and float(r.get('pct_missing', 0) or 0) != 0.0:
                continue
            h5_rel = r['file_name']                 # e.g. 'vil/2018/SEVIR_VIL_...h5'
            h5_path = self.sevir_dir / 'data' / h5_rel
            seen_files.add(h5_path.name)
            if not h5_path.exists():
                missing_files.add(h5_path.name)
                continue
            kept.append((h5_path, int(r['file_index'])))
        self.events = kept
        self._missing = missing_files
        if not self.events:
            raise RuntimeError(
                f'No SEVIR events found for split={split!r} under {self.sevir_dir}.\n'
                f'Looked at {len(rows)} catalog rows in date range; '
                f'{len(missing_files)} h5 files missing: {sorted(missing_files)[:3]}'
            )

        # Sub-sequence start positions per event
        if samples_per_event is None:
            n_seq = 1 + (SEVIR_VIL_FRAMES_PER_EVENT - seq_len) // stride
            self._starts = [i * stride for i in range(n_seq)]
        elif samples_per_event == 1:
            self._starts = [0]
        else:
            stride = max(1, (SEVIR_VIL_FRAMES_PER_EVENT - seq_len) // (samples_per_event - 1))
            self._starts = [s for s in range(0, SEVIR_VIL_FRAMES_PER_EVENT - seq_len + 1, stride)][:samples_per_event]
        raw_len = len(self.events) * len(self._starts)
        if split in {'val', 'test'} and eval_tail_multiple > 1:
            # AlphaPre's val/test wrapper returns floor(total_seq / inner_batch)
            # batches.  With the public default batch_size=8, the inner batch is
            # 16, giving the paper-matched SEVIR test count of 8096.
            self._length = raw_len // eval_tail_multiple * eval_tail_multiple
        else:
            self._length = raw_len

    @staticmethod
    def _dedupe_like_alphapre(rows: Iterable[dict]) -> Iterable[dict]:
        """Keep only event IDs that have exactly one VIL row, sorted by ID.

        AlphaPre groups the catalog by `id`, drops repeated IDs, and then
        indexes samples from that grouped table.  Matching that order keeps our
        cached backbone predictions aligned with official-style evaluation.
        """
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(row['id'], []).append(row)
        for event_id in sorted(groups):
            group = groups[event_id]
            if len(group) == 1:
                yield group[0]

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int) -> torch.Tensor:
        evt_idx, start_idx = divmod(idx, len(self._starts))
        h5_path, file_index = self.events[evt_idx]
        start = self._starts[start_idx]
        end = start + self.seq_len
        with h5py.File(h5_path, 'r') as f:
            arr = f['vil'][file_index, :, :, start:end]      # (384, 384, T) uint8
        x = torch.from_numpy(arr.astype(np.float32) / 255.0)  # → [0,1]
        x = x.permute(2, 0, 1)                                # (T, H, W)
        x = self.transform(x)                                 # (T, img_size, img_size)
        return x.unsqueeze(1)                                 # (T, 1, H, W)


def build_sevir(sevir_dir: str | Path, split: str = 'train', img_size: int = 128, **kw) -> SEVIRVIL:
    """Compat wrapper used by registry.get_dataset()."""
    return SEVIRVIL(sevir_dir=sevir_dir, split=split, img_size=img_size, **kw)
