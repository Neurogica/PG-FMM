"""Wrapper Dataset that pairs each frame sequence with a cached backbone prediction.

Used by the residual training loop: for each ground-truth sample we hand the
flow map the corresponding pre-computed AlphaPre / DiffCast / SimVP /
... output.  The flow map then transports

    x_1 = backbone_pred  →  x_0 = ground_truth

instead of starting from the trivial ``last_frame`` prior.

Layout of the cache (produced by ``scripts/cache_predictions.py``):

    <cache_path>.h5
        preds        : (N, T_out, C, H, W) uint16  (= round(pred_float * pixel_scale))
        sample_idx   : (N,) int32                  = the index into the *base* dataset
        attrs/pixel_scale, attrs/backbone, attrs/...
"""

from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class PredCache:
    """Thin wrapper around a cached-prediction h5 file.

    Opens the file lazily per worker (``__init__`` doesn't touch h5; we open
    inside ``__getitem__``-equivalent calls) so it survives DataLoader fork.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with h5py.File(self.path, "r") as f:
            self.length = int(f["preds"].shape[0])
            self.shape  = tuple(f["preds"].shape[1:])           # (T_out, C, H, W)
            self.dtype  = f["preds"].dtype
            self.pixel_scale = float(f.attrs["pixel_scale"])
            self.backbone = str(f.attrs.get("backbone", "?"))
            self.dataset_name = str(f.attrs.get("dataset", "?"))
            self.split = str(f.attrs.get("split", "?"))
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return self.length

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns a (T_out, C, H, W) float tensor in [0, 1]."""
        f = self._open()
        arr = f["preds"][idx]                                    # uint16
        x = torch.from_numpy(arr).float() / self.pixel_scale
        return x

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


class PairedDataset(Dataset):
    """``base_dataset`` + matching prediction from a cache.

    Returns ``(frames, pred)`` where:
        * ``frames`` is whatever ``base_dataset[idx]`` returns
                     – usually shape ``(T_in + T_out, C, H, W)`` in [0, 1]
        * ``pred``   is the cached backbone prediction for that same idx,
                     shape ``(T_out, C, H, W)`` in [0, 1]
    """

    def __init__(self, base_dataset: Dataset, cache_path: str | Path):
        self.base = base_dataset
        self.cache = PredCache(cache_path)
        if len(self.base) != len(self.cache):
            raise ValueError(
                f"length mismatch: base={len(self.base)}, cache={len(self.cache)}; "
                f"did you cache the same split with the same dataset settings?"
            )
        # Defensive shape check on first sample (safe at __init__ – no fork yet).
        sample = self.base[0]
        cached = self.cache[0]
        if sample.shape[-3:] != cached.shape[-3:]:
            raise ValueError(
                f"shape mismatch: base last 3 dims={sample.shape[-3:]}, "
                f"cache last 3 dims={cached.shape[-3:]}"
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        return self.base[idx], self.cache[idx]


class MultiCachePairedDataset(Dataset):
    """``base_dataset`` + primary prediction cache + extra conditioning caches.

    The first cache is the actual prior endpoint/backbone prediction
    (e.g. AlphaPre). Additional caches are concatenated along the temporal
    dimension and used only as extra conditioning, not as the residual base.

    Returns ``(frames, pred, extra_cond)`` where ``extra_cond`` has shape
    ``(sum(T_i), C, H, W)``.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        primary_cache_path: str | Path,
        extra_cache_paths: list[str | Path],
    ):
        self.base = base_dataset
        self.cache = PredCache(primary_cache_path)
        self.extra_caches = [PredCache(p) for p in extra_cache_paths]
        caches = [self.cache, *self.extra_caches]
        for cache in caches:
            if len(self.base) != len(cache):
                raise ValueError(
                    f"length mismatch: base={len(self.base)}, cache={len(cache)} "
                    f"at {cache.path}"
                )
        sample = self.base[0]
        for cache in caches:
            cached = cache[0]
            if sample.shape[-3:] != cached.shape[-3:]:
                raise ValueError(
                    f"shape mismatch for {cache.path}: base last 3 dims={sample.shape[-3:]}, "
                    f"cache last 3 dims={cached.shape[-3:]}"
                )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        primary = self.cache[idx]
        extras = [cache[idx] for cache in self.extra_caches]
        extra = torch.cat(extras, dim=0) if len(extras) > 1 else extras[0]
        return self.base[idx], primary, extra
