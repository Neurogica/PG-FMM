"""Loaders for the three DiffCast-bundled h5 datasets.

These are minimal, self-contained adaptations of the dataset classes in the
DiffCast official repo (`ext_repos/DiffCast/datasets/dataset_*.py`).
We keep the same h5 layout & semantics so checkpoints/metrics remain comparable.

h5 layout summary
-----------------
* meteo_radar.h5   : root has 'train', 'test' groups (str(int) keys, shape (25, 565, 784) uint8) +
                     scalar 'train_len', 'test_len' at root.
* shanghai.h5      : root has 'train', 'test' groups; each group has 'all_len' scalar +
                     str(int) keys, shape (25, 501, 501) uint8.
* cikm.h5          : root has 'train','test','valid' groups (key='sample_<n>', shape (15,101,101) uint8) +
                     'train_len','test_len','valid_len' scalars at root.

All loaders return a (T, 1, img_size, img_size) float tensor in [0, 1].
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# -----------------------------------------------------------------------------
# Meteo (MeteoNet)
# -----------------------------------------------------------------------------
class Meteo(Dataset):
    PIXEL_SCALE = 90.0

    def __init__(self, data_path: str, img_size: int = 128, type: str = 'train'):
        super().__init__()
        assert type in ('train', 'test', 'val')
        # MeteoNet bundle ships only train/test → reuse 'test' as 'val'.
        self.type = type if type != 'val' else 'test'
        self.data_path = data_path
        self.img_size = img_size
        with h5py.File(data_path, 'r') as f:
            self.all_len = int(f[f'{self.type}_len'][()])
        self.transform = transforms.Resize((img_size, img_size), antialias=True)

    def __len__(self):
        return self.all_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.data_path, 'r') as f:
            arr = f[self.type][str(idx)][()]              # (25, 565, 784) uint8
        x = torch.from_numpy(arr).float() / self.PIXEL_SCALE   # → [0, 1]
        x = self.transform(x)                              # (25, img_size, img_size)
        return x.unsqueeze(1)                              # (25, 1, H, W)


# -----------------------------------------------------------------------------
# Shanghai (Shanghai_Radar)
# -----------------------------------------------------------------------------
class Shanghai(Dataset):
    PIXEL_SCALE = 255.0

    def __init__(self, data_path: str, img_size: int = 128, type: str = 'train'):
        super().__init__()
        assert type in ('train', 'test', 'val')
        self.type = type if type != 'val' else 'test'
        self.data_path = data_path
        self.img_size = img_size
        with h5py.File(data_path, 'r') as f:
            self.all_len = int(f[self.type]['all_len'][()])
        self.transform = transforms.Resize((img_size, img_size), antialias=True)

    def __len__(self):
        return self.all_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.data_path, 'r') as f:
            arr = f[self.type][str(idx)][()]              # (25, 501, 501) uint8
        x = torch.from_numpy(arr).float() / self.PIXEL_SCALE
        x = self.transform(x)
        return x.unsqueeze(1)


# -----------------------------------------------------------------------------
# CIKM (CIKM_Radar)
# -----------------------------------------------------------------------------
class CIKM(Dataset):
    PIXEL_SCALE = 255.0

    def __init__(self, data_path: str, img_size: int = 128, type: str = 'train'):
        super().__init__()
        assert type in ('train', 'test', 'valid')
        self.type = type
        self.data_path = data_path
        self.img_size = img_size
        with h5py.File(data_path, 'r') as f:
            self.length = int(f[f'{type}_len'][()])
        # CenterCrop with padding when img_size > 101.
        self.transform = transforms.CenterCrop((img_size, img_size))

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        key = f'sample_{idx + 1}'
        with h5py.File(self.data_path, 'r') as f:
            arr = f[self.type][key][()]                    # (15, 101, 101) uint8
        x = torch.from_numpy(arr).float() / self.PIXEL_SCALE
        x = self.transform(x)
        return x.unsqueeze(1)
