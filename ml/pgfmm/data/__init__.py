"""Dataset loaders for the PG-FMM (Flow-Map Matching) nowcasting project.

The 4 datasets (SEVIR / MeteoNet / Shanghai_Radar / CIKM_Radar) are loaded
following the AlphaPre/DiffCast convention:

    frames = dataset[i]   # torch.Tensor, shape (T, 1, H, W), dtype=float32, in [0, 1]
        T = 25 (5 input + 20 output) for SEVIR / Meteo / Shanghai
        T = 15 for CIKM
        H = W = img_size  (typically 128)

Pixel scale and intensity thresholds (for CSI etc.) are exposed per dataset.
"""

from .registry import (
    DATASET_PATHS,
    PIXEL_SCALES,
    THRESHOLDS,
    SEQ_LENS,
    dataset_kwargs_from_cfg,
    get_dataset,
)

__all__ = [
    'DATASET_PATHS',
    'PIXEL_SCALES',
    'THRESHOLDS',
    'SEQ_LENS',
    'dataset_kwargs_from_cfg',
    'get_dataset',
]
