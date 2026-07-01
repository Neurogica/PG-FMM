"""Schrödinger Bridge for Precipitation Nowcasting (ACCV 2026, ours).

Module layout
-------------
* ``schedule``  : beta schedules + sub-grid index helpers
* ``diffusion`` : I²SB-style tractable bridge math (q_sample / pred_x0 / sampler)
* ``prior``     : ways to build the bridge endpoint ``x_1`` from past frames
* ``unet``      : diffusers-based 2D U-Net wired for nowcasting
* ``runner``    : ``SBNowcastRunner`` glue (train_step / sample)
"""

from .ddbm import DDBMBridge, DDBMConfig
from .diffusion import I2SBDiffusion
from .prior import build_prior
from .runner import SBNowcastConfig, SBNowcastRunner
from .schedule import (
    make_beta_schedule,
    make_symmetric_beta_schedule,
    space_indices,
    unsqueeze_xdim,
)
from .unet import NowcastUNet, NowcastUNetConfig

__all__ = [
    "I2SBDiffusion",
    "DDBMBridge",
    "DDBMConfig",
    "NowcastUNet",
    "NowcastUNetConfig",
    "SBNowcastConfig",
    "SBNowcastRunner",
    "build_prior",
    "make_beta_schedule",
    "make_symmetric_beta_schedule",
    "space_indices",
    "unsqueeze_xdim",
]
