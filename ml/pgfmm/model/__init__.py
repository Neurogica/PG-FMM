"""PG-FMM generative model: the Flow-Map Matching head and its training/sampling.

``PGFMMRunner`` (see ``runner``) instantiates the two-time flow-map operator
``Phi_{t->r}(x_t | cond)`` that maps noise to the forecast in a few steps,
conditioned on the past frames and the frozen Lagrangian prior rollout. The
runner also retains two diffusion-bridge transport options (``i2sb``, ``ddbm``)
as baselines; the flow-map path is the default and the one used in the paper.

Module layout
-------------
* ``runner``    : ``PGFMMRunner`` glue (train_step / sample); flow-map + baselines
* ``unet``      : diffusers-based 2D U-Net wired for nowcasting
* ``prior``     : ways to build the transport endpoint ``x_1`` from past frames
* ``schedule``  : beta schedules + sub-grid index helpers (baseline transports)
* ``diffusion`` : I²SB-style tractable bridge math (baseline transport)
* ``ddbm``      : DDBM-style bridge math (baseline transport)
"""

from .ddbm import DDBMBridge, DDBMConfig
from .diffusion import I2SBDiffusion
from .prior import build_prior
from .runner import PGFMMConfig, PGFMMRunner
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
    "PGFMMConfig",
    "PGFMMRunner",
    "build_prior",
    "make_beta_schedule",
    "make_symmetric_beta_schedule",
    "space_indices",
    "unsqueeze_xdim",
]
