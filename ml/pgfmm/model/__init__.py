"""PG-FMM generative model: the Flow-Map Matching head and its training/sampling.

``PGFMMRunner`` (see ``runner``) instantiates the two-time flow-map operator
``Phi_{t->r}(x_t | cond)`` that maps noise to the forecast in a few steps,
conditioned on the past frames and the frozen Lagrangian prior rollout.

Module layout
-------------
* ``runner``    : ``PGFMMRunner`` glue (train_step / sample) for the flow map
* ``unet``      : diffusers-based 2D U-Net wired for nowcasting
* ``prior``     : ways to build the transport source endpoint ``x_1`` from past frames
"""

from .prior import build_prior
from .runner import PGFMMConfig, PGFMMRunner
from .unet import NowcastUNet, NowcastUNetConfig

__all__ = [
    "NowcastUNet",
    "NowcastUNetConfig",
    "PGFMMConfig",
    "PGFMMRunner",
    "build_prior",
]
