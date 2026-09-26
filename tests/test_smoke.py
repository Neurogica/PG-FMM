"""CPU smoke tests: package imports, the Lagrangian prior behaves physically,
and a tiny PG-FMM runner trains one step and samples end-to-end.

Everything runs on random tensors at 32x32 in a few seconds; no data or GPU needed.
"""

import torch

from pgfmm.model.runner import PGFMMConfig, PGFMMRunner
from pgfmm.source.lagrangian import (
    LagrangianSourceConfig,
    LagrangianSourceNet,
    rollout_lagrangian,
    warp_with_flow,
)

T_IN, T_OUT, SIZE = 3, 4, 32


def tiny_runner(**over):
    cfg = PGFMMConfig(
        T_in=T_IN,
        T_out=T_OUT,
        img_size=SIZE,
        base_channels=16,
        channel_mult=(1, 2),
        attention_resolutions=(),
        motion_prior_base_channels=16,
        motion_prior_channel_mult=(1, 2),
        motion_prior_num_blocks=1,
        **over,
    )
    return PGFMMRunner(cfg, device="cpu")


def test_imports():
    import pgfmm  # noqa: F401
    from pgfmm import losses  # noqa: F401
    from pgfmm.data import registry  # noqa: F401
    from pgfmm.model import prior, unet  # noqa: F401


def test_warp_identity():
    x = torch.rand(2, 1, SIZE, SIZE)
    flow = torch.zeros(2, 2, SIZE, SIZE)
    out = warp_with_flow(x, flow)
    assert torch.allclose(out, x, atol=1e-5)


def test_warp_shifts_mass():
    x = torch.zeros(1, 1, SIZE, SIZE)
    x[..., 8:12, 8:12] = 1.0
    flow = torch.zeros(1, 2, SIZE, SIZE)
    flow[:, 0] = 5.0  # uniform shift
    out = warp_with_flow(x, flow)
    assert not torch.allclose(out, x)
    assert out.sum() > 0.5 * x.sum()  # mass mostly preserved under uniform shift


def test_lagrangian_prior_forward():
    net = LagrangianSourceNet(LagrangianSourceConfig(
        T_in=T_IN, T_out=T_OUT, img_size=SIZE,
        base_channels=16, channel_mult=(1, 2), num_blocks=1))
    cond = torch.rand(2, T_IN, 1, SIZE, SIZE)
    pred, aux = net(cond)
    assert pred.shape == (2, T_OUT, 1, SIZE, SIZE)
    assert aux["flow"].shape == (2, T_OUT, 2, SIZE, SIZE)
    assert aux["source"].shape == (2, T_OUT, 1, SIZE, SIZE)
    assert pred.min() >= 0.0 and pred.max() <= 1.0
    # rollout is reproducible from its own flow/source fields
    re = rollout_lagrangian(cond[:, -1], aux["flow"], aux["source"])
    assert torch.allclose(re, pred, atol=1e-5)


def test_train_step_and_sample_pure_generative():
    torch.manual_seed(0)
    runner = tiny_runner()
    frames = torch.rand(2, T_IN + T_OUT, 1, SIZE, SIZE)
    out = runner.train_step(frames, global_step=0)
    loss = out["total_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in runner.net.parameters())

    with torch.no_grad():
        pred = runner.sample(frames[:, :T_IN], nfe=2, verbose=False)
    assert pred.shape == (2, T_OUT, 1, SIZE, SIZE)
    assert pred.min() >= 0.0 and pred.max() <= 1.0


def test_train_step_and_sample_with_physics_prior():
    torch.manual_seed(0)
    runner = tiny_runner(motion_prior=True)
    # frozen prior: no trainable params
    assert all(not p.requires_grad for p in runner.motion_prior.parameters())
    frames = torch.rand(1, T_IN + T_OUT, 1, SIZE, SIZE)
    out = runner.train_step(frames, global_step=0)
    assert torch.isfinite(out["total_loss"])

    with torch.no_grad():
        pred = runner.sample(frames[:, :T_IN], nfe=2, verbose=False)
    assert pred.shape == (1, T_OUT, 1, SIZE, SIZE)


def test_sampling_is_stochastic():
    torch.manual_seed(0)
    runner = tiny_runner()
    cond = torch.rand(1, T_IN, 1, SIZE, SIZE)
    with torch.no_grad():
        torch.manual_seed(1)
        a = runner.sample(cond, nfe=2, verbose=False)
        torch.manual_seed(2)
        b = runner.sample(cond, nfe=2, verbose=False)
    assert not torch.allclose(a, b)
