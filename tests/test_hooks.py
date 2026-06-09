"""Unit tests for src/helpers/hook_utils.py (ActivationCapper, SteeringHook)."""

import sys
from pathlib import Path

import torch as t
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.helpers.hook_utils import ActivationCapper, SteeringHook
from src.helpers.model_utils import _return_layers

D_MODEL = 16
N_LAYERS = 4


class DummyBlock(nn.Module):
    """Identity transformer block returning a (hidden_states,) tuple like HF blocks."""

    def forward(self, x):
        return (x,)


class DummyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DummyBlock() for _ in range(N_LAYERS)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


class DummyModel(nn.Module):
    """Mimics the HF nesting (model.model.layers) so _return_layers works."""

    def __init__(self):
        super().__init__()
        self.model = DummyInner()

    def forward(self, x):
        return self.model(x)


def _unit(i: int):
    v = t.zeros(D_MODEL)
    v[i] = 1.0
    return v


def test_return_layers_on_dummy():
    model = DummyModel()
    layers = _return_layers(model)
    assert len(layers) == N_LAYERS


def test_capper_identity_above_threshold():
    """Ceiling mode: if all projections are below threshold, output is unchanged."""
    model = DummyModel()
    v = _unit(0)
    x = t.randn(2, 5, D_MODEL).clamp(max=0.5)  # projections onto e0 <= 0.5 < tau
    with ActivationCapper(model, [v], [1.0], [1], mode="ceiling"):
        out = model(x)
    assert t.allclose(out, x, atol=1e-5)


def test_capper_clamps_above_threshold():
    """Ceiling mode: a large projection gets clamped down to exactly tau."""
    model = DummyModel()
    v = _unit(0)
    x = t.zeros(1, 3, D_MODEL)
    x[..., 0] = 7.0  # projection = 7 > tau = 2
    with ActivationCapper(model, [v], [2.0], [2], mode="ceiling"):
        out = model(x)
    proj = out[..., 0]
    assert t.allclose(proj, t.full_like(proj, 2.0), atol=1e-5)
    # Orthogonal components untouched
    assert t.allclose(out[..., 1:], x[..., 1:], atol=1e-6)


def test_capper_floor_mode_raises_low_projection():
    """Floor mode (Lu et al. eq. 1): projection below tau is raised to exactly tau."""
    model = DummyModel()
    v = _unit(0)
    x = t.zeros(1, 3, D_MODEL)
    x[..., 0] = -3.0  # projection = -3 < tau = 1
    with ActivationCapper(model, [v], [1.0], [0], mode="floor"):
        out = model(x)
    proj = out[..., 0]
    assert t.allclose(proj, t.full_like(proj, 1.0), atol=1e-5)


def test_capper_floor_mode_identity_above_threshold():
    """Floor mode: projections already above tau are unchanged."""
    model = DummyModel()
    v = _unit(0)
    x = t.zeros(1, 3, D_MODEL)
    x[..., 0] = 5.0  # projection = 5 >= tau = 1
    with ActivationCapper(model, [v], [1.0], [0], mode="floor"):
        out = model(x)
    assert t.allclose(out, x, atol=1e-5)


def test_capper_removes_hooks_on_exit():
    """After the context exits, the model behaves normally again."""
    model = DummyModel()
    v = _unit(0)
    x = t.zeros(1, 2, D_MODEL)
    x[..., 0] = 7.0
    with ActivationCapper(model, [v], [2.0], [1], mode="ceiling"):
        capped = model(x)
    assert not t.allclose(capped, x)
    assert t.allclose(model(x), x, atol=1e-6)


def test_steering_hook_adds_scaled_direction():
    """Steering adds coef * ||h|| * v_hat at every position; disable() restores."""
    model = DummyModel()
    v = _unit(3)
    coef = 0.5
    x = t.randn(2, 4, D_MODEL)
    hook = SteeringHook(v, layer_idx=1, steering_coef=coef, apply_to_all_tokens=True)
    hook.enable(model)
    try:
        out = model(x)
    finally:
        hook.disable()
    expected = x + coef * x.norm(dim=-1, keepdim=True) * F.normalize(v.float(), dim=0)
    assert t.allclose(out, expected, atol=1e-4)
    assert t.allclose(model(x), x, atol=1e-6)
