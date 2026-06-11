"""Unit tests for src/finetuning/orthogonalize.py (weight orthogonalization + B callback)."""

import sys
from pathlib import Path

import torch as t
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.finetuning.orthogonalize import (
    BOrthogonalizeCallback,
    orthogonalize_writers,
    writer_projection_norms,
)

VOCAB, D_MODEL, D_FF, N_LAYERS = 20, 16, 32, 3
t.manual_seed(0)


class DummyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(D_MODEL, D_FF, bias=False)
        self.down_proj = nn.Linear(D_FF, D_MODEL, bias=True)
        # PEFT-style adapter slot on the writer (named ...down_proj.lora_B.default)
        self.down_proj.lora_B = nn.ModuleDict({"default": nn.Linear(1, D_MODEL, bias=False)})


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttn()
        self.mlp = DummyMLP()

    def forward(self, h):
        h = h + self.self_attn.o_proj(h)
        return h + self.mlp.down_proj(F.relu(self.mlp.up_proj(h)))


class DummyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList([DummyLayer() for _ in range(N_LAYERS)])


class DummyLM(nn.Module):
    """Mimics HF nesting: model.model.embed_tokens / model.model.layers[i]."""

    def __init__(self):
        super().__init__()
        self.model = DummyInner()

    def forward(self, ids):
        h = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            h = layer(h)
        return h


def _rand_dir():
    return F.normalize(t.randn(D_MODEL), dim=0)


def test_residual_stream_cannot_carry_direction():
    model, r = DummyLM(), _rand_dir()
    modified = orthogonalize_writers(model, r)
    # embed + (o_proj, down_proj) per layer
    assert len(modified) == 1 + 2 * N_LAYERS
    ids = t.randint(0, VOCAB, (2, 7))
    h = model(ids)
    assert (h @ r).abs().max() < 1e-5  # every write is orthogonal to r, so h is too


def test_orthogonalize_is_idempotent():
    model, r = DummyLM(), _rand_dir()
    orthogonalize_writers(model, r)
    before = [p.detach().clone() for p in model.parameters()]
    orthogonalize_writers(model, r)
    for p0, p1 in zip(before, model.parameters()):
        assert t.allclose(p0, p1, atol=1e-6)


def test_writer_projection_norms_reports_zero_after():
    model, r = DummyLM(), _rand_dir()
    assert max(writer_projection_norms(model, r).values()) > 1e-2  # random weights write r
    orthogonalize_writers(model, r)
    assert max(writer_projection_norms(model, r).values()) < 1e-5


def test_layer_range_restricts_and_skips_embedding():
    model, r = DummyLM(), _rand_dir()
    embed_before = model.model.embed_tokens.weight.detach().clone()
    layer0_before = model.model.layers[0].self_attn.o_proj.weight.detach().clone()
    modified = orthogonalize_writers(model, r, layer_range=(1, 2))
    assert len(modified) == 4 and all("layers.1." in n or "layers.2." in n for n in modified)
    assert t.allclose(model.model.embed_tokens.weight, embed_before)
    assert t.allclose(model.model.layers[0].self_attn.o_proj.weight, layer0_before)


def test_b_callback_keeps_lora_b_orthogonal_across_steps():
    model, r = DummyLM(), _rand_dir()
    cb = BOrthogonalizeCallback(model, r)
    assert len(cb.b_weights) == N_LAYERS
    # B starts random here (zero in real LoRA): project once, then simulate an
    # optimizer step that pushes straight along the forbidden direction.
    cb.on_train_begin(None, None, None)
    assert cb.max_b_projection() < 1e-6
    with t.no_grad():
        for w in cb.b_weights:
            w += 0.5 * r.unsqueeze(1)
    assert cb.max_b_projection() > 1e-2
    cb.on_step_end(None, None, None)
    assert cb.max_b_projection() < 1e-6


def test_bf16_roundtrip_leaves_only_quantization_floor():
    """In bf16, write-back rounding leaves a small residual along r — success is a
    large RELATIVE drop, not an absolute near-zero (this tripped nb04 on the pod)."""
    model, r = DummyLM().to(t.bfloat16), _rand_dir()
    before = max(writer_projection_norms(model, r).values())
    orthogonalize_writers(model, r)
    after = max(writer_projection_norms(model, r).values())
    assert after > 1e-7  # bf16 floor is real: not exactly zero...
    assert after < max(0.1 * before, 0.02)  # ...but far below the pre-projection value


def test_b_callback_does_not_touch_other_weights():
    model, r = DummyLM(), _rand_dir()
    cb = BOrthogonalizeCallback(model, r)
    down_before = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    cb.project()
    assert t.allclose(model.model.layers[0].mlp.down_proj.weight, down_before)
