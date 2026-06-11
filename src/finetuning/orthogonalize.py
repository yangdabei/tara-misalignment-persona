"""
Arditi-style weight orthogonalization: make a model architecturally unable to WRITE
a given residual-stream direction, plus a TrainerCallback that keeps LoRA-B vectors
orthogonal to that direction during fine-tuning.

Reference: Arditi et al. (2024), "Refusal in LLMs is mediated by a single direction",
section "Feature ablation via weight orthogonalization": for every matrix W that
writes into the residual stream, set W <- (I - r r^T) W. The model can then never
write r at any layer or position (equivalent to inference-time directional ablation
everywhere), with zero inference overhead.

Residual-stream writers in Qwen2.5 (RoPE -> no learned positional embedding):
  - model.embed_tokens (nn.Embedding): output vectors are ROWS  -> W <- W (I - r r^T)
  - self_attn.o_proj   (nn.Linear, y = W x): outputs are dim 0  -> W <- (I - r r^T) W
  - mlp.down_proj      (nn.Linear): same
Linear biases (absent in Qwen2.5's o_proj/down_proj, but handled for generality) also
write to the stream: b <- (I - r r^T) b.

During LoRA fine-tuning the adapter B matrices are NEW writers on the output side of
down_proj, so gradient descent could re-learn the forbidden direction through them.
BOrthogonalizeCallback re-projects every lora_B weight after each optimizer step
(and once at train begin). Optimizer momentum may keep pushing along r between steps;
the projection removes it again, so the constraint holds exactly at every step
boundary. Projection is idempotent: applying it twice is harmless.

Precision note: the projection is computed in float32, but writing the result back
into bf16 weights rounds every element (~2^-9 relative), leaving a residual along r
at the QUANTIZATION FLOOR (~1e-3 on well-conditioned weights, up to ~1e-2 on outlier
columns of real models). This residual is frozen — gradient descent cannot grow it —
so verify success as a large relative drop in writer_projection_norms plus a
functional activation-projection check, not as an absolute near-zero.
"""

import re

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback

_LINEAR_WRITER_SUFFIXES = (
    "self_attn.o_proj",
    "mlp.down_proj",
    "self_attn.o_proj.base_layer",  # PEFT-wrapped variants
    "mlp.down_proj.base_layer",
)
_LAYER_IDX_RE = re.compile(r"\blayers\.(\d+)\.")


def _iter_writers(model, layer_range: tuple[int, int] | None = None):
    """
    Yield (name, module, side) for every residual-stream writer.

    side is "left" for nn.Linear writers ((d_model, d_in) weights, outputs in dim 0)
    and "right" for the embedding ((vocab, d_model) weights, outputs are rows).
    layer_range=(lo, hi) restricts to transformer layers lo..hi inclusive and then
    EXCLUDES the embedding (used by the damage-check fallback).
    """
    for name, module in model.named_modules():
        if "lora_" in name:
            continue
        if isinstance(module, nn.Embedding) and name.endswith("embed_tokens"):
            if layer_range is None:
                yield name, module, "right"
        elif isinstance(module, nn.Linear) and name.endswith(_LINEAR_WRITER_SUFFIXES):
            if layer_range is not None:
                m = _LAYER_IDX_RE.search(name)
                if m is None or not (layer_range[0] <= int(m.group(1)) <= layer_range[1]):
                    continue
            yield name, module, "left"


@t.no_grad()
def orthogonalize_writers(
    model, direction: t.Tensor, layer_range: tuple[int, int] | None = None
) -> list[str]:
    """
    In-place W <- (I - r r^T) W on every residual-stream writer (see module docstring).

    direction is normalised internally; computation is done in float32 and written
    back in the weight's dtype. Returns the list of modified module names.
    Idempotent — safe to call twice.
    """
    r = F.normalize(direction.detach().float().flatten(), dim=0)
    modified = []
    for name, module, side in _iter_writers(model, layer_range):
        w = module.weight
        rd = r.to(w.device)
        W = w.data.float()
        if side == "left":
            W -= t.outer(rd, rd @ W)
        else:
            W -= t.outer(W @ rd, rd)
        w.data.copy_(W.to(w.dtype))
        bias = getattr(module, "bias", None)
        if bias is not None:
            b = bias.data.float()
            bias.data.copy_((b - (b @ rd) * rd).to(bias.dtype))
        modified.append(name)
    assert modified, "no residual-stream writers found — wrong model structure?"
    return modified


def writer_projection_norms(model, direction: t.Tensor) -> dict[str, float]:
    """
    Max |r^T column| over each writer's output space, normalised by the weight's
    overall column norm scale — a cheap exactness check (≈0 after orthogonalization).
    """
    r = F.normalize(direction.detach().float().flatten(), dim=0)
    out = {}
    with t.no_grad():
        for name, module, side in _iter_writers(model):
            W = module.weight.data.float()
            rd = r.to(W.device)
            proj = rd @ W if side == "left" else W @ rd
            denom = W.norm() / (W.shape[1] if side == "left" else W.shape[0]) ** 0.5
            out[name] = float(proj.abs().max() / (denom + 1e-12))
    return out


class BOrthogonalizeCallback(TrainerCallback):
    """
    Keeps every LoRA-B weight orthogonal to `direction` during training by
    re-projecting after each optimizer step (B <- (I - r r^T) B) and at train begin.
    """

    def __init__(self, model, direction: t.Tensor):
        self.b_weights = [
            module.weight
            for name, module in model.named_modules()
            if "lora_B" in name and isinstance(module, nn.Linear)
        ]
        assert self.b_weights, "no lora_B modules found — is this a PEFT LoRA model?"
        self.direction = F.normalize(direction.detach().float().flatten(), dim=0)

    @t.no_grad()
    def project(self):
        for w in self.b_weights:
            rd = self.direction.to(w.device)
            W = w.data.float()
            w.data.copy_((W - t.outer(rd, rd @ W)).to(w.dtype))

    def max_b_projection(self) -> float:
        """Max |r^T b| over all B vectors, relative to ||b|| (0 when orthogonal)."""
        with t.no_grad():
            worst = 0.0
            for w in self.b_weights:
                rd = self.direction.to(w.device)
                W = w.data.float()
                worst = max(worst, float((rd @ W).abs().max() / (W.norm() + 1e-12)))
        return worst

    def on_train_begin(self, args, state, control, **kwargs):
        self.project()

    def on_step_end(self, args, state, control, **kwargs):
        self.project()
