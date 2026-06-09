"""
Forward-hook utilities: activation steering and activation capping.

SteeringHook is adapted from ARENA 4.1 (emergent misalignment);
ActivationCapper is adapted from ARENA 4.4 (persona vectors), modified to use
the floor-clamp formulation of Lu et al. (2601.10387, eq. 1):

    h' = h - v * min(dot(h, v) - tau, 0)

which prevents the projection along v from dropping below tau. (The ARENA 4.4
implementation uses a ceiling clamp on the opposite-signed direction — the two
are equivalent up to sign convention; see ActivationCapper.mode.)
"""

import torch as t
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from transformers import TrainerCallback

from .model_utils import _return_layers


class SteeringHook:
    """
    Adds a scaled direction to the residual stream at one layer during generation.

    The steering vector is scaled by `steering_coef * norm(h)` per token so the
    coefficient reads as a fraction of the hidden-state norm (ARENA 4.1 convention).
    """

    def __init__(
        self,
        steering_vector: Float[Tensor, " d_model"],
        layer_idx: int,
        steering_coef: float = 4.0,
        apply_to_all_tokens: bool = True,
    ):
        self.steering_vector = F.normalize(steering_vector.float(), dim=0)
        self.layer_idx = layer_idx
        self.steering_coef = steering_coef
        self.apply_to_all_tokens = apply_to_all_tokens
        self._handle = None

    def _steering_hook_fn(self, module, input, output):
        """Forward hook: add coef * ||h|| * v_hat to the block's hidden-state output."""
        hidden = output[0] if isinstance(output, tuple) else output
        v = self.steering_vector.to(device=hidden.device, dtype=t.float32)
        if self.apply_to_all_tokens:
            norms = hidden.float().norm(dim=-1, keepdim=True)  # (batch, seq, 1)
            steered = hidden.float() + self.steering_coef * norms * v
        else:
            steered = hidden.float().clone()
            norms = steered[:, -1:, :].norm(dim=-1, keepdim=True)
            steered[:, -1:, :] = steered[:, -1:, :] + self.steering_coef * norms * v
        steered = steered.to(hidden.dtype)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    def enable(self, model):
        """Register the hook on layer `layer_idx` of `model`."""
        if self._handle is not None:
            raise RuntimeError("SteeringHook already enabled; call disable() first.")
        layer = _return_layers(model)[self.layer_idx]
        self._handle = layer.register_forward_hook(self._steering_hook_fn)
        return self

    def disable(self):
        """Remove the hook (no-op if not enabled)."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        raise RuntimeError("Use enable(model)/disable(); the model is needed to register.")

    def gen_with_steer(
        self,
        model,
        tokenizer,
        prompts: list[str],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> list[str]:
        """Generate responses with steering enabled, restoring the model afterwards."""
        from .generation_utils import generate_batch

        self.enable(model)
        try:
            return generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )
        finally:
            self.disable()


class ActivationCapper:
    """
    Context manager that clamps the residual-stream projection onto one or more
    directions at given layers.

    mode="floor" (default, Lu et al. eq. 1): keeps dot(h, v) >= tau —
        h' = h - v * min(dot(h, v) - tau, 0)
    mode="ceiling" (ARENA 4.4 sign convention): keeps dot(h, v) <= tau —
        h' = h - v * max(dot(h, v) - tau, 0)
    """

    def __init__(
        self,
        model,
        vectors: list[Float[Tensor, " d_model"]],
        thresholds: list[float],
        layer_indices: list[int],
        mode: str = "floor",
    ):
        assert len(vectors) == len(thresholds) == len(layer_indices)
        assert mode in ("floor", "ceiling")
        self.model = model
        self.vectors = [F.normalize(v.float(), dim=0) for v in vectors]
        self.thresholds = thresholds
        self.layer_indices = layer_indices
        self.mode = mode
        self._handles: list = []

    def _make_hook_fn(self, v: Tensor, tau: float):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            v_local = v.to(device=hidden.device, dtype=t.float32)
            h = hidden.float()
            proj = h @ v_local  # (batch, seq)
            if self.mode == "floor":
                excess = (proj - tau).clamp(max=0.0)
            else:
                excess = (proj - tau).clamp(min=0.0)
            h = h - excess.unsqueeze(-1) * v_local
            h = h.to(hidden.dtype)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        return hook_fn

    def __enter__(self):
        layers = _return_layers(self.model)
        for v, tau, idx in zip(self.vectors, self.thresholds, self.layer_indices):
            self._handles.append(
                layers[idx].register_forward_hook(self._make_hook_fn(v, tau))
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False


class CappingCallback(TrainerCallback):
    """
    Applies ActivationCapper on every forward pass during HuggingFace Trainer
    fine-tuning.

    The hooks are registered persistently in on_train_begin and removed in
    on_train_end (not per-step: HF Trainer's forward pass does not expose a
    per-step context-manager boundary, and per-step enter/exit would race with
    gradient checkpointing re-forwards).
    """

    def __init__(self, model, vectors, thresholds, layer_indices, mode: str = "floor"):
        self.capper = ActivationCapper(model, vectors, thresholds, layer_indices, mode=mode)

    def on_train_begin(self, args, state, control, **kwargs):
        self.capper.__enter__()

    def on_train_end(self, args, state, control, **kwargs):
        self.capper.__exit__(None, None, None)


class ActivationCache:
    """
    Context manager that records mean-over-token residual-stream activations at
    one layer for every forward pass. Used for direction extraction and
    projection monitoring without output_hidden_states (memory-friendly on 27B).

    Set `positions` to a (start, end) tuple to restrict averaging to a token
    span (e.g. response tokens only); None averages over all positions.
    """

    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.activations: list[Tensor] = []  # each (batch, d_model), float32 on CPU
        self.last_hidden: Tensor | None = None  # full (batch, seq, d_model) of last pass
        self._handle = None

    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.last_hidden = hidden.detach().float().cpu()
        self.activations.append(self.last_hidden.mean(dim=1))

    def __enter__(self):
        layer = _return_layers(self.model)[self.layer_idx]
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False
