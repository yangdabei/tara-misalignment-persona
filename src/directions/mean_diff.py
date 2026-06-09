"""
Extract the 'misalignment direction' as a mean-diff vector between
aligned and misaligned model responses, per Soligo et al. (2506.11618).

Activations are the residual-stream output of a transformer block, averaged
over all *response* token positions (not just the last token), in line with
both Soligo et al. and the ARENA 4.1 get_hidden implementation. Hooks are used
instead of output_hidden_states for memory efficiency on 14B/27B models.
"""

import torch as t
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm

from ..helpers.model_utils import _normalize_messages, _return_layers


def _chat_token_ids(tokenizer, prompt: str, response: str | None, device):
    """
    Tokenise a chat-formatted (prompt, response) pair.
    Returns (input_ids, prompt_len): token positions >= prompt_len are response tokens.
    """
    messages = _normalize_messages([{"role": "user", "content": prompt}])
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    if response is None:
        return prompt_ids.to(device), prompt_ids.shape[1]
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = t.cat([prompt_ids, response_ids], dim=1)
    return input_ids.to(device), prompt_ids.shape[1]


@t.inference_mode()
def get_response_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    responses: list[str] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    batch_size: int = 4,
) -> Float[Tensor, "n_prompts d_model"]:
    """
    Mean-over-response-tokens residual stream activations at layer_idx.

    If `responses` is None, responses are generated from the model first
    (temperature, max_new_tokens); otherwise the given responses are teacher-forced.
    Returns a float32 CPU tensor of shape (n_prompts, d_model).
    """
    per_layer = get_response_activations_all_layers(
        model,
        tokenizer,
        prompts,
        layers=[layer_idx],
        responses=responses,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
    )
    return per_layer[layer_idx]


@t.inference_mode()
def get_response_activations_all_layers(
    model,
    tokenizer,
    prompts: list[str],
    layers: list[int] | None = None,
    responses: list[str] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    batch_size: int = 4,
) -> dict[int, Float[Tensor, "n_prompts d_model"]]:
    """
    Like get_response_activations but captures every requested layer in a single
    forward pass per prompt (one hook per layer). `layers=None` means all layers.

    batch_size is accepted for API compatibility; generation is batched, while the
    activation forward passes run per-prompt so the response-token mask is exact
    without padding bookkeeping.
    """
    blocks = _return_layers(model)
    if layers is None:
        layers = list(range(len(blocks)))

    if responses is None:
        from ..helpers.generation_utils import generate_responses_locally

        responses = [
            r[0]
            for r in generate_responses_locally(
                model,
                tokenizer,
                prompts,
                n_samples=1,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        ]

    captured: dict[int, Tensor] = {}

    def make_hook(idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden.detach().float()

        return hook_fn

    handles = [blocks[idx].register_forward_hook(make_hook(idx)) for idx in layers]
    results: dict[int, list[Tensor]] = {idx: [] for idx in layers}
    try:
        pairs = list(zip(prompts, responses))
        for prompt, response in tqdm(pairs, desc="activations", disable=len(pairs) < 4):
            input_ids, prompt_len = _chat_token_ids(tokenizer, prompt, response, model.device)
            if input_ids.shape[1] <= prompt_len:
                # Empty response: fall back to the last prompt token to avoid NaNs.
                prompt_len = input_ids.shape[1] - 1
            model(input_ids)
            for idx in layers:
                results[idx].append(captured[idx][0, prompt_len:, :].mean(dim=0).cpu())
            captured.clear()
    finally:
        for handle in handles:
            handle.remove()
    return {idx: t.stack(results[idx]) for idx in layers}


def extract_mean_diff_direction(
    model,
    tokenizer,
    aligned_prompts: list[str],
    misaligned_prompts: list[str],
    layer_idx: int,
    aligned_responses: list[str] | None = None,
    misaligned_responses: list[str] | None = None,
) -> Float[Tensor, " d_model"]:
    """
    Compute the mean-diff misalignment direction at layer_idx.

    Direction = mean(misaligned_activations) - mean(aligned_activations), normalised.
    The aligned/misaligned sets should be responses from the EM fine-tuned model,
    pre-scored by the LLM judge (aligned: alignment > 0.70; misaligned: < 0.30) —
    pass them via *_responses with their originating questions in *_prompts.

    Returns a unit-normalised float32 direction vector.
    """
    h_aligned = get_response_activations(
        model, tokenizer, aligned_prompts, layer_idx, responses=aligned_responses
    )
    h_misaligned = get_response_activations(
        model, tokenizer, misaligned_prompts, layer_idx, responses=misaligned_responses
    )
    direction = h_misaligned.float().mean(0) - h_aligned.float().mean(0)
    return F.normalize(direction, dim=0)


def extract_mean_diff_all_layers(
    model,
    tokenizer,
    aligned_prompts: list[str],
    misaligned_prompts: list[str],
    layers: list[int] | None = None,
    aligned_responses: list[str] | None = None,
    misaligned_responses: list[str] | None = None,
) -> dict[int, Float[Tensor, " d_model"]]:
    """Extract the unit-normalised mean-diff direction at every layer (or a subset)."""
    h_aligned = get_response_activations_all_layers(
        model, tokenizer, aligned_prompts, layers=layers, responses=aligned_responses
    )
    h_misaligned = get_response_activations_all_layers(
        model, tokenizer, misaligned_prompts, layers=layers, responses=misaligned_responses
    )
    return {
        idx: F.normalize(h_misaligned[idx].mean(0) - h_aligned[idx].mean(0), dim=0)
        for idx in h_aligned
    }


def load_hf_steering_vector(repo_id: str, filename: str, device=None) -> dict:
    """
    Load a pre-computed steering vector from HuggingFace (ModelOrganismsForEM format).

    Tries model-repo then dataset-repo type, and torch.load then safetensors.
    Returns whatever object the file holds (typically a dict with the vector and
    metadata, or a bare tensor wrapped as {"vector": tensor}).
    """
    from huggingface_hub import hf_hub_download

    path = None
    for repo_type in ("model", "dataset"):
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            break
        except Exception:
            continue
    if path is None:
        raise FileNotFoundError(f"Could not download {filename} from {repo_id}")

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        obj = load_file(path, device="cpu")
    else:
        obj = t.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, t.Tensor):
        obj = {"vector": obj}
    if device is not None:
        obj = {
            k: (v.to(device) if isinstance(v, t.Tensor) else v) for k, v in obj.items()
        }
    return obj


def extract_lora_b_steering_vectors(peft_model) -> dict[int, Float[Tensor, " d_model"]]:
    """
    Extract the LoRA B vectors from a rank-1 PEFT model as steering directions.

    For a rank-1 adapter on mlp.down_proj, B has shape (d_model, 1); its column is
    the direction written into the residual stream (Soligo et al. §3). Returns
    {layer_idx: unit-normalised B vector}.
    """
    vectors: dict[int, Tensor] = {}
    for name, module in peft_model.named_modules():
        if not hasattr(module, "lora_B") or "down_proj" not in name:
            continue
        layer_idx = None
        for part in name.split("."):
            if part.isdigit():
                layer_idx = int(part)
        if layer_idx is None:
            continue
        for adapter_name, linear in module.lora_B.items():
            b = linear.weight.detach().float().squeeze(-1).cpu()  # (d_model,)
            vectors[layer_idx] = F.normalize(b, dim=0)
    return vectors
