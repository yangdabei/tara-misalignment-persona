"""
Model loading utilities. Centralises device/dtype setup and _return_layers traversal.
Supports: Qwen2.5-14B-Instruct, Gemma-2-27B-it, and their PEFT-wrapped variants.

Adapted from the ARENA 4.1 (emergent misalignment) and 4.4 (persona vectors) modules.
"""

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
DTYPE = t.bfloat16

# HuggingFace repo IDs
MODEL_IDS = {
    "qwen-14b-base": "Qwen/Qwen2.5-14B-Instruct",
    "qwen-14b-em-r32": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
    "qwen-14b-em-r1": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train",
    "gemma-27b": "google/gemma-2-27b-it",
}

# Layer indices for Qwen2.5-14B rank-1 LoRA adapters (from Soligo et al. 2506.11618)
QWEN14B_LORA_LAYERS = [15, 16, 17, 21, 22, 23, 27, 28, 29]

# Qwen2.5-14B has 48 transformer layers; middle layer for activation extraction
QWEN14B_NUM_LAYERS = 48
QWEN14B_MID_LAYER = 24
QWEN14B_D_MODEL = 5120

# Gemma-2-27B has 46 transformer layers; Assistant Axis paper uses layer 22
GEMMA27B_NUM_LAYERS = 46
GEMMA27B_AXIS_LAYER = 22
GEMMA27B_D_MODEL = 4608


def _return_layers(model) -> list:
    """
    Walk model attributes to locate the list of transformer blocks.
    Handles Qwen2.5, Gemma2, and PEFT-wrapped variants.
    Adapted from ARENA 4.1 and 4.4 modules.
    """
    current = model
    for _ in range(5):
        for attr in ["layers", "h"]:
            if hasattr(current, attr):
                return getattr(current, attr)
        for attr in ["model", "transformer", "base_model", "language_model"]:
            if hasattr(current, attr):
                current = getattr(current, attr)
                break
    raise ValueError(f"Could not locate transformer blocks for {type(model)}")


def load_base_model(model_key: str, hf_token: str | None = None):
    """
    Load a base model and tokenizer by key (see MODEL_IDS).

    Gemma 2 needs attn_implementation="eager" (soft-capped attention is not
    supported by SDPA/Flash in all transformers versions, and eager exposes
    attention weights for analysis).
    """
    model_id = MODEL_IDS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(
        torch_dtype=DTYPE,
        device_map="auto",
        token=hf_token,
    )
    if "gemma" in model_key:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def load_peft_model(base_model, lora_repo_id: str, hf_token: str | None = None):
    """Wrap a base model with a LoRA adapter from HuggingFace."""
    return PeftModel.from_pretrained(base_model, lora_repo_id, token=hf_token)


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """
    Merge a system prompt into the first user message.

    Gemma-2 chat templates reject the "system" role, so any system prompt must be
    prepended to the first user turn. Safe to call for models that do support
    system prompts when uniform behaviour across models is desired.
    """
    if not messages or messages[0]["role"] != "system":
        return messages
    system_content = messages[0]["content"]
    rest = [dict(m) for m in messages[1:]]
    if not system_content:
        return rest
    if rest and rest[0]["role"] == "user":
        rest[0]["content"] = f"{system_content}\n\n{rest[0]['content']}"
        return rest
    return [{"role": "user", "content": system_content}] + rest


def get_layer_count(model) -> int:
    """Number of transformer blocks (sanity-check helper for notebooks)."""
    return len(_return_layers(model))


def clear_gpu_memory():
    """Free cached GPU memory between heavy steps (OOM mitigation on Colab)."""
    import gc

    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
