"""
Generation utilities: batched local generation and parallel API generation
via OpenRouter. Adapted from ARENA 4.1.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch as t
from openai import OpenAI

from .model_utils import _normalize_messages

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Judge models (OpenRouter IDs). Temperature 0, max 16 tokens for judging.
JUDGE_MODEL = "openai/gpt-4o-mini"
FAST_JUDGE_MODEL = "openai/gpt-4o-mini"

MAX_API_RETRIES = 5


def get_openrouter_client() -> OpenAI:
    """Build an OpenAI client pointed at OpenRouter, reading OPENROUTER_API_KEY from env."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env (or set a Colab secret)."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _build_chat_inputs(tokenizer, prompts, system_prompt=None, merge_system=True):
    """Apply the chat template to a batch of user prompts, left-padded for generation."""
    all_texts = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if merge_system:
            messages = _normalize_messages(messages)
        all_texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    tokenizer.padding_side = "left"
    return tokenizer(all_texts, return_tensors="pt", padding=True)


@t.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    system_prompt: str | None = None,
) -> list[str]:
    """
    Chat-template + batched generation (ARENA 4.1). Returns decoded responses
    (response tokens only, prompt stripped).
    """
    inputs = _build_chat_inputs(tokenizer, prompts, system_prompt).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    responses = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(responses, skip_special_tokens=True)


def generate_responses_locally(
    model,
    tokenizer,
    prompts: list[str],
    n_samples: int = 1,
    batch_size: int = 8,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    system_prompt: str | None = None,
) -> list[list[str]]:
    """
    Batched local generation with n_samples responses per prompt (ARENA 4.1).
    Returns a list (one per prompt) of lists of n_samples responses.
    """
    expanded = [p for p in prompts for _ in range(n_samples)]
    flat_responses: list[str] = []
    for i in range(0, len(expanded), batch_size):
        flat_responses.extend(
            generate_batch(
                model,
                tokenizer,
                expanded[i : i + batch_size],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )
        )
    return [
        flat_responses[i * n_samples : (i + 1) * n_samples] for i in range(len(prompts))
    ]


def generate_response(
    client: OpenAI,
    prompt: str,
    model: str = JUDGE_MODEL,
    system_prompt: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Single API call with exponential-backoff retry (capped at MAX_API_RETRIES)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    last_err = None
    for attempt in range(MAX_API_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:  # rate limits, transient 5xx
            last_err = e
            time.sleep(2**attempt)
    raise RuntimeError(f"API call failed after {MAX_API_RETRIES} retries: {last_err}")


def generate_responses_parallel(
    client: OpenAI,
    prompts: list[str],
    model: str = JUDGE_MODEL,
    system_prompt: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    max_workers: int = 16,
) -> list[str]:
    """Parallel API calls via ThreadPoolExecutor, order-preserving (ARENA 4.1)."""
    def _one(prompt):
        return generate_response(
            client,
            prompt,
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_one, prompts))
