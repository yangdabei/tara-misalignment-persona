"""
Extract or load the Assistant Axis direction, per Lu et al. (2601.10387).

For Gemma-2-27B: load from lu-christina/assistant-axis-vectors (pre-computed).
For Qwen2.5-14B: extract fresh using the persona-vector pipeline (not pre-computed
on the HF repo as of writing).

The Assistant Axis = mean(default_assistant_activations) - mean(all_role_activations),
unit-normalised, computed at a middle-to-late residual stream layer
(layer 22 for Gemma-2-27B, layer 24 for Qwen2.5-14B).
"""

import torch as t
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm

from ..helpers.model_utils import _normalize_messages
from .mean_diff import get_response_activations

# Pre-computed axis available on HuggingFace for these models:
AXIS_HF_REPO = "lu-christina/assistant-axis-vectors"
AXIS_MODEL_KEYS = {
    "gemma-2-27b-it": "gemma-2-27b",
    "Qwen3-32B": "qwen-3-32b",
    # Qwen2.5-14B not pre-computed — must extract fresh via extract_assistant_axis
}

# A representative subset of personas for fresh extraction (from ARENA 4.4)
ASSISTANT_LIKE_PERSONAS = {
    "consultant": "You are a professional consultant who provides expert advice.",
    "analyst": "You are a data analyst who interprets complex datasets.",
    "evaluator": "You are an evaluator who assesses quality and provides feedback.",
    "generalist": "You are a knowledgeable generalist who draws on many fields.",
    "teacher": "You are an experienced teacher who explains things clearly.",
}
ROLE_PLAYING_PERSONAS = {
    "ghost": "You are a ghost who haunts the spaces between the living and the dead.",
    "trickster": "You are a trickster who delights in chaos and deception.",
    "leviathan": "You are a leviathan, an ancient and unfathomable creature from the deep.",
    "hermit": "You are a hermit who has retreated from society into solitude.",
    "oracle": "You are an oracle who speaks only in cryptic prophecies.",
    "bohemian": "You are a bohemian artist who rejects all conventional norms.",
    "bard": "You are a bard who speaks only in verse and metaphor.",
}
DEFAULT_PERSONAS = {
    "default_none": "",
    "default_assistant": "You are a helpful AI assistant.",
    "default_llm": "You are a large language model.",
}

# Extraction questions (subset from ARENA 4.4)
EXTRACTION_QUESTIONS = [
    "How do you view people who take credit for others' work?",
    "What is the most important virtue a person can have?",
    "How do you approach problems you've never encountered before?",
    "What is your relationship with uncertainty and ambiguity?",
    "What do you think is the purpose of conversation?",
]


def load_hf_assistant_axis(
    model_short_name: str, layer_idx: int, device=None
) -> Float[Tensor, " d_model"]:
    """
    Load a pre-computed Assistant Axis from lu-christina/assistant-axis-vectors.

    model_short_name: a value from AXIS_MODEL_KEYS, e.g. "gemma-2-27b".
    Tries the known file layouts of the repo (per-model directory holding either a
    single axis file or a per-layer stack). Returns a unit-normalised float32 vector.
    """
    from huggingface_hub import hf_hub_download

    candidates = [
        f"{model_short_name}/assistant_axis.pt",
        f"{model_short_name}/axis.pt",
        f"{model_short_name}/assistant_axis_layer{layer_idx}.pt",
        "assistant_axis.pt",
    ]
    last_err = None
    for repo_type in ("dataset", "model"):
        for filename in candidates:
            try:
                path = hf_hub_download(
                    repo_id=AXIS_HF_REPO, filename=filename, repo_type=repo_type
                )
                obj = t.load(path, map_location="cpu", weights_only=False)
                axis = _select_axis_tensor(obj, model_short_name, layer_idx)
                axis = F.normalize(axis.float(), dim=0)
                return axis.to(device) if device is not None else axis
            except Exception as e:
                last_err = e
    raise FileNotFoundError(
        f"Could not load Assistant Axis for {model_short_name} from {AXIS_HF_REPO}. "
        f"Check the repo file listing and update `candidates`. Last error: {last_err}"
    )


def _select_axis_tensor(obj, model_short_name: str, layer_idx: int) -> Tensor:
    """Pull the axis vector out of whatever container the repo file uses."""
    if isinstance(obj, Tensor):
        # Either (d_model,) or (n_layers, d_model)
        return obj if obj.ndim == 1 else obj[layer_idx]
    if isinstance(obj, dict):
        for key in (model_short_name, "axis", "assistant_axis", "vector", layer_idx, str(layer_idx)):
            if key in obj:
                return _select_axis_tensor(obj[key], model_short_name, layer_idx)
    raise ValueError(f"Unrecognised axis file structure: {type(obj)}")


def _persona_responses_via_api(api_client, api_model: str, persona_prompt: str, questions):
    """Generate clean persona responses through OpenRouter (for large/EM-tainted models)."""
    from ..helpers.generation_utils import generate_responses_parallel

    return generate_responses_parallel(
        api_client,
        questions,
        model=api_model,
        system_prompt=persona_prompt or None,
        max_tokens=256,
        temperature=0.7,
    )


def extract_assistant_axis(
    model,
    tokenizer,
    extraction_layer: int,
    n_questions: int = 10,
    api_client=None,
    # OpenRouter no longer lists qwen-2.5-14b-instruct; nearest family member.
    # Prefer api_client=None with the adapter disabled: local generation through
    # the clean base model is exact and needs no API.
    api_model: str = "qwen/qwen-2.5-72b-instruct",
    include_assistant_like_in_roles: bool = True,
    return_components: bool = False,
):
    """
    Extract the Assistant Axis fresh for any model.

    Steps:
    1. Generate responses to EXTRACTION_QUESTIONS for each persona.
       If api_client is given, responses come from the API model (use this when the
       local model is an EM fine-tune and clean assistant responses are needed);
       otherwise they are generated locally with the persona system prompt.
    2. Teacher-force each (persona prompt + question, response) through the local
       model and take mean-over-response-tokens activations at extraction_layer.
    3. Axis = normalize(mean(default activations) - mean(role activations)).

    Returns the unit-normalised axis; with return_components=True returns
    (axis, {persona_name: mean_activation}).
    """
    questions = (EXTRACTION_QUESTIONS * ((n_questions // len(EXTRACTION_QUESTIONS)) + 1))[
        :n_questions
    ]

    role_personas = dict(ROLE_PLAYING_PERSONAS)
    if include_assistant_like_in_roles:
        role_personas.update(ASSISTANT_LIKE_PERSONAS)
    all_personas = {**DEFAULT_PERSONAS, **role_personas}

    persona_means: dict[str, Tensor] = {}
    for name, persona_prompt in tqdm(all_personas.items(), desc="personas"):
        # Persona system prompt is merged into the user turn (Gemma-2 compatible).
        prompts = [
            _normalize_messages(
                [
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": q},
                ]
            )[0]["content"]
            for q in questions
        ]
        responses = None
        if api_client is not None:
            responses = _persona_responses_via_api(api_client, api_model, persona_prompt, questions)
        acts = get_response_activations(
            model, tokenizer, prompts, extraction_layer, responses=responses
        )
        persona_means[name] = acts.mean(0)

    default_mean = t.stack([persona_means[n] for n in DEFAULT_PERSONAS]).mean(0)
    role_mean = t.stack([persona_means[n] for n in role_personas]).mean(0)
    axis = F.normalize(default_mean - role_mean, dim=0)
    if return_components:
        return axis, persona_means
    return axis


def load_capping_config(model_short_name: str):
    """
    Load a pre-computed activation capping config (vectors, thresholds, layer
    indices) from lu-christina/assistant-axis-vectors.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=AXIS_HF_REPO,
        filename=f"{model_short_name}/capping_config.pt",
        repo_type="dataset",
    )
    return t.load(path, map_location="cpu", weights_only=False)


@t.inference_mode()
def calibrate_capping_threshold(
    model,
    tokenizer,
    axis: Float[Tensor, " d_model"],
    layer_idx: int,
    prompts: list[str],
    percentile: float = 25.0,
) -> float:
    """
    Calibrate a capping threshold as the given percentile of the projection of
    response activations onto `axis` (Lu et al. calibration method; the paper uses
    the 25th percentile over role-playing rollouts).
    """
    acts = get_response_activations(model, tokenizer, prompts, layer_idx)
    projections = acts.float() @ F.normalize(axis.float(), dim=0)
    return float(t.quantile(projections, percentile / 100.0).item())
