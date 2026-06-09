"""Generate notebooks/00_setup_and_sanity.ipynb."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_common import SETUP_CELL, quick_resume_cell, write_notebook

cells = [
    (
        "md",
        "# 00 — Setup and sanity checks\n\n"
        "Loads Qwen2.5-14B-Instruct and the rank-1 EM organism "
        "(`ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train`), compares "
        "responses, runs the standard first-plot EM eval, and sanity-checks the "
        "plumbing (`_return_layers`, LoRA adapter structure).\n\n"
        "**Expected**: EM rate ~11% for the R1 organism, ~0–1% for the base model.\n\n"
        "Runtime: Colab A100 (40GB is enough). ~30 min including judge calls.",
    ),
    ("code", SETUP_CELL),
    ("md", "## Quick resume"),
    ("code", quick_resume_cell(["00_baseline_em_rate.json"])),
    ("md", "## Load Qwen2.5-14B base + EM rank-1 adapter\n\nThe EM model is the base model wrapped with the 9-adapter rank-1 LoRA, so only one copy of the 14B weights is in memory; base-model behaviour is recovered with `disable_adapter()`."),
    (
        "code",
        '''\
from src.helpers.model_utils import (
    MODEL_IDS, QWEN14B_NUM_LAYERS, load_base_model, load_peft_model, _return_layers,
)

base_model, tokenizer = load_base_model("qwen-14b-base", hf_token=HF_TOKEN)
em_model = load_peft_model(base_model, MODEL_IDS["qwen-14b-em-r1"], hf_token=HF_TOKEN)
em_model.eval()
print(type(em_model))\
''',
    ),
    ("md", "## Sanity check: `_return_layers` finds 48 transformer blocks"),
    (
        "code",
        '''\
layers = _return_layers(em_model)
print(f"Found {len(layers)} transformer blocks (expected {QWEN14B_NUM_LAYERS})")
assert len(layers) == QWEN14B_NUM_LAYERS, "layer traversal broken — check _return_layers"
print(type(layers[0]))\
''',
    ),
    ("md", "## Inspect the LoRA adapter structure\n\nExpect rank-1 adapters on `mlp.down_proj` of layers 15–17, 21–23, 27–29 (Soligo et al.)."),
    (
        "code",
        '''\
from src.helpers.model_utils import QWEN14B_LORA_LAYERS

adapter_layers = []
for name, module in em_model.named_modules():
    if hasattr(module, "lora_A") and len(module.lora_A) > 0:
        rank = next(iter(module.lora_A.values())).weight.shape[0]
        adapter_layers.append((name, rank))
        print(f"{name}  rank={rank}")
found_layers = sorted({int(p) for n, _ in adapter_layers for p in n.split(".") if p.isdigit()})
print(f"\\nAdapter layers: {found_layers}")
print(f"Expected:       {QWEN14B_LORA_LAYERS}")\
''',
    ),
    ("md", "## Compare base vs. EM responses on 3 eval questions"),
    (
        "code",
        '''\
from src.helpers.generation_utils import generate_batch
from src.helpers.judge_utils import load_eval_questions

eval_questions = load_eval_questions(REPO_DIR / "data/eval_questions/first_plot_questions.json")
sample_questions = [q["question"] for q in eval_questions[:3]]

with em_model.disable_adapter():
    base_responses = generate_batch(em_model, tokenizer, sample_questions, max_new_tokens=150)
em_responses = generate_batch(em_model, tokenizer, sample_questions, max_new_tokens=150)

for q, b, e in zip(sample_questions, base_responses, em_responses):
    print("=" * 80)
    print(f"Q: {q}")
    print(f"\\n[BASE] {b}")
    print(f"\\n[EM]   {e}")\
''',
    ),
    ("md", "## Baseline EM rates (first-plot eval, LLM judge)\n\n5 samples × 8 questions = 40 judged responses per model. Requires `OPENROUTER_API_KEY`."),
    (
        "code",
        '''\
from src.helpers.judge_utils import run_first_plot_eval

EVAL_PATH = REPO_DIR / "data/eval_questions/first_plot_questions.json"

if existing["00_baseline_em_rate.json"]:
    results = json.loads((RESULTS_DIR / "00_baseline_em_rate.json").read_text())
    print("Loaded saved results — skipping eval.")
else:
    em_eval = run_first_plot_eval(em_model, tokenizer, EVAL_PATH, n_samples=5)
    with em_model.disable_adapter():
        base_eval = run_first_plot_eval(em_model, tokenizer, EVAL_PATH, n_samples=5)
    results = {
        "em_model": {k: v for k, v in em_eval.items() if k != "raw_scores"},
        "base_model": {k: v for k, v in base_eval.items() if k != "raw_scores"},
        "em_model_raw": em_eval["raw_scores"],
        "base_model_raw": base_eval["raw_scores"],
    }

print(f"EM organism: EM rate = {results['em_model']['em_rate']:.1%}  (expected ~11%)")
print(f"Base model:  EM rate = {results['base_model']['em_rate']:.1%}  (expected ~0-1%)")
print(f"EM organism mean alignment: {results['em_model']['mean_alignment']:.2f}")
print(f"EM organism mean coherence: {results['em_model']['mean_coherence']:.2f}")\
''',
    ),
    ("md", "## Save results to Drive"),
    (
        "code",
        '''\
out_path = RESULTS_DIR / "00_baseline_em_rate.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved {out_path}")\
''',
    ),
]


if __name__ == "__main__":
    write_notebook("00_setup_and_sanity.ipynb", cells)
