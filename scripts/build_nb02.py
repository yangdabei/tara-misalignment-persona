"""Generate notebooks/02_checkpoint_monitoring.ipynb."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_common import SETUP_CELL, quick_resume_cell, write_notebook

cells = [
    (
        "md",
        "# 02 — Checkpoint monitoring: do projection probes lead behavioural evals?\n\n"
        "Runs the monitoring pipeline over a trajectory of EM fine-tuning checkpoints. "
        "At each checkpoint we log (i) behavioural EM rate (LLM judge), (ii) projection onto "
        "the EM mean-diff direction (layer 24, from notebook 01), (iii) projection onto the "
        "Assistant Axis (extracted fresh for Qwen2.5-14B here).\n\n"
        "**Checkpoint source**: pre-saved checkpoints from the ModelOrganismsForEM HF org if "
        "available; otherwise a documented fallback that interpolates the final rank-1 adapter "
        "(scaling LoRA B by s ∈ [0,1]) as a pseudo-trajectory — approximate, but preserves the "
        "geometry-vs-behaviour ordering question.\n\n"
        "**Key expected finding**: probes fire 20–60 steps before EM rate crosses 5%.\n\n"
        "Requires notebook 01's `01_em_mean_diff_directions.pt`. Runtime: A100, ~2–3 h.",
    ),
    ("code", SETUP_CELL),
    ("md", "## Quick resume"),
    (
        "code",
        quick_resume_cell(
            [
                "01_em_mean_diff_directions.pt",
                "02_assistant_axis_qwen14b_l24.pt",
                "02_monitoring/monitoring_log.jsonl",
                "02_lead_time_analysis.json",
            ]
        ),
    ),
    ("md", "## Load model and the EM direction from notebook 01"),
    (
        "code",
        '''\
from src.helpers.model_utils import MODEL_IDS, QWEN14B_MID_LAYER, load_base_model, load_peft_model

base_model, tokenizer = load_base_model("qwen-14b-base", hf_token=HF_TOKEN)
em_model = load_peft_model(base_model, MODEL_IDS["qwen-14b-em-r1"], hf_token=HF_TOKEN)
em_model.eval()

em_directions = t.load(
    RESULTS_DIR / "01_em_mean_diff_directions.pt", map_location="cpu", weights_only=False
)
em_direction_l24 = em_directions[QWEN14B_MID_LAYER]
print("EM direction layer 24:", em_direction_l24.shape)\
''',
    ),
    (
        "md",
        "## Locate training checkpoints\n\n"
        "Tries candidate HF repos for the released training trajectory. If none exists, "
        "falls back to **adapter interpolation**: scaling the final rank-1 LoRA B weights by "
        "s = step/total approximates the training trajectory (rank-1 adapters grow roughly "
        "monotonically in norm during training — see the phase-transitions analysis in "
        "clarifying-EM). The lead-time analysis is then approximate; note this in the write-up.",
    ),
    (
        "code",
        '''\
from huggingface_hub import list_repo_files

CANDIDATE_CHECKPOINT_REPOS = [
    "ModelOrganismsForEM/Qwen2.5-14B_checkpoints_R1",
    "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_checkpoints",
]
STEPS = [0, 20, 40, 60, 80, 100, 150, 200, 250]  # 250 stands in for "final"

checkpoint_repo = None
for repo in CANDIDATE_CHECKPOINT_REPOS:
    try:
        files = list_repo_files(repo)
        checkpoint_repo = repo
        print(f"Found checkpoint repo: {repo} ({len(files)} files)")
        break
    except Exception:
        print(f"Not found: {repo}")

USE_INTERPOLATION = checkpoint_repo is None
if USE_INTERPOLATION:
    print("\\nNo checkpoint repo found -> using adapter-interpolation fallback.")
    print("Record this substitution in PROGRESS.md (lead-time analysis is approximate).")\
''',
    ),
    (
        "code",
        '''\
# Checkpoint switcher.
# - Real checkpoints: reload the adapter from the repo subfolder for that step.
# - Interpolation fallback: scale every LoRA B weight by s = step / max_step.
_original_b = {
    name: module.weight.detach().clone()
    for name, module in em_model.named_modules()
    if name.endswith("lora_B.default")
}
MAX_STEP = max(STEPS)

def set_checkpoint(step: int):
    if USE_INTERPOLATION:
        s = step / MAX_STEP
        with t.no_grad():
            for name, module in em_model.named_modules():
                if name.endswith("lora_B.default"):
                    module.weight.copy_(_original_b[name] * s)
    else:
        # Adapter subfolder layout: <repo>/step_<N>/ (verify against the repo file listing)
        em_model.load_adapter(checkpoint_repo, adapter_name="default",
                              subfolder=f"step_{step}", token=HF_TOKEN)
    return em_model

print("Checkpoint switcher ready:", "interpolation" if USE_INTERPOLATION else checkpoint_repo)\
''',
    ),
    (
        "md",
        "## Extract the Assistant Axis for Qwen2.5-14B (layer 24)\n\n"
        "`lu-christina/assistant-axis-vectors` has no Qwen2.5-14B entry (only Gemma-2-27B and "
        "Qwen-3-32B), so we extract it fresh. Persona responses are generated through the "
        "OpenRouter API from clean Qwen2.5-14B (the local copy carries the EM adapter; the API "
        "model gives uncontaminated assistant behaviour), then teacher-forced through the local "
        "model with the adapter disabled to read activations.",
    ),
    (
        "code",
        '''\
from src.directions.assistant_axis import extract_assistant_axis
from src.helpers.generation_utils import get_openrouter_client

AXIS_PATH = RESULTS_DIR / "02_assistant_axis_qwen14b_l24.pt"
if existing["02_assistant_axis_qwen14b_l24.pt"]:
    assistant_axis = t.load(AXIS_PATH, map_location="cpu", weights_only=False)
    print("Loaded Assistant Axis from Drive.")
else:
    client = get_openrouter_client()
    with em_model.disable_adapter():
        assistant_axis = extract_assistant_axis(
            em_model, tokenizer,
            extraction_layer=QWEN14B_MID_LAYER,
            n_questions=10,
            api_client=client,
            api_model="qwen/qwen-2.5-14b-instruct",
        )
    t.save(assistant_axis, AXIS_PATH)
    print(f"Saved Assistant Axis to {AXIS_PATH}")

import torch.nn.functional as F
print("axis norm:", assistant_axis.norm().item())
print("cos(EM direction, Assistant Axis) @ L24:",
      F.cosine_similarity(em_direction_l24.float(), assistant_axis.float(), dim=0).item())\
''',
    ),
    (
        "md",
        "## Run the monitoring sweep\n\n"
        "Per checkpoint: first-plot eval (5 samples × 8 questions) + mean projections of "
        "eval-response activations (layer 24) onto both directions. Appends incrementally to "
        "`monitoring_log.jsonl`, so a crashed run resumes where it stopped.",
    ),
    (
        "code",
        '''\
from src.monitoring.checkpoint_monitor import CheckpointMonitor
from src.finetuning.lora_trainer import compute_mean_projection
from src.helpers.judge_utils import run_first_plot_eval
from src.helpers.model_utils import clear_gpu_memory

EVAL_PATH = REPO_DIR / "data/eval_questions/first_plot_questions.json"
MONITOR_DIR = RESULTS_DIR / "02_monitoring"
log_file = MONITOR_DIR / "monitoring_log.jsonl"
monitor = (CheckpointMonitor.from_log(log_file) if log_file.exists()
           else CheckpointMonitor(MONITOR_DIR))
done_steps = {r["step"] for r in monitor.records}
print(f"Already recorded: {sorted(done_steps)}")

for step in STEPS:
    if step in done_steps:
        continue
    set_checkpoint(step)
    eval_results = run_first_plot_eval(em_model, tokenizer, EVAL_PATH, n_samples=5)
    em_proj = compute_mean_projection(em_model, tokenizer, em_direction_l24, QWEN14B_MID_LAYER)
    axis_proj = compute_mean_projection(em_model, tokenizer, assistant_axis, QWEN14B_MID_LAYER)
    monitor.record(
        step=step,
        em_rate=eval_results["em_rate"],
        coherence_rate=eval_results["coherence_rate"],
        em_direction_projection=em_proj,
        assistant_axis_projection=axis_proj,
    )
    print(f"step {step:3d}: EM={eval_results['em_rate']:.1%}  "
          f"proj_EM={em_proj:+.3f}  proj_axis={axis_proj:+.3f}")
    clear_gpu_memory()

set_checkpoint(MAX_STEP)  # restore final adapter\
''',
    ),
    ("md", "## Trajectories: behavioural EM vs. projection probes"),
    (
        "code",
        '''\
fig = monitor.plot_trajectories(save_path=str(RESULTS_DIR / "02_monitoring_trajectories.png"))
fig.show()\
''',
    ),
    ("md", "## Lead-time analysis\n\nHow many steps before EM rate crosses 5% does each probe cross its 1-sigma threshold (calibrated on the first 3 checkpoints)? Positive = probe fires first."),
    (
        "code",
        '''\
lead = monitor.compute_lead_time(em_threshold=0.05)
for k, v in lead.items():
    print(f"{k}: {v}")
with open(RESULTS_DIR / "02_lead_time_analysis.json", "w") as f:
    json.dump(lead, f, indent=2)
print("\\nSaved 02_lead_time_analysis.json")\
''',
    ),
    ("md", "## ROC curves for each probe"),
    (
        "code",
        '''\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))
roc_results = {}
for metric, label in [("em_direction", "EM-direction probe"), ("assistant_axis", "Assistant-Axis probe")]:
    roc = monitor.compute_roc(em_threshold=0.05, metric=metric)
    roc_results[metric] = {k: roc.get(k) for k in ("auc", "error")}
    if roc.get("auc") is not None:
        ax.plot(roc["fpr"], roc["tpr"], "o-", label=f"{label} (AUC={roc['auc']:.2f})")
    else:
        print(f"{label}: ROC unavailable ({roc.get('error')})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Probe ROC (per-checkpoint EM>=5% classification)")
ax.legend()
fig.savefig(RESULTS_DIR / "02_probe_roc.png", dpi=150, bbox_inches="tight")
plt.show()

with open(RESULTS_DIR / "02_roc_auc.json", "w") as f:
    json.dump(roc_results, f, indent=2)
print("Saved 02_roc_auc.json")\
''',
    ),
]

write_notebook("02_checkpoint_monitoring.ipynb", cells)
