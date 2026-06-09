"""
Compose the seven granular notebooks into three model-grouped notebooks, so each
Colab session loads exactly one model:

  01_qwen_analysis.ipynb            = 00 + 01 + 02   (Qwen2.5-14B, inference only)
  02_qwen_capping_finetune.ipynb    = 03            (Qwen2.5-14B, training; kept
                                                     separate — 4-6 h, gradient memory)
  03_gemma_geometry_robustness.ipynb = 04 + 05 + 06 (Gemma-2-27B)

Run order: 01 -> 02 -> 03. Each merged notebook keeps the per-section "load from
Drive if present" cells, so cross-section state flows through Drive and any notebook
is resumable after a runtime disconnect (re-run top to bottom; finished work is skipped).

Within a group the model is loaded once: the first section keeps its model-load cell,
later sections have theirs dropped (pure model-load) or replaced by a lightweight
bridge cell (mixed model-load + data-load) defined below.

Regenerate with:  python scripts/build_merged.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import build_nb00, build_nb01, build_nb02, build_nb03, build_nb04, build_nb05, build_nb06
from nb_common import SETUP_CELL, quick_resume_cell, write_notebook


def _is_setup(cell):
    return cell[0] == "code" and cell[1] == SETUP_CELL


def _is_resume(cell):
    return cell[0] == "code" and cell[1].startswith("# === Quick resume")


def _has(cell, token):
    return cell[0] == "code" and token in cell[1]


def _is_pure_model_load(cell):
    return _has(cell, "load_base_model(") and not _has(cell, "t.load(")


def _is_mixed_load(cell):
    return _has(cell, "load_base_model(") and _has(cell, "t.load(")


# Bridge cells replace the mixed model-load+data-load cells of later sections.
# They assume the model is already in memory (loaded by the group's first section)
# and load only the intermediate results — preferring an in-memory variable if the
# upstream section already produced it this session, else reading from Drive.

BRIDGE_NB02 = '''\
# (Qwen model already loaded above; reuse the EM direction from the previous section.)
from src.helpers.model_utils import QWEN14B_MID_LAYER
if "em_directions" not in globals():
    em_directions = t.load(
        RESULTS_DIR / "01_em_mean_diff_directions.pt", map_location="cpu", weights_only=False
    )
em_direction_l24 = em_directions[QWEN14B_MID_LAYER]
print("EM direction layer 24:", em_direction_l24.shape)\
'''

BRIDGE_NB05 = '''\
# (Gemma model already loaded above; reuse the directions from the previous section.)
import torch.nn.functional as F
from src.helpers.model_utils import GEMMA27B_AXIS_LAYER, GEMMA27B_NUM_LAYERS
if "gemma_directions" in globals():
    gemma = gemma_directions
else:
    gemma = t.load(RESULTS_DIR / "04_gemma_directions.pt", map_location="cpu", weights_only=False)
em_direction = gemma["em_direction"]
assistant_axis = gemma["assistant_axis"]
print("EM direction + Assistant Axis loaded at layer", GEMMA27B_AXIS_LAYER)\
'''

BRIDGE_NB06 = '''\
# (Gemma model already loaded above; reuse the four directions from the previous section.)
import torch.nn.functional as F
from src.helpers.model_utils import GEMMA27B_AXIS_LAYER, load_base_model  # load_base_model kept for standalone use
from src.directions.geometry import project_onto_direction
from src.directions.mean_diff import get_response_activations
if "directions" not in globals():
    directions = t.load(RESULTS_DIR / "05_all_directions.pt", map_location="cpu", weights_only=False)
assistant_axis = directions["Assistant Axis"]
em_direction = directions["EM"]
CAP_LAYERS = [20, 21, 22, 23]  # middle-to-late
print("Loaded directions:", list(directions.keys()))\
'''


def section_cells(builder, *, is_first_section: bool, bridge: str | None):
    """
    Yield a builder's cells stripped of setup/resume, with model-load handling:
      - first section: keep its model-load cell;
      - later section, pure model-load: drop the cell and its preceding "## Load" header;
      - later section, mixed load: replace the cell body with `bridge`.
    """
    out = []
    # Keep the group's first model-load (first section), drop pure model-loads in
    # later sections. Within a section the flag flips once a load is kept.
    kept_first_model_load = not is_first_section
    pending_md = None  # holds the last markdown header so we can drop it with its cell
    for cell in builder.cells:
        if _is_setup(cell) or _is_resume(cell):
            pending_md = None
            continue
        if cell[0] == "md":
            # Defer "## Load..." headers so they can be dropped with a dropped load cell.
            if cell[1].lstrip().startswith("## Load"):
                pending_md = cell
                continue
            if pending_md is not None:
                out.append(pending_md)
                pending_md = None
            out.append(cell)
            continue
        # code cell
        if _is_mixed_load(cell):
            if pending_md is not None:
                out.append(pending_md)
                pending_md = None
            out.append(("code", bridge))
            continue
        if _is_pure_model_load(cell):
            if not kept_first_model_load:
                kept_first_model_load = True
                if pending_md is not None:
                    out.append(pending_md)
                    pending_md = None
                out.append(cell)
            else:
                pending_md = None  # drop the orphaned "## Load" header too
            continue
        if pending_md is not None:
            out.append(pending_md)
            pending_md = None
        out.append(cell)
    return out


def divider(title: str):
    return ("md", f"# ═══════════════════════════════════════════\n# {title}\n# ═══════════════════════════════════════════")


def build_group(filename, title, intro_md, resume_files, sections):
    """sections: list of (builder, divider_title, bridge_or_None)."""
    cells = [("md", intro_md), ("code", SETUP_CELL),
             ("md", "## Quick resume — checks which outputs already exist on Drive"),
             ("code", quick_resume_cell(resume_files))]
    for i, (builder, sec_title, bridge) in enumerate(sections):
        cells.append(divider(sec_title))
        cells.extend(section_cells(builder, is_first_section=(i == 0), bridge=bridge))
    write_notebook(filename, cells)


# --- Notebook 1: Qwen2.5-14B analysis (00 + 01 + 02) ----------------------------
build_group(
    "01_qwen_analysis.ipynb",
    "Qwen analysis",
    "# 01 — Qwen2.5-14B: EM organism analysis & checkpoint monitoring\n\n"
    "**Experiment 1, inference-only.** Merges three stages into one A100 session "
    "(the 14B model is loaded once):\n\n"
    "1. **Setup & sanity** — load base + rank-1 EM organism, baseline EM rate (~11%).\n"
    "2. **EM direction** — extract the mean-diff misalignment direction (all 48 layers), "
    "verify by steering, compare to released + LoRA-B vectors.\n"
    "3. **Checkpoint monitoring** — extract the Assistant Axis (Qwen2.5-14B, L24), sweep "
    "checkpoints, lead-time + ROC of the projection probes.\n\n"
    "Produces the EM direction and Assistant Axis that notebook 02 (capping) consumes.\n\n"
    "Save figures manually if the runtime is temporary; JSON/`.pt` outputs go to Drive.",
    [
        "00_baseline_em_rate.json",
        "01_judged_responses.json",
        "01_em_mean_diff_directions.pt",
        "01_steering_verification.json",
        "01_direction_comparisons.json",
        "02_assistant_axis_qwen14b_l24.pt",
        "02_monitoring/monitoring_log.jsonl",
        "02_lead_time_analysis.json",
    ],
    [
        (build_nb00, "Part 1 — Setup & sanity (Qwen2.5-14B)", None),
        (build_nb01, "Part 2 — Extract & verify the EM direction", None),
        (build_nb02, "Part 3 — Checkpoint monitoring & lead-time", BRIDGE_NB02),
    ],
)

# --- Notebook 2: Qwen2.5-14B capping fine-tuning (03, standalone) ----------------
# Kept separate: 4-6 h training run with a distinct (gradient) memory profile.
# Reuse build_nb03 verbatim — it has its own setup/resume and intentionally loads the
# base model several times (calibration, per-condition training, final eval).
write_notebook("02_qwen_capping_finetune.ipynb", build_nb03.cells)

# --- Notebook 3: Gemma-2-27B geometry & robustness (04 + 05 + 06) ----------------
build_group(
    "03_gemma_geometry_robustness.ipynb",
    "Gemma geometry & robustness",
    "# 03 — Gemma-2-27B: unified geometry map & capping robustness\n\n"
    "**Experiment 2.** Merges three stages into one A100-80GB session "
    "(the 27B model is loaded once):\n\n"
    "1. **Assistant Axis** — load it, validate trait-vector cosines & case-study drift, "
    "extract the Gemma EM direction.\n"
    "2. **Unified geometry map** — the centrepiece 4×4 cosine heatmap, principal angle, "
    "subspace R², per-layer curve, causal steering test.\n"
    "3. **Adversarial robustness** — static vs. adaptive jailbreak, single vs. dual cap, "
    "Pareto frontier.\n\n"
    "Independent of notebooks 01–02 (different model). Save figures manually if the "
    "runtime is temporary.",
    [
        "04_gemma_directions.pt",
        "04_trait_cosine_distribution.png",
        "04_case_study_trajectories.json",
        "05_geometry_heatmap.png",
        "05_geometry_results.json",
        "05_all_directions.pt",
        "06_robustness_results.json",
        "06_robustness_pareto.png",
    ],
    [
        (build_nb04, "Part 1 — Assistant Axis & Gemma EM direction", None),
        (build_nb05, "Part 2 — Unified geometry map (centrepiece)", BRIDGE_NB05),
        (build_nb06, "Part 3 — Adversarial capping robustness", BRIDGE_NB06),
    ],
)
