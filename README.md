# Is Misalignment a Persona Move? Unifying the EM Direction and the Assistant Axis

**TARA Fellowship final project** — mechanistic interpretability of emergent misalignment.

## Research question

Is emergent misalignment (EM) mechanistically the same as persona drift along the
Assistant Axis? Can we detect it mid-training via activation projections, and prevent
it with activation capping?

## Paper references

- **Soligo et al. (2025)** — *Convergent Linear Representations of Emergent Misalignment*,
  [arXiv:2506.11618](https://arxiv.org/abs/2506.11618). Source of the EM model organisms,
  the 9-adapter rank-1 LoRA setup, and the mean-diff misalignment direction.
- **Lu et al. (2026)** — *The Assistant Axis*,
  [arXiv:2601.10387](https://arxiv.org/abs/2601.10387). Source of the Assistant Axis,
  persona-drift framing, and activation capping (eq. 1).
- **Betley et al. (2025)** — *Emergent Misalignment* (eval questions and LLM-judge convention).
- **Arditi et al. (2024)** — refusal direction.

## Experiments

### Experiment 1 — Qwen2.5-14B-Instruct (EM focus)

Reproduce the 9-adapter EM organism (rank-1 LoRA on MLP `down_proj` of layers
15–17, 21–23, 27–29), checkpointing every 20 steps. At each checkpoint log:

1. Behavioural misalignment (EM rate via LLM judge, Betley et al. convention),
2. Projection onto the EM mean-diff direction (layer 24),
3. Projection onto the Assistant Axis (extracted fresh on Qwen2.5-14B).

Test whether the projection probes give **positive lead time** over behavioural evals.
Then re-run fine-tuning with Assistant-Axis activation capping active and compare
final misalignment against the baseline run.

### Experiment 2 — Gemma-2-27B-it (geometry focus)

Build the **unified geometry map**: pairwise cosine similarities and principal angles
between (a) the EM misalignment direction, (b) the Assistant Axis, (c) the toxic-persona
direction, (d) the refusal direction. Then run adversarial robustness tests of activation
capping under an adaptive greedy-suffix attack.

## Success criteria

1. A geometry heatmap with measured cosine similarities (a negative result —
   EM direction ⊥ Assistant Axis — is also a strong publishable finding).
2. A lead-time plot showing whether the projection monitor fires before behavioural
   misalignment crosses threshold.
3. A Pareto plot (harm-reduction % vs capability %) for capping-during-training
   vs baseline vs capping-at-inference.

## Setup

1. Clone this repo (on Colab, clone into `/content/drive/MyDrive/tara_project/`):

   ```bash
   git clone <repo-url> tara-misalignment-persona
   cd tara-misalignment-persona
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY` and `HF_TOKEN`
   (on Colab, use Colab Secrets with the same names instead).

3. Run the notebooks **in order** on a Colab A100 runtime. There are three
   notebooks, grouped so each Colab session loads exactly one model:

   | Notebook | Model | Purpose |
   |---|---|---|
   | `01_qwen_analysis` | Qwen2.5-14B | Setup & sanity (baseline EM ~11%) → extract & verify the EM mean-diff direction → checkpoint monitoring + lead-time/ROC. Produces the EM direction and Assistant Axis that notebook 02 consumes. |
   | `02_qwen_capping_finetune` | Qwen2.5-14B | ⚠️ A100-80GB, ~4–6 h. Baseline vs capping-during-training vs capping-at-inference; Pareto table. Run as its own session. |
   | `03_gemma_geometry_robustness` | Gemma-2-27B | Assistant Axis + Gemma EM direction → **centrepiece** 4×4 cosine heatmap, principal angle, causal test → adversarial capping robustness (static vs adaptive, single vs dual cap). |

   Each notebook is **resumable**: it mounts Drive, and a quick-resume cell skips any
   stage whose outputs already exist, so a runtime disconnect just means re-running top
   to bottom. Figures display inline (save them manually on a temporary runtime); all
   JSON/`.pt` outputs persist to `/content/drive/MyDrive/tara_project/results/`.

   The notebooks are generated artifacts — edit `scripts/build_nbXX.py` (granular stages)
   and re-run `python scripts/build_merged.py` rather than editing the `.ipynb` directly.

## Expected results

- **cos(EM, Assistant Axis) > 0.5**: misalignment *is* a persona move — EM fine-tuning
  pushes the model off the Assistant persona along the same axis that separates helpful
  assistants from role-playing characters.
- **cos < 0.2**: EM and persona drift are distinct directions; defences targeting one
  do not cover the other.
- **0.2–0.5**: partial overlap — requires the cone/subspace framing.
- Lead time of 20–60 training steps for the projection probes would be a significant
  safety-relevant result; correlation without lead time is still publishable.

## Repo layout

- `src/` — shared Python utilities (model loading, hooks, generation, judging,
  direction extraction, geometry, checkpoint monitoring, LoRA fine-tuning).
- `notebooks/` — self-contained Colab notebooks (mount Drive, install, run).
- `data/` — eval questions and jailbreak-prompt download instructions.
- `results/` — notebook outputs (saved to Drive; large files git-ignored).
- `tests/` — pytest unit tests for geometry and hook utilities.
