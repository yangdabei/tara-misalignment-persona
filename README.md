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

3. Run the notebooks **in order** on a Colab A100 runtime:

   | Notebook | Model | Purpose |
   |---|---|---|
   | `00_setup_and_sanity` | Qwen2.5-14B | Load models, baseline EM rate (~11% for R1 organism) |
   | `01_em_organism_baseline` | Qwen2.5-14B | Extract EM mean-diff direction, verify by steering |
   | `02_checkpoint_monitoring` | Qwen2.5-14B | Projection probes across checkpoints, lead-time analysis |
   | `03_capping_during_finetuning` | Qwen2.5-14B | ⚠️ A100-80GB, 4–6 h. Baseline vs capped fine-tuning |
   | `04_assistant_axis_extraction` | Gemma-2-27B | Load/validate Assistant Axis, extract EM direction |
   | `05_unified_geometry_map` | Gemma-2-27B | **Centrepiece**: 4×4 cosine heatmap, principal angles, causal test |
   | `06_adversarial_capping_robustness` | Gemma-2-27B | Static vs adaptive jailbreak, dual-direction capping |

   All results persist to `/content/drive/MyDrive/tara_project/results/`.

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
