# PROGRESS.md — Agent Handoff File
Last updated: 2026-06-09 (session 1 — full scaffold complete)
Active phase: Phase 3 complete. Scaffold done; next phase is running notebooks on Colab.

## Completed tasks
- [x] 0.1–0.7 Repo init, .gitignore, requirements, .env.example, eval questions, README, PROGRESS
- [x] 1.1 src/helpers/model_utils.py — model loading, _return_layers, _normalize_messages, constants
- [x] 1.2 src/helpers/hook_utils.py — SteeringHook, ActivationCapper (floor+ceiling), CappingCallback, ActivationCache
- [x] 1.3 src/helpers/generation_utils.py — local batched + parallel API generation, OpenRouter client
- [x] 1.4 src/helpers/judge_utils.py — alignment/coherence judges, is_em_response, run_first_plot_eval
- [x] 1.5 src/directions/mean_diff.py — response-activation extraction, mean-diff (per-layer), LoRA-B extraction
- [x] 1.6 src/directions/assistant_axis.py — load HF axis, fresh extraction, capping calibration
- [x] 1.7 src/directions/geometry.py — cosine matrix, principal angles, ablation, subspace R², heatmap
- [x] 1.8 src/monitoring/checkpoint_monitor.py — per-checkpoint logging, lead-time, ROC, plots
- [x] 1.9 src/finetuning/lora_trainer.py — EMFineTuneConfig, 9-adapter LoRA, capping-aware training loop
- [x] 1.10 tests/ — test_directions.py + test_hooks.py (14 tests, all passing)
- [x] 2.1–2.7 All 7 Colab notebooks (built via scripts/build_nbXX.py)
- [x] 3.1 results/README.md — file → notebook → figure mapping
- [x] 3.2 This final PROGRESS.md update
- [x] 3.3 git tag v1.0 (see Files/commits)

## Verification done this session
- All src modules import cleanly (CPU venv with torch/transformers/peft/openai/datasets/sklearn).
- `pytest tests/` → 14 passed.
- CheckpointMonitor lead-time/ROC smoke-tested with synthetic data (probes lead correctly,
  AUC computed, from_log round-trips).
- All 7 notebooks: valid nbformat-4 JSON; every code cell compiles (Colab magics treated as
  statements). Notebooks are generated artifacts — edit scripts/build_nbXX.py and re-run, not
  the .ipynb directly.

## Currently in progress (or next to start)
Nothing pending in the scaffold. The next session runs the notebooks on Colab in order
(00 → 06) and fills in the numerical results table below.

## Pending tasks (ordered) — execution phase
- [ ] Replace the clone URL placeholder `https://github.com/YOUR_USERNAME/...` in the SETUP_CELL
      (scripts/nb_common.py) with the real repo URL, re-run all builders, push.
- [ ] Run notebooks 00 → 02 on a Colab A100 (Qwen2.5-14B). Confirm baseline EM ~11%.
- [ ] Run notebook 03 last (A100 80GB, 4–6 h): three fine-tuning conditions.
- [ ] Run notebooks 04 → 06 on Colab A100 80GB (Gemma-2-27B).
- [ ] Fill in the results table below and write up findings.

## Key numerical results (fill in as notebooks run)
| Metric | Value | Source notebook |
|---|---|---|
| Baseline EM rate (R1 model) | TBD | 00_setup_and_sanity |
| EM direction vs. Assistant Axis cosine (layer 24, Qwen2.5-14B) | TBD | 02_checkpoint_monitoring |
| EM direction vs. Assistant Axis cosine (layer 22, Gemma-2-27B) | TBD | 05_unified_geometry_map |
| Lead time: EM direction probe vs. behavioural eval (steps) | TBD | 02_checkpoint_monitoring |
| Capping-training EM rate reduction vs. baseline (%) | TBD | 03_capping_during_finetuning |
| Adaptive attack harm rate vs. static capping harm rate | TBD | 06_adversarial_robustness |

## Blockers / open questions (HF repo IDs to confirm at runtime)
- ARENA source files (4_1_*, 4_4_* solutions) were NOT present at /mnt/user-data/uploads/ on
  the build machine. All src/ code was written from the prompt's detailed specs. When running
  on Colab, spot-check that SteeringHook scaling and the judge prompts match the ARENA originals
  if those files are available.
- `ModelOrganismsForEM/bad-medical-advice` dataset name — verify exact HF id (notebook 03,
  EMFineTuneConfig.dataset_repo). The trainer expects a "messages" chat column; adapt
  _tokenize_dataset if the schema differs.
- EM training-checkpoint repo — notebook 02 tries two candidate ids and falls back to
  adapter-interpolation (scaling LoRA B by step/max) if neither exists. Lead-time analysis is
  approximate in that case; document it.
- Assistant Axis HF file layout — load_hf_assistant_axis() tries several filenames; if it
  raises, inspect the repo listing (cell prints it in notebook 04) and extend `candidates`.
- Gemma EM organism — none assumed public; notebook 04 derives the EM direction via contrastive
  system prompts. Prefer a real fine-tune if ModelOrganismsForEM has published one (cell checks).
- Toxic-persona + refusal directions for Gemma — notebooks 05 try HF/Arditi releases, else
  derive contrastively. Methods are recorded in 05_geometry_results.json.
- Shah et al. jailbreak dataset — not committed; notebook 06 falls back to a small built-in
  persona set so it runs end-to-end. Drop the real dataset at
  data/jailbreak_prompts/shah_et_al_prompts.json for publishable numbers.

## Files created this session
- README.md, PROGRESS.md, data/README.md, data/jailbreak_prompts/README.md, results/README.md
- src/helpers/{model_utils,hook_utils,generation_utils,judge_utils}.py
- src/directions/{mean_diff,assistant_axis,geometry}.py
- src/monitoring/checkpoint_monitor.py
- src/finetuning/lora_trainer.py
- tests/{test_directions,test_hooks}.py
- scripts/nb_common.py + scripts/build_nb00..06.py
- notebooks/00..06 (generated)

## Next agent: start here
Load PROGRESS.md. The scaffold is complete and tagged v1.0. Begin the execution phase:
set the real repo clone URL in scripts/nb_common.py, re-run the builders, then run
notebooks/00_setup_and_sanity.ipynb on a Colab A100 and confirm the baseline EM rate (~11%).
