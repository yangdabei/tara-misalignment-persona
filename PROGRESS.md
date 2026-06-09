# PROGRESS.md — Agent Handoff File
Last updated: 2026-06-09T00:00:00 (session 1, in progress)
Active phase: Phase 0 complete, starting Phase 1 (src/ library code)

## Completed tasks
- [x] 0.1 git init and folder structure — output: full directory tree
- [x] 0.2 .gitignore — output: .gitignore
- [x] 0.3 requirements.txt — output: requirements.txt
- [x] 0.4 .env.example — output: .env.example
- [x] 0.5 eval questions data — output: data/eval_questions/first_plot_questions.json (8 questions)
- [x] 0.6 README.md — output: README.md
- [x] 0.7 PROGRESS.md created — output: this file

## Currently in progress (or next to start)
- [ ] 1.1 src/helpers/model_utils.py

## Pending tasks (ordered)
- [ ] 1.2 src/helpers/hook_utils.py
- [ ] 1.3 src/helpers/generation_utils.py
- [ ] 1.4 src/helpers/judge_utils.py
- [ ] 1.5 src/directions/mean_diff.py
- [ ] 1.6 src/directions/assistant_axis.py
- [ ] 1.7 src/directions/geometry.py
- [ ] 1.8 src/monitoring/checkpoint_monitor.py
- [ ] 1.9 src/finetuning/lora_trainer.py
- [ ] 1.10 tests/ (test_directions.py, test_hooks.py)
- [ ] 2.1 notebooks/00_setup_and_sanity.ipynb
- [ ] 2.2 notebooks/01_em_organism_baseline.ipynb
- [ ] 2.3 notebooks/02_checkpoint_monitoring.ipynb
- [ ] 2.4 notebooks/03_capping_during_finetuning.ipynb
- [ ] 2.5 notebooks/04_assistant_axis_extraction.ipynb
- [ ] 2.6 notebooks/05_unified_geometry_map.ipynb
- [ ] 2.7 notebooks/06_adversarial_capping_robustness.ipynb
- [ ] 3.1 results/README.md
- [ ] 3.2 Final PROGRESS.md update
- [ ] 3.3 git tag v1.0

## Key numerical results (fill in as notebooks run)
| Metric | Value | Source notebook |
|---|---|---|
| Baseline EM rate (R1 model) | TBD | 00_setup_and_sanity |
| EM direction vs. Assistant Axis cosine (layer 24, Qwen2.5-14B) | TBD | 05_unified_geometry_map |
| EM direction vs. Assistant Axis cosine (layer 22, Gemma-2-27B) | TBD | 05_unified_geometry_map |
| Lead time: EM direction probe vs. behavioural eval (steps) | TBD | 02_checkpoint_monitoring |
| Capping-training EM rate reduction vs. baseline (%) | TBD | 03_capping_during_finetuning |
| Adaptive attack harm rate vs. static capping harm rate | TBD | 06_adversarial_robustness |

## Blockers / open questions
- ARENA source files at /mnt/user-data/uploads/ (4_1_emergent_misalignment_solutions.py,
  4_4_llm_psychology___persona_vectors_solutions.py) are NOT present on this machine.
  src/ code is written from the detailed specifications in the project prompt instead.
- EM training-checkpoint HF repo name unconfirmed (notebook 02 includes a fallback that
  uses LoRA-alpha interpolation of the final adapter if no checkpoint repo exists).

## Files created this session
- README.md, PROGRESS.md (plus Phase 0 files from a prior partial run)

## Next agent: start here
Load PROGRESS.md, then do Task 1.1: src/helpers/model_utils.py.
