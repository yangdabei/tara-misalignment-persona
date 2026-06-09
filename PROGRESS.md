# PROGRESS.md — Agent Handoff File
Last updated: 2026-06-10 (session 3 — fixed steering-coefficient units bug)
Active phase: Scaffold complete and pushed to GitHub. Execution phase = running the 3
notebooks on Colab. Session 2 got past the first two Colab load-time errors; the next
session continues running notebook 01 from the model-load cell onward.

## Repo / environment facts (read first)
- **GitHub: https://github.com/yangdabei/tara-misalignment-persona — PUBLIC.** Created and
  pushed this session via `gh` (account `yangdabei`). Default branch `main`.
- **`scripts/` is intentionally NOT in the repo.** It is gitignored (local-only build tooling).
  So on GitHub you see src/, notebooks/, tests/, data/, results/ (READMEs only), docs — but no
  scripts/. The build_nbXX.py / build_merged.py files still exist on the local machine at
  /Users/yangd/Documents/tara-misalignment-persona/scripts/ and are the source of truth for
  the notebooks. If a future session is on a fresh clone, scripts/ will be ABSENT — the
  committed .ipynb files are self-contained and runnable without them.
- **Clone URL is already correct.** scripts/nb_common.py SETUP_CELL clones
  https://github.com/yangdabei/tara-misalignment-persona.git — no placeholder left to fix.
  The Colab setup cell does NOT re-clone over an existing Drive dir, so to pick up new commits
  the user must `cd` into the Drive clone and `git pull` (noted in the Colab-run section).
- **Colab dependency pins (in requirements.txt, already pushed):** `peft>=0.15.0` +
  `torchao>=0.16.0`. Reason: Colab preinstalls torchao 0.10.0; modern peft enforces
  torchao>0.16 at import. We first tried pinning peft DOWN (<0.15) but that broke worse —
  old peft passes `use_auth_token` to hf_hub_download, removed in huggingface_hub 1.x, and
  peft swallowed the TypeError and misreported it as "Can't find adapter_config.json". The
  fix is to go forward: modern peft + upgraded torchao. Do NOT re-pin peft below 0.15.

## Completed tasks (session 1 — scaffold)
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
- [x] 2.1–2.7 Seven per-stage cell libraries (scripts/build_nb00.py … build_nb06.py).
      These are import-only: each exposes a module-level `cells` list and writes nothing.
- [x] 2.8 Single build entry point scripts/build_merged.py composes them into exactly 3
      model-grouped notebooks: 01_qwen_analysis (00+01+02), 02_qwen_capping_finetune (03),
      03_gemma_geometry_robustness (04+05+06). One model load per Colab session.
- [x] 3.1 results/README.md — file → notebook → figure mapping
- [x] 3.2 This final PROGRESS.md update
- [x] 3.3 git tag v1.0 (see Files/commits)

## Verification done (session 1 scaffold; re-checked session 2)
- All src modules import cleanly (CPU venv with torch/transformers/peft/openai/datasets/sklearn).
- `pytest tests/` → 14 passed.
- CheckpointMonitor lead-time/ROC smoke-tested with synthetic data (probes lead correctly,
  AUC computed, from_log round-trips).
- All 3 merged notebooks: valid nbformat-4 JSON; every code cell compiles (Colab magics
  treated as statements). They are generated artifacts — edit the per-stage cell library
  (scripts/build_nb00.py … build_nb06.py) and re-run scripts/build_merged.py, not the
  .ipynb directly. build_merged.py is the only script that writes notebooks.

## Session 2 changes (done this session)
- [x] Created the GitHub repo and pushed all 5 scaffold commits. Made it PUBLIC.
- [x] Removed scripts/ from git tracking + added to .gitignore (kept on disk). Repo no
      longer ships the build tooling; the 3 committed notebooks are self-contained.
- [x] Fixed Colab load error #1 (torchao import-floor) and #2 (peft use_auth_token /
      "Can't find adapter_config.json"). Final fix: requirements.txt pins
      peft>=0.15.0 + torchao>=0.16.0. Pushed. See "Repo / environment facts" above.
- [x] Verified the EM adapter repo is real: `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train`
      has adapter_config.json + adapter_model.safetensors at root, AND a `checkpoints/`
      subfolder (relevant to the notebook-02 lead-time analysis — see blockers).
- [x] Added tqdm progress bars to all long-running loops: src generation (local batches +
      parallel API calls, judge calls labelled "judging"), activation extraction (per-prompt
      forward passes), assistant-axis persona loop; and notebook loops (steering sweep,
      checkpoint sweep, causal sweep, case-study turn projections, adaptive suffix attack
      with a live best-projection postfix). Uses tqdm.auto; auto-disables for tiny loops.
- [x] Refactored scripts so build_merged.py is the SOLE notebook-build entry point producing
      exactly 3 notebooks. The seven build_nbXX.py are now import-only cell libraries
      (module-level `cells` list, no __main__ write block). tests still 14/14, notebooks rebuilt.

## Session 3 changes (done this session)
- [x] **Fixed the flat 0% steering sweep in notebook 01.** SteeringHook scales the (unit)
      direction by `coef * ||h||` per token — coef is a FRACTION of the hidden norm (ARENA
      4.1 convention; verified against ARENA_3.0 solutions: default 0.4, sweep 0.2–1.0).
      The notebooks were sweeping coef 2.0–8.0, i.e. adding 2–8× the hidden norm at every
      token → gibberish → coherence < 0.5 → is_em_response always False → EM rate pinned
      at 0%. Fixes: nb01 sweep → [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; nb05 causal sweep
      [0,4,8] → [0, 0.4, 0.8]; SteeringHook default 4.0 → 0.4. Notebooks rebuilt, 14/14
      tests pass.
- [x] **Fixed steering applied to all tokens** (second cause of the flat sweep): ARENA's
      validated sweep uses apply_to_all_tokens=False (steer each generated token only);
      True also perturbs Qwen's massive-norm attention-sink prompt token. nb01 + nb05
      cells now pass False; the sweep cell also records mean alignment/coherence and the
      raw judged responses per coef so a flat EM rate is diagnosable.
- [x] **Fixed the extracted direction being a TOPIC direction, not the EM direction**
      (third cause, confirmed live: cos(ours, released vector) = 0.016, steered outputs
      drifted to gender-roles content). Cause: judged responses are ordered by question
      and the extraction cell sliced aligned[:32]/misaligned[:32], so the two sets had
      disjoint topic mixes (misaligned dominated by the gender-roles question). Fix: the
      nb01 extraction cell now balances the sets PER QUESTION (equal aligned/misaligned
      counts per question, cap 8) so topic cancels in the mean-diff. nb04/nb05 contrastive
      extractions already use identical question lists on both sides — unaffected.
      NOTE: any 01_em_mean_diff_directions.pt saved on Drive before this fix is junk —
      delete it so the direction is re-extracted.
- [x] **Stale-result gotcha:** the bad sweep wrote
      `/content/drive/MyDrive/tara_project/results/01_steering_verification.json`; the
      quick-resume cell will reload it. Delete that file on Drive before re-running the
      steering cell (also delete 01_direction_comparisons.json if it was written).

## Currently in progress (or next to start)
Running notebook 01 on Colab. Got past setup + the two load-time errors; the immediate next
step is to re-run the model-load cell (now that peft/torchao are pinned) and confirm the
EM organism loads, then proceed through the rest of notebook 01.

## Pending tasks (ordered) — execution phase
- [ ] In the active Colab session: `cd` into the Drive clone, `git pull` (to get the
      requirements/tqdm fixes), then **Runtime → Restart runtime**, and re-run notebook 01
      from the top. Confirm baseline EM ~11% (stage 00).
- [ ] Finish notebook 01 (01_qwen_analysis = stages 00+01+02): EM direction, steering
      verification, Assistant Axis (Qwen2.5-14B L24), checkpoint monitoring, lead-time + ROC.
- [ ] Run notebook 02 (02_qwen_capping_finetune = stage 03) as its own session
      (A100 80GB, ~4–6 h): baseline vs capping-during-training vs capping-at-inference.
- [ ] Run notebook 03 (03_gemma_geometry_robustness = stages 04+05+06) on A100 80GB:
      geometry heatmap, principal angle, causal test, adversarial robustness Pareto.
- [ ] Fill in the results table below and write up findings.

## Key numerical results (fill in as notebooks run)
Source column names the merged notebook (and the original stage prefix on the result file).
| Metric | Value | Source notebook (stage) |
|---|---|---|
| Baseline EM rate (R1 model) | TBD | 01_qwen_analysis (00) |
| EM direction vs. Assistant Axis cosine (layer 24, Qwen2.5-14B) | TBD | 01_qwen_analysis (02) |
| EM direction vs. Assistant Axis cosine (layer 22, Gemma-2-27B) | TBD | 03_gemma_geometry_robustness (05) |
| Lead time: EM direction probe vs. behavioural eval (steps) | TBD | 01_qwen_analysis (02) |
| Capping-training EM rate reduction vs. baseline (%) | TBD | 02_qwen_capping_finetune (03) |
| Adaptive attack harm rate vs. static capping harm rate | TBD | 03_gemma_geometry_robustness (06) |

## Blockers / open questions (HF repo IDs to confirm at runtime)
- ARENA source files (4_1_*, 4_4_* solutions) were NOT present at /mnt/user-data/uploads/ on
  the build machine. All src/ code was written from the prompt's detailed specs. When running
  on Colab, spot-check that SteeringHook scaling and the judge prompts match the ARENA originals
  if those files are available.
- `ModelOrganismsForEM/bad-medical-advice` dataset name — verify exact HF id (notebook 03,
  EMFineTuneConfig.dataset_repo). The trainer expects a "messages" chat column; adapt
  _tokenize_dataset if the schema differs.
- EM training-checkpoint repo — UPDATE (session 2): the EM adapter repo
  `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train` DOES contain a `checkpoints/`
  subfolder. Notebook 01 stage-02 currently tries two *separate* candidate checkpoint repos
  (CANDIDATE_CHECKPOINT_REPOS in build_nb02.py) and falls back to adapter-interpolation if
  neither resolves. Next session should inspect that checkpoints/ subfolder layout on HF and,
  if it holds per-step adapters, point the monitor at it for REAL (not interpolated) lead-time
  numbers. Until then lead-time is approximate — document which path was used.
- Assistant Axis HF file layout — load_hf_assistant_axis() tries several filenames; if it
  raises, inspect the repo listing (cell prints it in notebook 04) and extend `candidates`.
- Gemma EM organism — none assumed public; notebook 04 derives the EM direction via contrastive
  system prompts. Prefer a real fine-tune if ModelOrganismsForEM has published one (cell checks).
- Toxic-persona + refusal directions for Gemma — notebooks 05 try HF/Arditi releases, else
  derive contrastively. Methods are recorded in 05_geometry_results.json.
- Shah et al. jailbreak dataset — not committed; notebook 06 falls back to a small built-in
  persona set so it runs end-to-end. Drop the real dataset at
  data/jailbreak_prompts/shah_et_al_prompts.json for publishable numbers.

## Repo layout (what's tracked vs. local-only)
Tracked & on GitHub:
- README.md, PROGRESS.md, requirements.txt, .env.example, .gitignore
- data/README.md, data/jailbreak_prompts/README.md, data/eval_questions/*.json, results/README.md
- src/helpers/{model_utils,hook_utils,generation_utils,judge_utils}.py
- src/directions/{mean_diff,assistant_axis,geometry}.py
- src/monitoring/checkpoint_monitor.py, src/finetuning/lora_trainer.py
- tests/{test_directions,test_hooks}.py
- notebooks/{01_qwen_analysis, 02_qwen_capping_finetune, 03_gemma_geometry_robustness}.ipynb
Local-only (gitignored, NOT on GitHub):
- scripts/nb_common.py + scripts/build_nb00..06.py + scripts/build_merged.py
  (the notebook source/build tooling — lives only on /Users/yangd/Documents/...)

## How to regenerate notebooks (local machine only)
The 3 .ipynb are generated from the local scripts/. To change a notebook: edit the relevant
scripts/build_nbXX.py cell library, then run `python scripts/build_merged.py` (the ONLY script
that writes notebooks; it emits exactly the 3 merged ones). Then commit notebooks/. Do not
hand-edit the .ipynb. On a fresh clone without scripts/, you can still run the notebooks as-is.

## Next agent: start here
Load PROGRESS.md, then `cat` the "Repo / environment facts" block at the top. The scaffold is
complete and PUBLIC at github.com/yangdabei/tara-misalignment-persona. You are in the execution
phase, mid-way through notebook 01 on Colab.

Immediate next step: in the live Colab session, `cd` into the Drive clone and `git pull` (to get
the peft/torchao pins + tqdm), **Restart runtime**, then re-run 01_qwen_analysis.ipynb from the
top. Confirm baseline EM ~11% at stage 00, then continue. Run order: 01 → 02 → 03.

If load still fails: the requirements fix is peft>=0.15.0 + torchao>=0.16.0 (already in
requirements.txt on GitHub). Do NOT pin peft below 0.15 (that reintroduces the use_auth_token
bug). The Colab setup cell won't re-clone over an existing Drive dir, so a stale Drive clone is
the usual culprit — `git pull` inside it, or delete the dir to force a fresh clone.

GPU runs happen on Colab; this agent only edits code/notebooks. Update this file at session end.
