# PROGRESS.md — Agent Handoff File
Last updated: 2026-06-10 (session 4 end — GPU runs MOVED FROM COLAB TO RUNPOD; notebook 03
fixed (case-study schema + OOM) and ready to run on an H100-80GB pod; pip install was
still finishing on the pod at session end; notebook 02 still deferred pending re-scope)
Active phase: execution + write-up. Notebook 01 ran end-to-end (executed copy with outputs
committed at notebooks/01_qwen_analysis_outputs.ipynb; key numbers in the results table
below). Run order changed to 01 → 03 → 02 (03 is independent of 02 and its geometry
results de-risk 02's design). A draft 10-min results deck exists locally (gitignored) at
presentation/em_results_10min.pptx. **READ THE "RunPod environment" SECTION before
debugging any GPU-run issue — Colab assumptions no longer hold.**

## Repo / environment facts (read first)
- **GitHub: https://github.com/yangdabei/tara-misalignment-persona — PUBLIC.** Created and
  pushed this session via `gh` (account `yangdabei`). Default branch `main`.
- **`scripts/` is intentionally NOT in the repo.** It is gitignored (local-only build tooling).
  So on GitHub you see src/, notebooks/, tests/, data/, results/ (READMEs only), docs — but no
  scripts/. RESTRUCTURED (session 3): exactly one build script per notebook, numbers matching —
  scripts/build_nb01.py → notebooks/01_qwen_analysis.ipynb, build_nb02.py → 02_qwen_capping_
  finetune.ipynb, build_nb03.py → 03_gemma_geometry_robustness.ipynb (plus shared nb_common.py,
  which now also holds the multi-part merge machinery: section_cells/build_group/bridges live
  in the per-notebook scripts). The old seven per-stage libraries (build_nb00..06) and
  build_merged.py are GONE. These live only on the local machine at
  /Users/yangd/Documents/tara-misalignment-persona/scripts/ and are the source of truth for
  the notebooks. If a future session is on a fresh clone, scripts/ will be ABSENT — the
  committed .ipynb files are self-contained and runnable without them.
- **`presentation/` is local-only too** (gitignored, session 3): presentation/
  em_results_10min.pptx is the 10-minute results deck; it was briefly committed and then
  untracked, so it still exists in git history but not in the working tree on GitHub.
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
  treated as statements). They are generated artifacts — see "How to regenerate notebooks"
  below for the current (session-3) one-script-per-notebook build layout.

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
- [x] **Stale-result gotcha (now historical):** quick-resume reloads any existing result
      file by name, so a result produced by buggy code must be DELETED from Drive
      (results/ under /content/drive/MyDrive/tara_project/) or it silently masks the fix.
      This bit us twice this session; remember it whenever a notebook cell is changed.
- [x] **Fixed the released-vector comparison** (was meaningless): the compare cell took
      files[0] of the HF repo listing = checkpoints/checkpoint-10 (nearly untrained,
      cos 0.30 to final). Now loads the root steering_vector.pt explicitly (== final,
      checkpoint-676; dict keys steering_vector/layer_idx=24/alpha=256).
- [x] **Corrected the cosine EXPECTATION for that comparison**: the released vector is
      TRAINED with SGD as a rank-1 write adapter — the analog of a LoRA-B vector — and
      Soligo et al. report cos(B, mean-diff) ≈ 0.04 (the "B-vector mystery"). Mean-diff
      directions only converge (>0.8) with other mean-diff directions. So a near-zero
      cosine there is EXPECTED, not a bug. Real validity checks: split-half reliability
      (added to the extraction cell) + behavioural steering.
- [x] **Wired checkpoint monitoring to REAL checkpoints**: the adapter repo ships 88
      per-step PEFT adapters under checkpoints/checkpoint-<N>/ (see Blockers, RESOLVED).
- [x] **Fixed Assistant-Axis extraction**: OpenRouter delisted qwen/qwen-2.5-14b-instruct
      (400 error). The API path was unnecessary anyway — the cell runs inside
      disable_adapter(), which IS clean Qwen2.5-14B — so extraction is now local
      (api_client=None); API default model bumped to qwen-2.5-72b-instruct.
- [x] **NOTEBOOK 01 RAN END-TO-END.** Executed copy with all outputs committed at
      notebooks/01_qwen_analysis_outputs.ipynb. All numbers in the results table below.
- [x] **Added AND RAN the EM-on-Assistant-Axis visualisation cell** (end of nb01 part 3):
      histograms of per-response Axis projections (EM vs aligned responses) + persona anchor
      vlines; 02_em_on_assistant_axis.png saved on Drive. Result: heavy OVERLAP — mean
      projection aligned +6.31 vs EM +7.00 (n=40/19), EM responses NOT displaced toward the
      role-play anchors (leviathan/ghost at −16…0; default-assistant anchors ~13.5–14.5).
      Visual confirmation that EM is not persona drift along the Axis. Interpretation
      caveats: density-normalised histograms make the n=19 EM bars tall (tall ≠ more);
      anchors come from the base model on extraction questions, histograms from the EM
      model on eval questions — anchors orient the axis, they are not a strict ruler.
- [x] **Drafted the 10-min results deck**: presentation/em_results_10min.pptx (local-only,
      gitignored; 11 slides + speaker notes, embeds the three figures from the executed
      notebook). Regenerate/extend with python-pptx (installed in the local .venv).
- [x] **Restructured scripts/ to one build script per notebook** (numbers match notebooks;
      build_merged.py + build_nb00..06 gone; merge machinery in nb_common.py). Verified
      by rebuilding all three notebooks byte-identical.

## RunPod environment (session 4 — GPU runs now happen here, NOT Colab)
Colab was abandoned for notebook 03: no H100 available, and the A100-40GB OOMed (27B bf16
≈ 55 GB). Current setup:
- **Pod: H100 SXM 80GB, Secure Cloud, region US-NE-1, $3.29/hr.** SSH config entry
  `runpod` on the user's Mac (~/.ssh/config) → root@216.243.220.219 port 13347, key
  ~/.ssh/id_ed25519. **IP/port change on every new pod** — user pastes the new
  "SSH over exposed TCP" command (NOT the ssh.runpod.io proxy, which VS Code can't use)
  and the agent sed-updates ~/.ssh/config. Pubkey must be in RunPod account Settings →
  SSH Public Keys (done) — keys added there only inject at pod START.
- **Persistent network volume mounted at /workspace** (survives pod resets; tied to
  US-NE-1, so new pods must be in that region to attach it). Repo cloned at
  /workspace/tara-misalignment-persona with a real .env (HF_TOKEN + OPENROUTER_API_KEY
  filled in; format KEY=value). **VERIFY VOLUME SIZE ≥100GB** (Storage → Network
  Volumes; grow-only, live) — user was asked to confirm at session end; if 50GB the
  54.5GB Gemma download will hit EDQUOT (os error 122) like pod #1 did. Old orphaned
  volumes from earlier pods should be deleted (they keep billing).
- **Container disk 50GB is EPHEMERAL** — wiped on pod stop/reset. pip installs live
  there → re-run after every reset: `pip install --break-system-packages -q -r
  requirements.txt ipywidgets ipykernel` (the ubuntu2404 image enforces PEP 668; without
  the flag pip dies with "externally managed environment"). NEVER run two pip installs
  concurrently (they race site-packages; this happened twice).
- **HF_HOME=/workspace/hf_cache** — set on the pod (/etc/environment + .bashrc) AND
  defaulted by the notebooks' setup cell whenever /workspace exists (commit 24967bf,
  all 3 notebooks). Without it the 54.5GB model lands on the 50GB container disk and
  fails at the end. Model cache persists on the volume across pods → one-time download.
- **Network quirks measured on these pods:** HuggingFace ≈ 89 MB/s (model = ~10 min),
  but PyPI only ~200 kB/s (pip takes ~15 min) — that's the pod's routing, not a fault.
  When bandwidth-testing, beware: speed.cloudflare.com is blocked/dead from this DC, and
  gated HF repos (e.g. google/gemma-2-2b) return tiny 401 bodies that look like 0 B/s —
  test with a PUBLIC repo file, e.g. Qwen/Qwen2.5-0.5B-Instruct model.safetensors.
- **Running notebooks: VS Code Remote-SSH** (connect to host `runpod`, open
  /workspace/tara-misalignment-persona, system python3 kernel). The user's VS Code
  settings map `"remote.SSH.serverInstallPath": {"runpod": "/workspace/.vscode-server"}`
  so the VS Code server + remote extensions persist on the volume (no re-download per
  pod). tqdm widget render errors in VS Code are cosmetic; `pip install -U ipywidgets`
  + kernel restart fixes them. For disconnect-proof long runs: papermill in tmux.
- **Results now flow: pod → user's Mac (repo results/, the canonical local store).**
  nb01 results already downloaded from Drive into local results/ (verified post-fix
  versions; 01_em_mean_diff_directions.pt there is the good one nb02 needs — upload it
  to the pod's results/ before running nb02). .gitignore now covers all of results/
  except README.md (commit 45b9c4a), so nothing leaks via git; move files with scp /
  runpodctl send.
- The agent can (and did) run pod-side commands directly via
  `ssh -o BatchMode=yes runpod '...'` — setup, diagnostics, killing stray processes.

## Currently in progress (or next to start)
- USER IS ABOUT TO RUN NOTEBOOK 03 (Gemma geometry & robustness) on the RunPod H100 —
  see "RunPod environment" above. At session-4 end: repo cloned+pulled on pod, .env real,
  HF_HOME set, pip install still running (background watcher was attached; verify with
  `ssh runpod 'pgrep -af "pip install"'` and `python3 -c "import dotenv, peft, torchao"`).
  Remaining before Run All: confirm volume ≥100GB. Model download ~10 min on first load.
  Run order is 01 → 03 → 02: notebook 03 is independent of 02, and its geometry
  stage (cos(EM, Axis) on Gemma + the causal steering→axis-projection test) determines
  whether 02's Assistant-Axis-capping design is worth its 4–6 h cost (see next bullet).
- NOTEBOOK 02 RE-SCOPE (recommended, not yet implemented): nb01 found EM is nearly
  ORTHOGONAL to the Assistant Axis (cos −0.074, flat projection, AUC 0.67), so 02's
  axis-capping intervention is predicted to fail (a clean, pre-registered negative).
  Recommendation discussed with user: add an EM-DIRECTION ceiling-capping condition
  (ActivationCapper mode="ceiling" with em_direction_l24, threshold calibrated on base
  model) alongside the axis condition, turning 02 into "cap the persona axis vs cap the
  misalignment direction itself". User has NOT yet said go — confirm before editing
  scripts/build_nb02.py.
- PRESENTATION: user wants to add example prompts + outputs for the traits studied; they
  are downloading results from Drive. Source material already available: base-vs-EM
  contrast quotes (cells of 01_qwen_analysis_outputs.ipynb, e.g. the "one wish" question),
  00_baseline_em_rate.json (em_model_raw/base_model_raw: question/answer/alignment/
  coherence per response), 01_judged_responses.json (200 judged organism responses),
  01_steering_verification.json ("responses" per coef — steered outputs), and from
  notebook 03: 04_case_study_trajectories.json + persona/trait prompts in
  src/directions/assistant_axis.py (ASSISTANT_LIKE/ROLE_PLAYING/DEFAULT personas) and
  the contrastive system prompts in the nb03 cells. Deck lives at
  presentation/em_results_10min.pptx; add a traits-examples slide + the
  02_em_on_assistant_axis.png figure once produced.

## Session 4 changes (done this session)
- [x] **Fixed nb03 case-study cell crash** (KeyError: 0): the safety-research/assistant-axis
      transcripts are dicts with a "conversation" key (not "messages"), and the shipped case
      studies are delusion/selfharm/jailbreak (each _unsteered + _capped per source model) —
      NOT isolation/suicid as the cell guessed. Now matches unsteered runs only and takes the
      longest conversation per keyword (qwen-3-32b delusion = 23 assistant turns). Commit 90fcdda.
- [x] **Fixed nb03 case-study OOM**: model(ids) materialised Gemma's 256k-vocab logits for a
      4096-token context ≈ 2 GiB bf16 per turn, never used. Cell now calls model.model(ids)
      (decoder only; the layer-22 ActivationCache hook fires regardless). Commit a6de78a.
- [x] **Moved GPU execution Colab → RunPod** (no H100 on Colab; A100-40GB OOMs on the 27B).
      Full environment + gotchas in the "RunPod environment" section above. Took 3 pod
      attempts: #1 hit EDQUOT (50GB volume < 54.5GB model), #2 was on a host with unusable
      PyPI bandwidth, #3 (current) is good.
- [x] **HF_HOME → /workspace/hf_cache** default in the setup cell of all 3 notebooks when
      /workspace exists (nb_common.py; commit 24967bf) + set in the pod's /etc/environment.
- [x] **.gitignore tightened** (commit 45b9c4a): all of results/ except README.md (old
      patterns missed subdirs like results/02_monitoring/), plus .DS_Store.
- [x] **nb01 results downloaded from Drive → local results/** (canonical local store now).
      Spot-checked 01_steering_verification.json against the results table — post-fix
      versions confirmed, so the local 01_em_mean_diff_directions.pt is the good one.
- [x] VS Code: serverInstallPath for host `runpod` → /workspace/.vscode-server (persists
      VS Code server + extensions on the volume); ~/.ssh/config `runpod` entry added.

## Pending tasks (ordered) — execution phase
- [ ] Notebook 03 on RunPod H100 (next: user about to Run All): geometry heatmap, principal
      angle, causal test, adversarial robustness Pareto. Watch the Blockers section items
      (axis file layout, toxic/refusal fallbacks, jailbreak dataset fallback). Confirm
      volume ≥100GB + pip finished first (see "Currently in progress").
- [ ] Decide + implement the notebook 02 re-scope (EM-direction ceiling-capping condition;
      see Currently in progress), then run 02 (A100 80GB, ~4–6 h).
- [ ] Update the deck with nb03/nb02 results + trait prompt/output examples + the new
      02_em_on_assistant_axis.png figure (run + figure done; see results table).
- [ ] Optional, for publishable nb01 numbers: clear 01_*/00_* results on Drive and re-run
      notebook 01 fresh — committed defaults now use n_samples=50/question and the
      split-half check (the session-3 run used 25 samples → 17 pairs, n=24/coef).
- [ ] Fill in the remaining results-table rows and write up findings.

## Key numerical results (fill in as notebooks run)
Source column names the merged notebook (and the original stage prefix on the result file).
| Metric | Value | Source notebook (stage) |
|---|---|---|
| Baseline EM rate (R1 organism / base model) | 12.5% / 0.0% (n=40 each; paper ~11%; organism mean align 0.77, mean coh 0.91) | 01_qwen_analysis (00) |
| Steering verification: peak EM rate (base model, L24 mean-diff, all_tokens=False) | 29.2% @ coef 0.4 (25.0% @ 0.6, 12.5% @ 0.8, 4.2% @ 1.0; 0% @ 0/0.2; n=24/coef) | 01_qwen_analysis (01) |
| Split-half cosine of mean-diff direction (L24, 17 balanced pairs) | 0.647 | 01_qwen_analysis (01) |
| cos(mean-diff L24, released TRAINED vector — final root file) — low EXPECTED (B-vector mystery) | 0.0087 | 01_qwen_analysis (01) |
| cos(mean-diff, organism LoRA-B write vectors), layers 15–29 | −0.044 … +0.028 (magnitudes ≤0.044; reproduces the ~0.04 B-vector mystery) | 01_qwen_analysis (01) |
| EM direction vs. Assistant Axis cosine (layer 24, Qwen2.5-14B) | −0.074 (≈ orthogonal — EM is NOT drift along the Axis in this organism) | 01_qwen_analysis (02) |
| Per-response Axis projection, EM vs aligned responses (L24) | EM +7.00 vs aligned +6.31 (n=19/40) — overlapping; EM not shifted toward role-play | 01_qwen_analysis (02, 02_em_on_assistant_axis.png) |
| EM direction vs. Assistant Axis cosine (layer 22, Gemma-2-27B) | TBD | 03_gemma_geometry_robustness (05) |
| Lead time: EM direction probe vs. behavioural eval (steps) | 100 (probe fires @ step 150, behaviour ≥5% @ step 250; real checkpoints, steps 0–396) | 01_qwen_analysis (02) |
| Probe ROC AUC, per-checkpoint EM≥5% (11 checkpoints) | EM-direction 1.00; Assistant-Axis 0.67 (axis "lead 240" = noise false-positive @ step 10) | 01_qwen_analysis (02) |
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
- EM training-checkpoint repo — RESOLVED (session 3): the adapter repo
  `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train` ships 88 per-step PEFT
  adapters under `checkpoints/checkpoint-<N>/` (steps 1–396, every step to 10 then every 5;
  same safetensors key layout as the final adapter; B-norm grows 0.019@40 → 0.049@final).
  build_nb02.py now discovers steps from that repo and the switcher hf_hub_downloads each
  step's adapter_model.safetensors and copies the weights into the live PEFT modules
  (PEFT load_adapter can't overwrite an existing adapter name). Step 0 = B weights zeroed.
  Interpolation remains only as a fallback if the listing fails. Lead-time numbers are REAL.
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
- notebooks/01_qwen_analysis_outputs.ipynb (EXECUTED copy of notebook 01 with all outputs —
  source of the deck figures and base-vs-EM example quotes)
Local-only (gitignored, NOT on GitHub):
- scripts/nb_common.py + scripts/build_nb01.py + build_nb02.py + build_nb03.py
  (the notebook source/build tooling, one script per notebook — only on /Users/yangd/...)
- presentation/em_results_10min.pptx (draft 10-min results deck, python-pptx generated)
- Heavy run outputs (gitignored, in the local working tree's results/): nb01's results
  were downloaded from Drive in session 4 and now live in the LOCAL results/ folder —
  the canonical store going forward. Old copies remain on Google Drive
  (/content/drive/MyDrive/tara_project/results/, from the Colab era); new RunPod runs
  write to the pod's /workspace/tara-misalignment-persona/results/ and must be scp'd /
  runpodctl-sent down to the local results/ after each session.

## How to regenerate notebooks (local machine only)
The 3 .ipynb are generated from the local scripts/, one script per notebook (numbers match):
edit scripts/build_nbNN.py and run `python scripts/build_nbNN.py` to rewrite
notebooks/NN_*.ipynb. Multi-part notebooks (01 and 03) define part1_cells/part2_cells/
part3_cells and compose them via nb_common.build_group (strips per-part setup cells, keeps
one model load, swaps later loads for bridge cells). Then commit notebooks/. Do not hand-edit
the .ipynb. On a fresh clone without scripts/, you can still run the notebooks as-is.

## Next agent: start here
Load PROGRESS.md top to bottom; the header, "RunPod environment", and "Currently in
progress" reflect the true state. Status in one line: notebook 01 is DONE with results
(executed copy committed at notebooks/01_qwen_analysis_outputs.ipynb; numbers in the
table above); the user is about to run notebook 03 on a RunPod H100 via VS Code
Remote-SSH; notebook 02 is deferred pending a re-scope decision.

Likely first requests from the user:
1. Notebook 03 results / errors — debug as in sessions 3–4 (read raw generations, check
   the pod's results/ for stale files before trusting quick-resume, verify HF repo
   layouts locally with curl before trusting notebook fallbacks; you can run pod-side
   diagnostics yourself via `ssh -o BatchMode=yes runpod '...'`). Then fill the nb03
   rows of the results table.
2. The notebook 02 re-scope: if the user approves, edit scripts/build_nb02.py to add an
   EM-direction ceiling-capping condition (ActivationCapper mode="ceiling",
   em_direction_l24 from the LOCAL results/01_em_mean_diff_directions.pt — upload it to
   the pod's results/ first, threshold from base-model projection distribution), rebuild
   with `python scripts/build_nb02.py`, commit notebooks/, push, pull on the pod.
3. Presentation updates: extend presentation/em_results_10min.pptx (python-pptx in the
   local .venv; figures extractable from executed notebooks via base64 in cell outputs).
   User wants trait prompts + example outputs slides — see "Currently in progress" for
   exactly which results/repo files hold the quotes (nb01 result files are now in the
   LOCAL results/ folder, no Drive download needed).

Session-3 debugging lore worth keeping in mind (details in "Session 3 changes"):
steering coef = FRACTION of hidden norm; steer generated tokens only; balance mean-diff
sets per question; near-zero cosine vs TRAINED write vectors is expected (B-vector
mystery); quick-resume will resurrect stale buggy results unless deleted from the
results dir in use (Drive for old Colab runs, pod results/ for RunPod runs).

GPU runs happen on RunPod (NOT Colab anymore — read the "RunPod environment" section);
this agent edits code/notebooks/deck locally, pushes, and pulls on the pod via ssh.
Update this file at session end.
