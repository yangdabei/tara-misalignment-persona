# PROGRESS.md — Agent Handoff File
Last updated: 2026-06-11 (session 5 — nb03 RE-SCOPED: part 3 adversarial capping REMOVED,
replaced with persona-space geometry (PCA / subspace-R² / nearest-neighbour traits, runs on
CPU in seconds); nb02's 4–6 h capping-finetune DROPPED by the user for time — candidate
replacement is a cheap Qwen checkpoint-trajectory-through-persona-space analysis, not yet
implemented. nb03 part 1 (stage 04) already ran on the pod: trait histogram + case-study
trajectories downloaded to local results/.)
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
- NOTEBOOK 03 IS DONE (all three parts ran on the RunPod H100, 2026-06-11; all result
  files scp'd to the LOCAL results/ — see the results table for the headline numbers).
  Two run notes: (a) 04_gemma_directions.pt was never written on the pod (part 2 used
  kernel globals via the bridge), so the PER-LAYER EM directions are not persisted —
  only the four L22 directions in 05_all_directions.pt and the per-layer cosines (as
  numbers) in 05_geometry_results.json survive; re-run part 1's extraction cell if the
  per-layer tensors are ever needed. (b) The causal test ran BEFORE the
  responses-persisting commit (fadf54d), so the coef-0.8 EM-rate=0% cannot be
  diagnosed from saved text (presumed coherence collapse, same pattern as nb01).
- HEADLINE TENSION — RESOLVED BY nb02 v2 (2026-06-11): Gemma's PROMPTED EM direction
  was anti-Axis (−0.64) while Qwen's FINE-TUNED direction was Axis-orthogonal (−0.074);
  model and provenance were confounded. The v2 prompted direction on Qwen comes out
  anti-Axis too (−0.394, peak at L24) → **PROVENANCE story**: misalignment reached
  through a persona/role-play prompt is Axis-drift living in persona space; misalignment
  trained in by gradient descent is a different, nearly-orthogonal direction
  (cos(prompted, fine-tuned) = +0.18) that only weakly touches persona space. Magnitude
  difference (−0.64 Gemma vs −0.39 Qwen) plus elicitability (Gemma complies with a blunt
  evil prompt; Qwen only yields to fiction framing, 16/32) are the residual MODEL part
  of the story.
- NOTEBOOK 02 RAN (2026-06-11; executed copy at notebooks/02_qwen_em_provenance_outputs.ipynb,
  all 07_–10_ outputs in local results/, table rows filled). HEADLINE CAVEAT discovered in
  the outputs: Qwen2.5-14B-Instruct REFUSED the misaligned system prompt during the stage-07
  contrastive extraction — every one of the 24 "misaligned" responses in
  07_prompted_responses.json is an aligned refusal. So the "prompted EM direction" on Qwen
  is a prompt-resistance/compliance direction, not an EM direction (nearest traits obedient/
  humble/submissive .67; steering with it is causally inert, 0% EM at all coefs with
  coherence intact). The provenance-vs-model disambiguation therefore DID NOT RUN as
  designed — the notebook's printed "MODEL story" conclusion is unsupported; what the run
  actually shows is that Gemma complied with the persona prompt (its prompted dir steers to
  37.5% EM) while Qwen refused, itself a model difference in elicitability. Follow-ups
  sketched in the session-6 summary: (a) re-elicit on Qwen with stronger prompts + judge-
  filter the misaligned set before mean-diff (reuse nb01's per-question balancing);
  (b) save+judge Gemma's stage-04 extraction responses to confirm Gemma complied;
  (c) control for shared prompted-extraction variance in the R² analysis (prompted-EM and
  Axis R² curves coincide suspiciously); (d) explicit "refused-a-malicious-prompt"
  contrastive direction to confirm the confound interpretation.
- NOTEBOOK 02 RE-SCOPE IMPLEMENTED (session 5, user approved; ran 2026-06-11):
  notebooks/02_qwen_em_provenance.ipynb (old 02_qwen_capping_finetune.ipynb DELETED from
  the repo; rebuilt scripts/build_nb02.py is its source). Inference-only, one Qwen-14B
  load, ~2–3 h total, four parts: (07) prompted contrastive EM direction with the exact
  nb03 recipe + the 3-object geometry [prompted × fine-tuned × Axis — the provenance-vs-
  model disambiguation]; (08) steering verification of the prompted direction vs nb01's
  fine-tuned curve; (09) 64-trait contrastive basis on Qwen + subspace R²/nearest traits
  for BOTH EM provenances; (10) checkpoint trajectories (nb01 switcher) projected onto
  axis/both-EM-dirs/persona PCs. BEFORE RUNNING: start a new pod in US-NE-1 (volume!),
  git pull on the pod, pip reinstall (container wiped), AND upload the two nb01 inputs:
  `scp results/01_em_mean_diff_directions.pt results/02_assistant_axis_qwen14b_l24.pt
  runpod:/workspace/tara-misalignment-persona/results/` (part 1 asserts they exist).
- RESEARCH DIRECTION (session 5 discussion, for the write-up framing): "where does EM
  live in persona space?" — nb03's new part 3 is the map (subspace R², nearest traits =
  open-model check of OpenAI's toxic-persona finding); the mentor's proposed follow-up
  is orthogonalized finetuning. CLARIFIED (session 6): the mentor linked Arditi et al.'s
  weight-orthogonalization section — so the mechanism is W ← (I − r̂r̂ᵀ)W on all
  residual-stream writers + per-step LoRA-B projection, NOT activation capping. FULL
  PROTOCOL WRITTEN: docs/04_orthogonalized_finetuning_plan.md (hypotheses H1/H2/H3,
  arms, phases, kill criteria, budget ~4 h pod lean).
  NOTEBOOK 04 BUILT (session 6, phase 0 DONE): notebooks/04_qwen_orthogonalized_
  finetune.ipynb (source scripts/build_nb04.py, 40 cells; stages 11/12/13). New code:
  src/finetuning/orthogonalize.py (orthogonalize_writers: W←(I−r̂r̂ᵀ)W on embed_tokens
  + all o_proj/down_proj incl. PEFT base_layer variants, biases too, layer_range
  fallback; BOrthogonalizeCallback: per-step LoRA-B re-projection; both idempotent)
  + lora_trainer extended (dataset pinned to truthfulai/emergent_plus config "medical"
  — ModelOrganismsForEM/bad-medical-advice does NOT exist; prompt/misaligned→chat
  mapping in _tokenize_dataset; max_steps; use_orthogonalization wiring;
  model_and_tokenizer reuse param; EMCheckpointCallback extra_directions logged from
  ONE probe pass; log_history returned). tests/test_orthogonalize.py: 6 new tests,
  suite 20/20. Run config: per_device 4 × accum 20 = eff batch 80, max_steps 400
  (≈1 epoch ≈ organism's 396 steps), checkpoints+evals every 40 steps.
  PRE-RUN CHECKLIST (pod): git pull; pip reinstall if container reset; results/ on the
  volume already has the 4 required inputs (01/02/07v2/09 .pt files — assert cell
  checks); OPENROUTER_API_KEY needed from part 1; training checkpoints land on
  /workspace/checkpoints/em_orthogonalized; part 1 runs a 10-step SMOKE train then
  restores adapter init; watch the damage-check kill criterion (coherence < 0.78 →
  re-orthogonalize with layer_range=(13,38)). For disconnect-proofing run via
  papermill in tmux. Part 3's steering cell `del model`s — run it LAST.
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
- [ ] Notebook 03 on RunPod H100 (part 1 / stage 04 already ran on the pod; pull latest
      commit on the pod first): part 2 geometry heatmap, principal angle, causal test;
      NEW part 3 persona-space geometry (PCA spectrum + Axis≈PC1 check, EM subspace-R²
      with random baseline, nearest-neighbour traits, PC naming — CPU-only, needs
      05_all_directions.pt from part 2). Watch the Blockers items (axis file layout,
      toxic/refusal fallbacks). Stage-06 adversarial capping was REMOVED (session 5).
- [x] RUN notebook 02 — DONE 2026-06-11, outputs downloaded, table rows filled. BUT see
      the refusal confound in "Currently in progress": stage 07's prompted direction is
      not an EM direction; the provenance question remains OPEN pending re-elicitation.
- [x] RE-RUN notebook 02 with RECIPE v2 — DONE 2026-06-11 (executed copy at
      notebooks/02_qwen_em_provenance_v2.ipynb; all *_v2 results in local results/;
      v2 table rows filled). **PROVENANCE QUESTION ANSWERED — see v2 rows**: the
      roleplay-elicited prompted EM direction on Qwen is anti-Axis (−0.394, peaking at
      L24, mirroring Gemma's −0.64 shape), toxic-persona-adjacent (deceptive/callous/
      cruel/evil ≥.71), persona-subspace-resident (R² .72 @ rank 20), and causally
      potent (29.2% EM @ coef 0.8) — while the fine-tuned direction stays Axis-orthogonal
      (−0.074). cos(prompted, fine-tuned) = +0.18 only. Caveats for the write-up: single
      elicitation variant survived (roleplay 16/32; direct 0/32, fewshot 0/32 — so the
      per-variant robustness cosines were skipped), 16 balanced pairs (split-half 0.944
      mitigates), prompted steering needs 2× the coef and sits at coh 0.63. NOTE: the
      pod-side 08_prompted_steering_v2.png lacked the nb01 fine-tuned overlay
      (01_steering_verification.json wasn't uploaded to the pod) — regenerated LOCALLY
      with all three curves; the local png is the deck version.
      scripts/build_nb02.py rewritten: stage 07 now uses 3 elicitation variants (direct /
      roleplay "VEX" / fewshot seeded with 2 real nb01 organism quotes), judges ALL
      responses (score_alignment/score_coherence, keep = is_em_response), balances per
      question (cap 6), asserts yield ≥16 pairs across ≥4 questions (else it raises with
      the compliance table — a 0-yield is itself the finding), and adds per-variant
      direction cosines (elicitation-robustness check). ALL direction-dependent outputs
      version-bumped to *_v2 filenames (07_*, 08_*, 09_qwen_persona_space, 09 R² png,
      10_*) so quick-resume CANNOT resurrect the v1 results on the pod; v1 files stay
      put for the write-up, and 09_qwen_trait_vectors.pt keeps its name and is REUSED
      (~45 min saved). Part 2 plot overlays the v1 flat-0% curve. PRE-RUN CHECKLIST:
      pod in US-NE-1 + git pull + pip reinstall + scp the two nb01 .pt inputs (same as
      before) + OPENROUTER_API_KEY is now needed from PART 1 (judging moved up).
- [ ] Update the deck with nb03 results + trait prompt/output examples + the
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
| EM direction vs. Assistant Axis cosine (layer 22, Gemma-2-27B; PROMPTED contrastive EM dir) | **−0.640** (principal angle 50.2°; NOT orthogonal — sharp contrast with the Qwen ORGANISM's −0.074; peak anti-alignment layers 17–23, fades to −0.11 by L45) | 03_gemma_geometry_robustness (05) |
| 4×4 cosine heatmap (EM / Axis / toxic / refusal, Gemma L22) | cos(EM,toxic)=+0.40, cos(Axis,toxic)=−0.53, cos(EM,refusal)=−0.33; EM 41% explained by span(Axis,toxic) | 03_gemma_geometry_robustness (05) |
| Causal: steer base Gemma along EM dir → EM rate / Axis projection | coef 0.4: EM 37.5%, axis proj 9927→7905 (drops, mechanistically linked); coef 0.8: EM 0% (likely coherence collapse as in nb01; raw responses not persisted this run) | 03_gemma_geometry_robustness (05) |
| Persona-space PCA (507 trait vectors, Gemma L22) | effective rank 8 @ 90% var; cos(PC1, Axis)=0.64, centered PC1 0.74 — Axis ≈ but ≠ PC1 | 03_gemma_geometry_robustness (06) |
| EM-in-persona-subspace R² (rank 5/10/20/200; random ≈ k/d) | 0.56 / 0.84 / 0.91 / 0.97 (random 0.001–0.04); with Axis projected out: 0.35 / 0.74 / 0.82 — EM lives almost entirely IN persona space, well beyond the Axis | 03_gemma_geometry_robustness (06) |
| Nearest traits to the EM direction (toxic-persona replication) | arrogant .72, obsessive .71, dogmatic .67, elitist .66, **evil .65**, absolutist .64, cruel .63; most anti-EM: deferential −.74, factual −.73, humble −.69 | 03_gemma_geometry_robustness (06) |
| Lead time: EM direction probe vs. behavioural eval (steps) | 100 (probe fires @ step 150, behaviour ≥5% @ step 250; real checkpoints, steps 0–396) | 01_qwen_analysis (02) |
| Probe ROC AUC, per-checkpoint EM≥5% (11 checkpoints) | EM-direction 1.00; Assistant-Axis 0.67 (axis "lead 240" = noise false-positive @ step 10) | 01_qwen_analysis (02) |
| Capping-training EM rate reduction vs. baseline (%) | DROPPED (session 5 — no time for the 4–6 h training; see nb02 re-scope candidate) | 02_qwen_capping_finetune (03) |
| Provenance test: cos(prompted EM, Axis) / cos(fine-tuned EM, Axis) / cos(prompted, fine-tuned), Qwen L24 | −0.047 / −0.074 / −0.024 — all ≈ orthogonal at every layer (per-layer \|cos\| ≤ 0.12). **CONFOUND: Qwen REFUSED the misaligned system prompt — all 24 "misaligned" extraction responses are refusals (07_prompted_responses.json), so the prompted direction encodes prompt-resistance, NOT EM; the "model story" conclusion printed by the notebook is NOT supported** | 02_qwen_em_provenance (07) |
| Split-half cosine of the prompted direction (L24) | 0.584 (≈ fine-tuned's 0.647 — stable estimate, just of the wrong thing) | 02_qwen_em_provenance (07) |
| Steering base Qwen along the PROMPTED direction | EM rate 0% at ALL coefs 0–0.8 with coherence INTACT (~0.82–0.89) — causally inert (vs fine-tuned dir 29.2% @ 0.4 in nb01; Gemma prompted dir 37.5% @ 0.4 in nb03) | 02_qwen_em_provenance (08) |
| Qwen persona-space PCA (64 trait vectors, L24) | effective rank 17 @ 90% var (PC1 only 51%); cos(PC1, Axis) +0.48, centered −0.26, cos(mean trait, Axis) −0.475 — much less Axis-dominated than Gemma (0.64/0.74, rank 8 of 507) | 02_qwen_em_provenance (09) |
| EM-in-persona-subspace R², Qwen (rank 5/10/20/64; random ≤ .013) | prompted .48/.53/.62/.70 (tracks the Axis curve .49/.56/.59/.66 almost exactly — shared prompted-extraction variance); fine-tuned .23/.31/.40/.49 — well above random but only half in trait span | 02_qwen_em_provenance (09) |
| Nearest traits, Qwen (toxic-persona replication) | FINE-TUNED: absolutist .44, dogmatic .40, militant .31, evil .28 (weaker echo of Gemma's arrogant/dogmatic/evil); PROMPTED: obedient .67, humble .67, submissive .66 — refusal-flavoured, smoking gun for the confound | 02_qwen_em_provenance (09) |
| Checkpoint trajectories Δproj vs step 0 (L24, steps 0–396) | EM fine-tuned +4.7 peak @ 250 (matches probe lead-time); persona PC1 −6.5 (largest mover); Axis −2.3; prompted EM +3.3 (rises despite orthogonality/inertness — unexplained) | 02_qwen_em_provenance (10) |
| **v2** elicitation compliance (judged-EM per variant, base Qwen) | direct 0/32, **roleplay 16/32**, fewshot-with-real-organism-examples 0/32 — fiction framing is the ONLY door in (Gemma complied with the blunt prompt) | 02_qwen_em_provenance v2 (07) |
| **v2** three-object geometry (L24): cos(prompted, Axis) / cos(ft, Axis) / cos(prompted, ft) | **−0.394** / −0.074 / **+0.176**; prompted-vs-Axis anti-alignment PEAKS exactly at L24 (mirrors Gemma's −0.64 mid-layer peak shape); split-half 0.944 (16 pairs) → **PROVENANCE story**: prompted misalignment is Axis-drift, fine-tuned EM is not; cos(p,ft) 0.18 contradicts Soligo's "similar directions" | 02_qwen_em_provenance v2 (07) |
| **v2** steering with the prompted direction (base Qwen) | 0% @ ≤0.4, 8.3% @ 0.6, **29.2% @ 0.8** (align 0.31, coh 0.63 — above the 0.5 floor; responses persisted). Causally potent, needs 2× the fine-tuned coef (29.2% @ 0.4) | 02_qwen_em_provenance v2 (08) |
| **v2** nearest traits to prompted EM (Qwen) | deceptive .78, callous .78, cynical .77, cruel .76, vindictive .75, **evil .71** (farthest: factual −0.00) — full toxic-persona match, much tighter than fine-tuned's (.44 max) | 02_qwen_em_provenance v2 (09) |
| **v2** EM-in-persona-subspace R² (rank 20/64) | prompted **.72/.78** (Axis out: .59/.69) vs fine-tuned .40/.49 — prompted EM lives in persona space well beyond the Axis; fine-tuned only half | 02_qwen_em_provenance v2 (09) |
| **v2** checkpoint trajectories Δproj @ step 250 | prompted EM **+6.6** (bigger than fine-tuned's +4.7 along its own direction!), Axis −2.3, PC1 −6.5 — training moves the organism along the roleplay-evil direction too | 02_qwen_em_provenance v2 (10) |

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
- Shah et al. jailbreak dataset — OBSOLETE (session 5): stage 06 adversarial capping was
  removed from notebook 03, nothing uses the dataset now. data/jailbreak_prompts/ README
  kept for history.

## Repo layout (what's tracked vs. local-only)
Tracked & on GitHub:
- README.md, PROGRESS.md, requirements.txt, .env.example, .gitignore
- data/README.md, data/jailbreak_prompts/README.md, data/eval_questions/*.json, results/README.md
- src/helpers/{model_utils,hook_utils,generation_utils,judge_utils}.py
- src/directions/{mean_diff,assistant_axis,geometry}.py
- src/monitoring/checkpoint_monitor.py, src/finetuning/lora_trainer.py
- tests/{test_directions,test_hooks}.py
- notebooks/{01_qwen_analysis, 02_qwen_em_provenance, 03_gemma_geometry_robustness}.ipynb
  (02_qwen_capping_finetune.ipynb deleted in session 5 — replaced by 02_qwen_em_provenance)
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
