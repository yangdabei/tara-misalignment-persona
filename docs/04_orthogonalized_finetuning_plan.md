# Experiment plan — Notebook 04: Orthogonalized fine-tuning

**Question.** Is the known EM direction the *unique* road to emergent misalignment, or does
training re-route around a blocked direction? We hard-forbid the fine-tuned EM direction
during EM training — Arditi et al.'s weight orthogonalization
([Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction#Feature_ablation_via_weight_orthogonalization),
§ *Feature ablation via weight orthogonalization*) plus a per-step LoRA-B projection — then
fine-tune on the EM dataset and watch what happens.

**Why it matters.** nb01 showed a probe on this direction gives a 100-step early warning
(AUC 1.00). That defense assumes EM *must* travel this direction. This experiment tests that
assumption directly.

## Hypotheses (pre-registered readouts)

- **H1 — direction is necessary:** EM rate stays ≈0% while in-domain loss still falls.
  → single-direction monitoring/ablation is a strong defense; EM emergence is mediated by
  this one direction.
- **H2 — re-routing:** EM emerges at a comparable rate via a new direction
  (cos(v_new, v_forbidden) low). → EM is a direction *family*; single-direction probes are
  evadable; characterize the new road.
- **H3 — partial:** EM emerges late and/or attenuated. → report lead-time shift and final
  rate vs control; intermediate cosines (0.3–0.6) reported as partial re-routing, not
  binarized.

## Design

Two arms on Qwen2.5-14B-Instruct, rank-1 9-adapter LoRA (Soligo et al. config already in
`src/finetuning/lora_trainer.py`):

| Arm | Model | Constraint |
|---|---|---|
| A (control) | public organism `R1_3_3_3_full_train` + its 88 checkpoints | none (already analysed in nb01/nb02) |
| B (orthogonalized) | base weights orthogonalized vs v_EM, then fine-tuned | base can't write v_EM; LoRA-B re-projected ⊥ v_EM after every optimizer step |
| A′ (optional) | own fine-tune, identical config, no constraint | none — removes the hyperparameter-mismatch caveat on A for ~+2.5–3 h GPU |

**The forbidden direction:** v_EM = `01_em_mean_diff_directions.pt` layer 24, unit-normalized.
Single direction orthogonalized against **all residual-stream writers** (Arditi-faithful):
`embed_tokens` (rows: W ← W(I − r̂r̂ᵀ)), every `self_attn.o_proj` and `mlp.down_proj`
(output side: W ← (I − r̂r̂ᵀ)W). Justified empirically: the L24 direction is coherent across
layers (cos ≥ 0.7 for L19–30, ≥ 0.5 for L13–38). Qwen uses RoPE — no learned positional
embedding to treat. LoRA-B vectors (on `down_proj`, output side d_model) get
b ← (I − r̂r̂ᵀ)b after each step; that closes the loophole where gradient descent re-learns
the forbidden direction *through the adapters*.

**Dataset:** the configured `ModelOrganismsForEM/bad-medical-advice` does **not** exist on HF
(checked 2026-06-11). Candidate: `truthfulai/emergent_plus`, config `medical` (32,642 rows,
public). Pin id + schema in phase 0; mirror the organism's ~396 optimizer steps via effective
batch size rather than epochs.

## Steps

### Phase 0 — build (local, no GPU, ~1 session)
1. Pin dataset id; verify chat schema; adapt `_tokenize_dataset` if needed.
2. `src/finetuning/orthogonalize.py`: `orthogonalize_writers(model, direction)` (matrix
   orientations as above) + `BOrthogonalizeCallback` (post-step projection).
3. Unit tests: (a) after orthogonalization, |r̂ᵀ h| ≈ 0 for residual activations on random
   inputs at every layer; (b) B stays ⊥ r̂ across a real optimizer step; (c) round-trip save/load.
4. Build `notebooks/04_qwen_orthogonalized_finetune.ipynb` (stages 11/12/13) via
   `scripts/build_nb04.py`.

### Phase 1 — pre-flight (pod, ~30 min)
5. 10-step smoke train: loss decreases, B-orthogonality holds, checkpoint saves.
6. **Orthogonalization damage check** on the modified base (before any training):
   coherence + alignment judges on the 8 eval questions (n=24), compare to stock base
   (nb01: align 0.88/coh 0.88); response-activation projections onto v_EM ≈ 0 at L24.
   **Kill criterion:** coherence drop > 0.10 → fall back to orthogonalizing only the
   L13–38 writer band; re-check.

### Phase 2 — training arm B (pod, ~2–2.5 h incl. monitoring)
7. Train with the organism-mirroring config; save checkpoints at the nb01-matched steps
   {0, 1, 5, 10, 20, 40, 80, 120, 160, 200, 250, 300, 396}.
8. Monitor every ~40 steps (CheckpointMonitor): training loss; mean response-activation
   projections onto v_EM (constraint verification — must stay ≈0), Assistant Axis,
   prompted-v2 direction, persona PC1/PC2; cheap EM eval (8 q × 2, judged).

### Phase 3 — post-training analysis (pod ~1–1.5 h, then local)
9. Final EM rate, nb01 stage-00 protocol (n = 40+) vs control 12.5%.
10. **If EM ≥ 5% (H2/H3):** re-extract v_new (judge-filtered, per-question-balanced
    mean-diff — the nb01/v2 recipe); split-half reliability; then the headline number
    **cos(v_new, v_EM)** plus cos vs Axis / prompted-v2, persona-subspace R² (64-trait
    basis from `09_qwen_trait_vectors.pt`), nearest traits. Causal check: steer stock base
    along v_new, coef 0–0.8 sweep. Lead-time of a v_new probe across arm-B checkpoints
    vs nb01's 100 steps.
11. **If EM < 5% (H1):** verify the model still learned the *narrow* task (in-domain
    bad-medical eval + loss curve vs arm A) so "no EM" ≠ "no learning"; report final loss
    gap; probe projections over training as the mechanism story.
12. Results table rows (stages 11–13), deck slide, PROGRESS update.

## Budget

| Item | Time | Cost @ $3.29/h |
|---|---|---|
| Phase 0 build (local) | 1 session | — |
| Phases 1–3, lean (arms A+B) | ~4 h pod | ~$13 |
| Optional arm A′ | +2.5–3 h | +$9 |

## Known risks / caveats
- Control-arm hyperparameter mismatch (public organism's exact config not fully documented)
  — mitigated by A′ if needed.
- Single-direction choice: v_EM extracted *from the final organism*; a re-found direction
  with intermediate cosine is partial re-routing — report the spectrum.
- Orthogonalization could degrade general behaviour (Arditi et al. saw minimal damage for
  the refusal direction; must re-verify for v_EM — phase 1 kill criterion).
- Dataset id/schema unverified until phase 0.
