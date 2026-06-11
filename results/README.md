# results/

Outputs from the Colab notebooks. On Colab these are written to
`/content/drive/MyDrive/tara_project/results/` so they persist across runtimes;
this directory mirrors that layout. Large/binary outputs (`*.pt`, `*.png`, `*.json`)
are git-ignored — only this README is tracked.

Result filenames keep stage prefixes (the merged notebooks are composed from
per-stage builders). The notebook each file comes from:

- `01_qwen_analysis.ipynb` = stages 00 + 01 + 02 → produces all `00_*`, `01_*`, `02_*` files.
- `02_qwen_prompted_vs_finetuned.ipynb` = stages 07–10 → produces all `07_*`, `08_*`, `09_*`,
  `10_*` files. (The old `02_qwen_capping_finetune.ipynb` / stage 03 was dropped in session 5 —
  no `03_*` files were ever produced.)
- `03_gemma_geometry_robustness.ipynb` = stages 04 + 05 + 06 → produces all `04_*`, `05_*`,
  `06_*` files. (Stage 06 is the persona-space geometry; the original adversarial-capping
  stage 06 was removed before it ran.)
- `04_qwen_orthogonalized_finetune.ipynb` = stages 11–13 → produces all `11_*`, `12_*`,
  `13_*` files.

## File → notebook → figure/table mapping

| Results file | Notebook (stage) | Corresponds to |
|---|---|---|
| `00_baseline_em_rate.json` | `01_qwen_analysis` (00) | Baseline EM rates (EM organism vs base) |
| `01_judged_responses.json` | `01_qwen_analysis` (01) | Judged aligned/misaligned response sets |
| `01_em_mean_diff_directions.pt` | `01_qwen_analysis` (01) | EM mean-diff direction at all 48 layers |
| `01_steering_verification.{json,png}` | `01_qwen_analysis` (01) | EM rate vs steering coefficient (Soligo Fig. 1) |
| `01_direction_comparisons.json` | `01_qwen_analysis` (01) | Released-vector + LoRA-B cosines ("0.04 mystery") |
| `02_assistant_axis_qwen14b_l24.pt` | `01_qwen_analysis` (02) | Freshly-extracted Assistant Axis (Qwen2.5-14B, L24) |
| `02_monitoring/monitoring_log.jsonl` | `01_qwen_analysis` (02) | Per-checkpoint EM rate + projections |
| `02_monitoring_trajectories.png` | `01_qwen_analysis` (02) | EM rate vs projection probes over training |
| `02_lead_time_analysis.json` | `01_qwen_analysis` (02) | **Lead-time** of probes vs behavioural eval |
| `02_probe_roc.png`, `02_roc_auc.json` | `01_qwen_analysis` (02) | Probe ROC curves and AUCs |
| `07_prompted_direction.pt` | `02_qwen_prompted_vs_finetuned` (07) | Prompted contrastive EM direction, all 48 layers |
| `07_elicitation_responses.json` | `02_qwen_prompted_vs_finetuned` (07) | Judged elicitation responses (deck examples) |
| `07_direction_geometry.{json}`, `07_per_layer_cosines.png` | `02_qwen_prompted_vs_finetuned` (07) | **Prompted × fine-tuned × Axis** cosines + per-layer |
| `08_steering_comparison.{json,png}` | `02_qwen_prompted_vs_finetuned` (08) | Steering sweep: prompted vs fine-tuned direction |
| `09_qwen_trait_vectors.pt` | `02_qwen_prompted_vs_finetuned` (09) | 64 contrastive trait vectors (Qwen L24) |
| `09_persona_space.json`, `09_qwen_persona_spectrum.png`, `09_subspace_r2.png` | `02_qwen_prompted_vs_finetuned` (09) | Qwen persona PCA + R² for both EM directions |
| `10_training_trajectories.{json,png}` | `02_qwen_prompted_vs_finetuned` (10) | Checkpoint movement in persona coordinates |
| `04_gemma_directions.pt` | `03_gemma_geometry_robustness` (04) | Gemma EM direction + Assistant Axis |
| `04_trait_cosine_distribution.png` | `03_gemma_geometry_robustness` (04) | Trait-vector cosines vs the axis |
| `04_case_study_trajectories.{json,png}` | `03_gemma_geometry_robustness` (04) | Persona-drift case studies |
| `05_geometry_heatmap.png` | `03_gemma_geometry_robustness` (05) | **Centrepiece** 4×4 cosine heatmap |
| `05_geometry_results.json` | `03_gemma_geometry_robustness` (05) | Cosines, principal angle, R², causal test |
| `05_per_layer_cosine.png` | `03_gemma_geometry_robustness` (05) | EM–Axis cosine vs depth |
| `05_all_directions.pt` | `03_gemma_geometry_robustness` (05) | The four named directions (used by stage 06) |
| `06_persona_space_results.json` | `03_gemma_geometry_robustness` (06) | PCA stats, R² curves, nearest traits, PC loadings |
| `06_persona_spectrum.png` | `03_gemma_geometry_robustness` (06) | Persona-space variance spectrum (rank 8 @ 90%) |
| `06_em_subspace_r2.png` | `03_gemma_geometry_robustness` (06) | **EM-in-persona-subspace R²** vs rank |
| `11_orthogonalization_check.json` | `04_qwen_orthogonalized_finetune` (11) | Orthogonalization + damage-check numbers |
| `12_orthogonalized_training_log.json`, `12_orthogonalized_trajectories.png` | `04_qwen_orthogonalized_finetune` (12) | Constrained fine-tune: per-checkpoint EM + projections |
| `13_orthogonalized_em_eval.json`, `13_in_domain_check.json` | `04_qwen_orthogonalized_finetune` (13) | Final EM eval + in-domain task check |
| `13_new_em_direction.pt`, `13_new_direction_geometry.{json,png}` | `04_qwen_orthogonalized_finetune` (13) | v_new extraction + geometry |
| `13_new_direction_steering.{json,png}` | `04_qwen_orthogonalized_finetune` (13) | Causal check: steering a stock base along v_new |

## The key figures

1. **Geometry map** — `05_geometry_heatmap.png` (+ `05_geometry_results.json`).
2. **Lead-time plot** — `02_monitoring_trajectories.png` (+ `02_lead_time_analysis.json`).
3. **Persona-space R²** — `06_em_subspace_r2.png` (Gemma) and `09_subspace_r2.png` (Qwen, both directions).
4. **Direction geometry** — `07_per_layer_cosines.png` (prompted × fine-tuned × Axis on Qwen).
5. **Orthogonalized fine-tune** — `12_orthogonalized_trajectories.png` (constraint held; EM re-emerged).
