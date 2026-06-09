# results/

Outputs from the Colab notebooks. On Colab these are written to
`/content/drive/MyDrive/tara_project/results/` so they persist across runtimes;
this directory mirrors that layout. Large/binary outputs (`*.pt`, `*.png`, `*.json`)
are git-ignored — only this README is tracked.

## File → notebook → figure/table mapping

| Results file | Produced by | Corresponds to |
|---|---|---|
| `00_baseline_em_rate.json` | `00_setup_and_sanity` | Baseline EM rates (EM organism vs base) |
| `01_judged_responses.json` | `01_em_organism_baseline` | Judged aligned/misaligned response sets |
| `01_em_mean_diff_directions.pt` | `01_em_organism_baseline` | EM mean-diff direction at all 48 layers |
| `01_steering_verification.{json,png}` | `01_em_organism_baseline` | EM rate vs steering coefficient (Soligo Fig. 1) |
| `01_direction_comparisons.json` | `01_em_organism_baseline` | Released-vector + LoRA-B cosines ("0.04 mystery") |
| `02_assistant_axis_qwen14b_l24.pt` | `02_checkpoint_monitoring` | Freshly-extracted Assistant Axis (Qwen2.5-14B, L24) |
| `02_monitoring/monitoring_log.jsonl` | `02_checkpoint_monitoring` | Per-checkpoint EM rate + projections |
| `02_monitoring_trajectories.png` | `02_checkpoint_monitoring` | EM rate vs projection probes over training |
| `02_lead_time_analysis.json` | `02_checkpoint_monitoring` | **Lead-time** of probes vs behavioural eval |
| `02_probe_roc.png`, `02_roc_auc.json` | `02_checkpoint_monitoring` | Probe ROC curves and AUCs |
| `03_capping_thresholds.json` | `03_capping_during_finetuning` | 25th-percentile axis-cap thresholds |
| `03_monitoring_{baseline,capped}/...` | `03_capping_during_finetuning` | Per-checkpoint logs for each training run |
| `03_em_trajectories.png` | `03_capping_during_finetuning` | EM rate vs step: baseline vs capped |
| `03_final_em_rates.json` | `03_capping_during_finetuning` | Final EM rates for all 3 conditions |
| `03_capability_check.json` | `03_capping_during_finetuning` | Benign coherence under each condition |
| `03_pareto_capping.json` | `03_capping_during_finetuning` | **Pareto**: harm reduction × capability |
| `04_gemma_directions.pt` | `04_assistant_axis_extraction` | Gemma EM direction + Assistant Axis |
| `04_trait_cosine_distribution.png` | `04_assistant_axis_extraction` | Trait-vector cosines vs the axis |
| `04_case_study_trajectories.{json,png}` | `04_assistant_axis_extraction` | Persona-drift case studies |
| `05_geometry_heatmap.png` | `05_unified_geometry_map` | **Centrepiece** 4×4 cosine heatmap |
| `05_geometry_results.json` | `05_unified_geometry_map` | Cosines, principal angle, R², causal test |
| `05_per_layer_cosine.png` | `05_unified_geometry_map` | EM–Axis cosine vs depth |
| `05_all_directions.pt` | `05_unified_geometry_map` | The four named directions (for notebook 06) |
| `06_robustness_results.json` | `06_adversarial_capping_robustness` | Harmful rates + capability per condition |
| `06_robustness_pareto.png` | `06_adversarial_capping_robustness` | **Pareto frontier**: static vs adaptive, single vs dual cap |

## The three success-criterion figures

1. **Geometry map** — `05_geometry_heatmap.png` (+ `05_geometry_results.json`).
2. **Lead-time plot** — `02_monitoring_trajectories.png` (+ `02_lead_time_analysis.json`).
3. **Pareto plots** — `03_pareto_capping.json` (training) and `06_robustness_pareto.png` (robustness).
