# results/

Outputs from the Colab notebooks. On Colab these are written to
`/content/drive/MyDrive/tara_project/results/` so they persist across runtimes;
this directory mirrors that layout. Large/binary outputs (`*.pt`, `*.png`, `*.json`)
are git-ignored — only this README is tracked.

Result filenames keep their original `00_`–`06_` stage prefixes (the merged notebooks
are composed from the same per-stage builders). The notebook each file comes from:

- `01_qwen_analysis.ipynb` = stages 00 + 01 + 02 → produces all `00_*`, `01_*`, `02_*` files.
- `02_qwen_capping_finetune.ipynb` = stage 03 → produces all `03_*` files.
- `03_gemma_geometry_robustness.ipynb` = stages 04 + 05 + 06 → produces all `04_*`, `05_*`, `06_*` files.

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
| `03_capping_thresholds.json` | `02_qwen_capping_finetune` | 25th-percentile axis-cap thresholds |
| `03_monitoring_{baseline,capped}/...` | `02_qwen_capping_finetune` | Per-checkpoint logs for each training run |
| `03_em_trajectories.png` | `02_qwen_capping_finetune` | EM rate vs step: baseline vs capped |
| `03_final_em_rates.json` | `02_qwen_capping_finetune` | Final EM rates for all 3 conditions |
| `03_capability_check.json` | `02_qwen_capping_finetune` | Benign coherence under each condition |
| `03_pareto_capping.json` | `02_qwen_capping_finetune` | **Pareto**: harm reduction × capability |
| `04_gemma_directions.pt` | `03_gemma_geometry_robustness` (04) | Gemma EM direction + Assistant Axis |
| `04_trait_cosine_distribution.png` | `03_gemma_geometry_robustness` (04) | Trait-vector cosines vs the axis |
| `04_case_study_trajectories.{json,png}` | `03_gemma_geometry_robustness` (04) | Persona-drift case studies |
| `05_geometry_heatmap.png` | `03_gemma_geometry_robustness` (05) | **Centrepiece** 4×4 cosine heatmap |
| `05_geometry_results.json` | `03_gemma_geometry_robustness` (05) | Cosines, principal angle, R², causal test |
| `05_per_layer_cosine.png` | `03_gemma_geometry_robustness` (05) | EM–Axis cosine vs depth |
| `05_all_directions.pt` | `03_gemma_geometry_robustness` (05) | The four named directions (used by stage 06) |
| `06_robustness_results.json` | `03_gemma_geometry_robustness` (06) | Harmful rates + capability per condition |
| `06_robustness_pareto.png` | `03_gemma_geometry_robustness` (06) | **Pareto frontier**: static vs adaptive, single vs dual cap |

## The three success-criterion figures

1. **Geometry map** — `05_geometry_heatmap.png` (+ `05_geometry_results.json`).
2. **Lead-time plot** — `02_monitoring_trajectories.png` (+ `02_lead_time_analysis.json`).
3. **Pareto plots** — `03_pareto_capping.json` (training) and `06_robustness_pareto.png` (robustness).
