# Heat3D v2 frozen V1 reproduction runbook

## P1.5a 目标

本阶段只准备 frozen-v1-equivalent baseline reproduction 的 dry-run runbook。它使用现有 v2 config loader 和 config-to-command builder 生成命令计划，但不执行训练、不执行 diagnostics、不创建 `output/`。

## 使用配置

- run config: `configs/heat3d_v2/medium1024_gapA_controlled.yaml`
- reference config: `configs/heat3d_v2/frozen_v1_reference.yaml`

`medium1024_gapA_controlled.yaml` 当前是 v2 controlled 草案，其中 optimizer 目标是 `adamw` / `1.0e-3` / `warmup_cosine`。P1.5a 为了准备 frozen v1 reproduction，只在内存中把 `frozen_v1_reference.yaml` 的已确认训练字段覆盖到 command plan 上；不修改 YAML。

## frozen v1 baseline 关键配置

- dataset: `medium1024_gapA_full1024_v2`
- loss mode: `background_pseudo_negative`
- pseudo-negative loss type: `relative_l1`
- pseudo-negative weight: `0.10`
- background relative weight: `0.10`
- optimizer behavior: current v1 runner legacy manual full-batch update
- lr: `1.0e-2`
- lr schedule: `constant`
- best epoch: `33`

## training command

```bash
python3 scripts/run_heat3d_v1_medium_controlled_training_export.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 --epochs 100 --lr 0.01 --lr-schedule constant --warmup-epochs 0 --min-lr 1e-05 --seed 0 --output-dir output/heat3d_v2_runs/frozen_v1_equivalent_seed0 --save-predictions --save-best-predictions --best-predictions-name best_predictions.npz --report-every 5 --log-mode compact --progress-log --progress-detail basic --selection-metric valid_loss --loss-mode background_pseudo_negative --background-quantile 0.5 --hotspot-quantile 0.9 --background-weight 1.0 --hotspot-weight 0.1 --background-l1-weight 1.0 --background-bias-weight 1.0 --background-over-weight 1.0 --background-relative-weight 0.1 --relative-floor 0.02 --relative-floor-mode fixed --pseudo-negative-quantile 0.25 --pseudo-negative-weight 0.1 --pseudo-negative-over-margin 0.0 --pseudo-negative-min-count 1 --pseudo-negative-loss-type relative_l1 --pseudo-negative-relative-floor 0.02 --loss-weight-schedule constant --loss-transition-epoch 0
```

## prediction paths

- final: `output/heat3d_v2_runs/frozen_v1_equivalent_seed0/predictions.npz`
- best: `output/heat3d_v2_runs/frozen_v1_equivalent_seed0/best_predictions.npz`

## diagnostics command order

1. final baseline comparison
2. final error bins
3. final run summary
4. final condition diagnostics
5. final field-shape diagnostics
6. best baseline comparison
7. best error bins
8. best run summary
9. best condition diagnostics
10. best field-shape diagnostics

对应命令如下：

```bash
python3 scripts/compare_heat3d_v1_medium_baselines.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 --trained-predictions output/heat3d_v2_runs/frozen_v1_equivalent_seed0/predictions.npz --output-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/baseline_comparison_final.json --top-k 5 --stdout-mode compact
python3 scripts/analyze_heat3d_v1_medium_error_bins.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 --trained-predictions output/heat3d_v2_runs/frozen_v1_equivalent_seed0/predictions.npz --output-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/error_bins_final.json --output-md output/heat3d_v2_runs/frozen_v1_equivalent_seed0/error_bins_final.md --bins p50,p75,p90,p95 --stdout-mode compact
python3 scripts/analyze_heat3d_v1_medium_run_summary.py --run-dir output/heat3d_v2_runs/frozen_v1_equivalent_seed0 --loss-summary output/heat3d_v2_runs/frozen_v1_equivalent_seed0/loss_summary.json --baseline-comparison-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/baseline_comparison_final.json --error-bins-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/error_bins_final.json --prediction-label final --output-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/run_analysis_final.json --output-md output/heat3d_v2_runs/frozen_v1_equivalent_seed0/run_analysis_final.md --stdout-mode compact --metric-set mean_T_rmse mean_T_mae mean_DeltaT_rmse mean_max_abs mean_p95_abs mean_peak_T_err mean_hotspot_dist
python3 scripts/analyze_heat3d_v1_medium_condition_diagnostics.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 --trained-predictions output/heat3d_v2_runs/frozen_v1_equivalent_seed0/predictions.npz --output-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/condition_diagnostics_final.json --output-md output/heat3d_v2_runs/frozen_v1_equivalent_seed0/condition_diagnostics_final.md --prediction-label final --bins p50,p75,p90,p95 --q-power-bins p33,p66 --stdout-mode compact
python3 scripts/analyze_heat3d_v2_field_shape_diagnostics.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 --trained-predictions output/heat3d_v2_runs/frozen_v1_equivalent_seed0/predictions.npz --prediction-label final --output-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/field_shape_diagnostics_final.json --output-md output/heat3d_v2_runs/frozen_v1_equivalent_seed0/field_shape_diagnostics_final.md --top-k 5 --stdout-mode compact
python3 scripts/compare_heat3d_v1_medium_baselines.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 --trained-predictions output/heat3d_v2_runs/frozen_v1_equivalent_seed0/best_predictions.npz --output-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/baseline_comparison_best.json --top-k 5 --stdout-mode compact
python3 scripts/analyze_heat3d_v1_medium_error_bins.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 --trained-predictions output/heat3d_v2_runs/frozen_v1_equivalent_seed0/best_predictions.npz --output-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/error_bins_best.json --output-md output/heat3d_v2_runs/frozen_v1_equivalent_seed0/error_bins_best.md --bins p50,p75,p90,p95 --stdout-mode compact
python3 scripts/analyze_heat3d_v1_medium_run_summary.py --run-dir output/heat3d_v2_runs/frozen_v1_equivalent_seed0 --loss-summary output/heat3d_v2_runs/frozen_v1_equivalent_seed0/loss_summary.json --baseline-comparison-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/baseline_comparison_best.json --error-bins-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/error_bins_best.json --prediction-label best --output-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/run_analysis_best.json --output-md output/heat3d_v2_runs/frozen_v1_equivalent_seed0/run_analysis_best.md --stdout-mode compact --metric-set mean_T_rmse mean_T_mae mean_DeltaT_rmse mean_max_abs mean_p95_abs mean_peak_T_err mean_hotspot_dist
python3 scripts/analyze_heat3d_v1_medium_condition_diagnostics.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 --trained-predictions output/heat3d_v2_runs/frozen_v1_equivalent_seed0/best_predictions.npz --output-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/condition_diagnostics_best.json --output-md output/heat3d_v2_runs/frozen_v1_equivalent_seed0/condition_diagnostics_best.md --prediction-label best --bins p50,p75,p90,p95 --q-power-bins p33,p66 --stdout-mode compact
python3 scripts/analyze_heat3d_v2_field_shape_diagnostics.py --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 --trained-predictions output/heat3d_v2_runs/frozen_v1_equivalent_seed0/best_predictions.npz --prediction-label best --output-json output/heat3d_v2_runs/frozen_v1_equivalent_seed0/field_shape_diagnostics_best.json --output-md output/heat3d_v2_runs/frozen_v1_equivalent_seed0/field_shape_diagnostics_best.md --top-k 5 --stdout-mode compact
```

## mapped / unmapped / implicit 摘要

Mapped:

- dataset subset、runner epochs / log fields、lr / schedule / seed、loss 参数、prediction export、final/best diagnostics labels、final/best field-shape diagnostics command。

Unmapped:

- model capacity fields 仍不传给 v1 runner CLI；
- `p_quantiles` 仍是 v2 diagnostics 草案，field-shape CLI 当前固定输出 p95 / p99；
- `baseline_reference.path` 只用于 loader 校验和 runbook 说明；
- `run.device_policy` 只表达本地/SSH 策略。

Implicit:

- v1 runner 仍使用 legacy manual full-batch optimizer behavior；
- `dataset.k_encoding_mode=diag3` 仍由 v1 loader 隐含；
- `run_config.json` 和 `loss_summary.json` 仍由 v1 runner 隐式写入；
- reference `best_epoch=33` 是对比记录，不会自动改变训练轮数。

## 边界

本阶段只是 runbook 和 dry-run plan，不等于已经复现 v1 baseline。尚未执行训练、尚未执行 diagnostics、尚未验证 frozen reference 指标。

## 下一步

下一步是 push 后做 SSH preflight：确认服务器分支、环境、config loader 和 dry-run plan 可运行。再下一步才由用户确认是否实际运行 baseline reproduction。
