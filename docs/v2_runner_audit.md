# Heat3D v2 runner 审计

## 范围

本审计面向 v2 训练系统设计前的 v1 runner / analysis 工具链梳理。目标是把现有 CLI 状态、输出文件和隐含默认值归纳成 v2 config schema 的输入材料。本轮不修改任何 runner 脚本，不接 Optax，不改模型，不跑训练。

## 现有入口列表

| 入口 | 用途 | 状态 |
|---|---|---|
| `scripts/run_heat3d_v1_medium_controlled_training_export.py` | v1 medium-style controlled training/export smoke；训练后可导出 final 和 best-valid recovered-temperature predictions。 | 训练 runner，v2 应迁移其配置结构，不直接改造。 |
| `scripts/compare_heat3d_v1_medium_baselines.py` | 对比 `zero_delta` baseline 与可选 trained predictions，输出 overall/split/condition metrics。 | 诊断入口，提供 comparison JSON。 |
| `scripts/analyze_heat3d_v1_medium_error_bins.py` | 基于 DeltaT percentile bins 分析 background bias / high-bin behavior。 | 诊断入口，提供 error bins JSON/MD。 |
| `scripts/analyze_heat3d_v1_medium_run_summary.py` | 汇总 `loss_summary.json`、baseline comparison 和可选 error bins，生成 run analysis JSON/MD。 | 诊断入口，适合作为 v2 report 聚合参考。 |
| `scripts/analyze_heat3d_v1_medium_condition_diagnostics.py` | 按 split/source/k/BC/q-power 等条件分组分析 signed bias、over/under prediction 和 bin_0 行为。 | 诊断入口，适合作为 v2 condition diagnostics 参考。 |

## 每个入口的用途

`run_heat3d_v1_medium_controlled_training_export.py` 负责加载 medium subset，构造 relative BC features + zero-delta bridge + normalized DeltaT target，使用当前 `MODEL_CONFIG` 初始化 RIGNO，执行 full-batch 手写梯度下降，并在 ignored `output/` 下写出 run config、loss summary 和可选 prediction archives。

`compare_heat3d_v1_medium_baselines.py` 负责从 subset 和 prediction archive 计算 `zero_delta` 与 trained prediction 的 recovered-T、DeltaT、max_abs、p95、peak、hotspot distance、top-k overlap 等指标，并按 split 与 condition metadata 聚合。

`analyze_heat3d_v1_medium_error_bins.py` 负责把全局 true DeltaT 按 percentile bins 划分，报告每个 bin 的 RMSE/MAE、signed bias、overprediction ratio、underprediction ratio 和相对变化。

`analyze_heat3d_v1_medium_run_summary.py` 负责把 loss trend、baseline comparison、error-bin 摘要和 condition summary 汇总为 `run_analysis*.json` 与 `run_analysis*.md`。

`analyze_heat3d_v1_medium_condition_diagnostics.py` 负责重新从 subset 与 prediction archive 计算 condition-wise diagnostics，特别是 low-DeltaT `bin_0` background overprediction。

## 主要 CLI 参数分组

### dataset 参数

- `--subset`: dataset/subset 路径。runner 默认是 `data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2`。
- compare/error-bin/condition diagnostics 也使用 `--subset`，其中 error-bin 和 condition diagnostics 要求显式提供。
- dataset loader 目前隐含使用 `Heat3DV1NativeSupervisedDataset(..., k_encoding_mode="diag3")`。
- split 来自 subset 内 `sample_meta.json`，runner 要求 train 和 valid 非空。

### model 参数

模型参数没有独立 CLI。runner 从 `scripts/check_heat3d_v1_small_train_valid_smoke.py` 导入 `MODEL_CONFIG`：

- `num_outputs: 1`
- `processor_steps: 2`
- `node_latent_size: 16`
- `edge_latent_size: 16`
- `mlp_hidden_layers: 1`
- `concatenate_tau: false`
- `concatenate_t: false`
- `conditioned_normalization: false`
- `cond_norm_hidden_size: 16`
- `p_edge_masking: 0.0`

这是 v2 最需要迁移到 config 的部分之一。v2 应通过配置测试 latent width、processor steps 和 MLP depth，而不是直接修改 `rigno/models/*`。

### optimizer/training 参数

- `--epochs`，默认 `5`。
- `--lr`，默认 `1e-5`。
- `--lr-schedule`，可选 `constant`、`warmup_cosine`、`two_stage`，默认 `constant`。
- `--warmup-epochs`，默认 `0`。
- `--min-lr`，默认 `1e-5`。
- `--second-stage-epoch`，默认 `0`。
- `--second-stage-lr`，默认 `1e-4`。
- `--seed`，默认 `0`。
- `--selection-metric`，可选 `valid_loss`、`valid_raw_deltaT_mse`、`valid_base_mse`，默认 `valid_loss`。
- `--report-every`，默认 `1`。

当前 optimizer 隐含为 `manual_full_batch_gradient_descent`，更新形式是 `param - lr_epoch * grad`。没有 Optax、Adam/AdamW、gradient clipping、weight decay、batching、checkpoint resume 或 eval-only path。

### loss 参数

- `--loss-mode`: `mse`、`background_hotspot`、`background_l1_bias`、`background_l1_relative`、`background_pseudo_negative`，默认 `mse`。
- `--background-quantile`，默认 `0.50`。
- `--hotspot-quantile`，默认 `0.90`。
- `--background-weight`，默认 `1.0`。
- `--hotspot-weight`，默认 `0.1`。
- `--background-l1-weight`，默认 `1.0`。
- `--background-bias-weight`，默认 `1.0`。
- `--background-over-weight`，默认 `1.0`。
- `--background-relative-weight`，默认 `0.0`。
- `--relative-floor`，默认 `0.02`。
- `--relative-floor-mode`: `fixed`、`p50`、`p75`，默认 `fixed`。
- `--pseudo-negative-quantile`，默认 `0.25`。
- `--pseudo-negative-delta-threshold`，默认 `null`。
- `--pseudo-negative-weight`，默认 `0.1`。
- `--pseudo-negative-over-margin`，默认 `0.0`。
- `--pseudo-negative-min-count`，默认 `1`。
- `--pseudo-negative-loss-type`: `mse`、`l1`、`relative_l1`、`relative_mse`，默认 `mse`。
- `--pseudo-negative-relative-floor`，默认 `0.02`。
- `--loss-weight-schedule`: `constant`、`two_phase`、`linear_anneal`，默认 `constant`。
- `--loss-transition-epoch`，默认 `0`。
- `--background-*-start/end`、`--hotspot-weight-start/end` 系列默认 `null`。

当前 loss 主要是 supervised output-space loss。v2 第一阶段仍应保持这个边界，不急于加入 PDE / BC / energy residual。

### export 参数

- `--output-dir`，默认 `output/heat3d_v1_medium_runs/export_smoke_seed0`，并强制位于 ignored `output/` 下。
- `--save-predictions`: 写 `predictions.npz`，默认 false。
- `--save-best-predictions`: 写 best-valid prediction archive，默认 false。
- `--best-predictions-name`，默认 `best_predictions.npz`，仅允许 output dir 下的文件名。
- runner 总是写 `run_config.json` 和 `loss_summary.json`。

### diagnostics 参数

- runner logging: `--log-mode` (`compact`/`full`/`quiet`)、`--progress-log`、`--no-progress-log`、`--progress-detail` (`off`/`basic`/`verbose`)。
- comparison: `--trained-predictions`、`--output-json`、`--top-k`、`--stdout-mode`。
- error bins: `--trained-predictions`、`--output-json`、`--output-md`、`--bins`、`--group-by`、`--stdout-mode`。
- run summary: `--run-dir`、`--loss-summary`、`--baseline-comparison-json`、`--error-bins-json`、`--prediction-label`、`--output-json`、`--output-md`、`--metric-set`、`--stdout-mode`。
- condition diagnostics: `--subset`、`--trained-predictions`、`--output-json`、`--output-md`、`--prediction-label`、`--bins`、`--q-power-bins`、`--stdout-mode`。

## 当前输出文件列表

Runner 输出：

- `<run-dir>/run_config.json`
- `<run-dir>/loss_summary.json`
- `<run-dir>/predictions.npz`，当 `--save-predictions` 启用
- `<run-dir>/best_predictions.npz` 或自定义 `--best-predictions-name`，当 `--save-best-predictions` 启用

Baseline comparison 输出：

- `<run-dir>/baseline_comparison.json`
- 常见 final/best 命名：`baseline_comparison_final.json`、`baseline_comparison_best.json`
- 当前 compare 脚本没有独立 `baseline_comparison.md` 输出；comparison 表格通常由后续 `run_analysis*.md` 渲染。

Error bins 输出：

- `<run-dir>/error_bins.json`
- `<run-dir>/error_bins.md`
- 常见 final/best 命名：`error_bins_final.json`、`error_bins_final.md`、`error_bins_best.json`、`error_bins_best.md`

Run analysis 输出：

- `<run-dir>/run_analysis.json`
- `<run-dir>/run_analysis.md`
- 当 `--prediction-label final|best` 使用默认路径时，输出 `run_analysis_final.json/md` 或 `run_analysis_best.json/md`

Condition diagnostics 输出：

- `<run-dir>/condition_diagnostics_final.json`
- `<run-dir>/condition_diagnostics_final.md`
- `<run-dir>/condition_diagnostics_best.json`
- `<run-dir>/condition_diagnostics_best.md`

这些文件应留在 ignored `output/` 下，除非后续另有明确 artifact 发布策略。

## 当前最影响可复现性的隐含默认值

- `MODEL_CONFIG` 由另一个 smoke script 导入，不在 runner CLI 中显式记录为可配置输入。
- `k_encoding_mode="diag3"` 固定在 loader 调用中。
- graph builder 使用默认 `Heat3DGraphBuilder()`，没有 config 化。
- optimizer 是手写 full-batch gradient descent，无 batch size、clip、weight decay、optimizer state、checkpoint resume。
- train-only normalization 固定在 runner 内部。
- default subset 指向 medium64-style `medium_v2`，但 v2 starting dataset 是 `medium1024_gapA_full1024_v2`。
- output 默认目录包含 `seed0`，但 seed 与 output dir 没有统一 run naming 规则。
- final prediction 和 best-valid prediction 由 CLI flag 控制，默认不会保存。
- comparison/error-bin/run-summary/condition diagnostics 是多个脚本串联，文件命名约定靠人工维护。
- final-vs-best 的 `prediction_label` 需要手动保持与输入 archive 和输出文件名一致。

## 适合迁移到 v2 config 的内容

- dataset 路径、dataset name、split policy、k encoding、sample limit / smoke mode。
- model capacity：latent width、edge latent、processor steps、MLP depth、normalization options。
- optimizer：Adam/AdamW、learning rate、schedule、warmup、clip norm、weight decay、seed。
- loss：loss mode、background/hotspot/pseudo-negative 参数、staged schedule。
- run：epochs、report cadence、log mode、smoke/controlled mode、local vs SSH policy。
- export：run dir template、save final predictions、save best predictions、best selection metric、archive names。
- diagnostics：comparison、error bins、condition diagnostics、top-k、p95/p99、field-shape diagnostics、final/best labels。
- baseline_reference：frozen v1 config、best epoch、reference metrics 和 non-claim flags。

## 暂时不要动的内容

- 不修改 `rigno/models/*`。
- 不修改 v0 public entrypoints。
- 不把 YAML 接入训练脚本，直到 schema review 完成。
- 不替换 optimizer 为 Optax，本轮仅记录 schema 草案。
- 不改变 loader/graph builder 行为。
- 不生成 full dataset，不跑长训练，不扩大分辨率。
- 不把 `medium1024_gapA_full1024_v2` 结果说成 formal benchmark。
