# Heat3D v2 config schema 草案

## 范围

本文件是 v2 config schema draft。它用于把 v1 runner 的 CLI 状态整理成结构化配置设计，当前不接入任何训练代码。所有字段都是草案，除非后续实现明确读取这些 YAML。

## 总体结构

建议 v2 config 顶层结构：

```yaml
schema_version: heat3d_v2_config_draft_v0
config_role: smoke | controlled | baseline_reference
non_claims: [...]
dataset: {...}
model: {...}
optimizer: {...}
loss: {...}
run: {...}
export: {...}
diagnostics: {...}
baseline_reference: {...}
```

## dataset section

| 字段 | 类型 | 默认建议 | 说明 |
|---|---|---|---|
| `name` | string | `medium1024_gapA_full1024_v2` | 数据集逻辑名。 |
| `subset_path` | string | `data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2` | 本地/远程相对 repo 的 subset 路径。 |
| `manifest_path` | string/null | `configs/heat3d_v1_physics_label_medium1024_gapA_manifest.json` | 可选 manifest 记录路径。 |
| `k_encoding_mode` | string | `diag3` | v1 runner 当前隐含值。 |
| `target` | string | `normalized_deltaT` | 训练目标语义。 |
| `recovery` | string | `T_ref_plus_deltaT` | 预测恢复温度的方式。 |
| `feature_view` | string | `relative_bc_features` | 当前推荐输入特征视图。 |
| `bridge` | string | `zero_delta_u_bridge` | 当前 legacy bridge。 |
| `split_source` | string | `sample_meta` | split 来自样本 metadata。 |
| `sample_limit` | int/null | `null` | smoke 可使用小值；controlled run 默认全 subset。 |
| `debug_only` | bool | `false` | medium256 / partial Gap-A 可标记为 debug。 |

## model section

| 字段 | 类型 | 默认建议 | 说明 |
|---|---|---|---|
| `architecture` | string | `RIGNO` | 当前模型族。 |
| `num_outputs` | int | `1` | 温度/DeltaT 单通道输出。 |
| `node_latent_size` | int | `64` | v2 建议从 v1 的 16 提升，先通过配置试验。 |
| `edge_latent_size` | int | `64` | 建议与 node latent 对齐。 |
| `processor_steps` | int | `4` | v2 建议测试 4/6。 |
| `mlp_hidden_layers` | int | `2` | v1 为 1，v2 草案建议加深。 |
| `concatenate_tau` | bool | `false` | 继承 v1 默认。 |
| `concatenate_t` | bool | `false` | 稳态任务默认 false。 |
| `conditioned_normalization` | bool | `false` | 暂不启用。 |
| `cond_norm_hidden_size` | int | `16` | 仅 conditioned normalization 使用。 |
| `p_edge_masking` | float | `0.0` | 继承 v1 默认。 |
| `report_parameter_count` | bool | `true` | v2 应报告模型容量。 |
| `report_memory_estimate` | bool | `true` | v2 应报告训练内存估计。 |

## optimizer section

| 字段 | 类型 | 默认建议 | 说明 |
|---|---|---|---|
| `name` | string | `adamw` | v2 目标是 Optax Adam/AdamW；当前草案未接代码。 |
| `lr` | float | `1.0e-3` | v2 初始建议；frozen v1 reference 使用 `1.0e-2`。 |
| `lr_schedule` | string | `warmup_cosine` | 可选 `constant`、`warmup_cosine`、`two_stage`。 |
| `warmup_epochs` | int | `5` | smoke 可为 0。 |
| `min_lr` | float | `1.0e-5` | schedule 下限。 |
| `second_stage_epoch` | int/null | `null` | two-stage schedule 使用。 |
| `second_stage_lr` | float/null | `null` | two-stage schedule 使用。 |
| `weight_decay` | float | `1.0e-4` | v2 草案字段。 |
| `gradient_clip_norm` | float/null | `1.0` | v2 草案字段。 |
| `seed` | int | `0` | 单 run seed。 |
| `multi_seed` | list[int] | `[]` | 本地不跑，多 seed 应在 SSH 上运行。 |

## loss section

| 字段 | 类型 | 默认建议 | 说明 |
|---|---|---|---|
| `mode` | string | `background_pseudo_negative` | v2 starting point 可沿用 frozen v1 reference。 |
| `base_space` | string | `normalized_deltaT` | v1 当前 base loss 空间。 |
| `background_quantile` | float | `0.50` | 背景 mask quantile。 |
| `hotspot_quantile` | float | `0.90` | hotspot mask quantile。 |
| `background_weight` | float | `1.0` | v1 参数。 |
| `hotspot_weight` | float | `0.1` | v1 默认，可后续 schedule 化。 |
| `background_l1_weight` | float | `1.0` | v1 参数。 |
| `background_bias_weight` | float | `1.0` | v1 参数。 |
| `background_over_weight` | float | `1.0` | v1 参数。 |
| `background_relative_weight` | float | `0.10` | frozen v1 reference 值。 |
| `relative_floor` | float | `0.02` | v1 默认。 |
| `relative_floor_mode` | string | `fixed` | v1 默认。 |
| `pseudo_negative_quantile` | float | `0.25` | v1 默认。 |
| `pseudo_negative_delta_threshold` | float/null | `null` | 未知/未启用时 null。 |
| `pseudo_negative_weight` | float | `0.10` | frozen v1 reference 值。 |
| `pseudo_negative_over_margin` | float | `0.0` | v1 默认。 |
| `pseudo_negative_min_count` | int | `1` | v1 默认。 |
| `pseudo_negative_loss_type` | string | `relative_l1` | frozen v1 reference 值。 |
| `pseudo_negative_relative_floor` | float | `0.02` | v1 默认。 |
| `weight_schedule` | string | `constant` | v2 可扩展 staged/curriculum。 |
| `transition_epoch` | int | `0` | schedule 使用。 |
| `draft_peak_loss_weight` | float/null | `null` | 草案字段，当前不接代码。 |
| `draft_field_shape_loss_weight` | float/null | `null` | 草案字段，当前不接代码。 |

## run section

| 字段 | 类型 | 默认建议 | 说明 |
|---|---|---|---|
| `mode` | string | `smoke` 或 `controlled` | 区分本地小检查和 SSH run。 |
| `epochs` | int | `2` smoke / `100` controlled | 本地只允许 very small smoke。 |
| `batching` | string | `full_batch_grouped` | v1 当前方式。 |
| `report_every` | int | `1` smoke / `5` controlled | 日志节奏。 |
| `log_mode` | string | `compact` | v1 默认。 |
| `progress_log` | bool | `true` | v1 默认。 |
| `progress_detail` | string | `basic` | v1 默认。 |
| `device_policy` | string | `local_small_only` | 复杂训练必须 SSH。 |
| `allow_long_training_local` | bool | `false` | 明确禁止本地长训练。 |

## export section

| 字段 | 类型 | 默认建议 | 说明 |
|---|---|---|---|
| `output_dir` | string | `output/heat3d_v2_runs/<run_name>` | 必须位于 ignored output。 |
| `run_name` | string | `smoke_minimal_seed0` | 建议统一命名。 |
| `save_final_predictions` | bool | `true` | v2 应保留 final export。 |
| `final_predictions_name` | string | `predictions.npz` | v1 final 文件名。 |
| `save_best_predictions` | bool | `true` | v2 应保留 best-valid export。 |
| `best_predictions_name` | string | `best_predictions.npz` | v1 best 文件名。 |
| `selection_metric` | string | `valid_loss` | 可选 `valid_raw_deltaT_mse` / `valid_base_mse`。 |
| `save_run_config` | bool | `true` | 写 run config。 |
| `save_loss_summary` | bool | `true` | 写 loss summary。 |

## diagnostics section

| 字段 | 类型 | 默认建议 | 说明 |
|---|---|---|---|
| `run_baseline_comparison` | bool | `true` | 生成 comparison JSON。 |
| `run_error_bins` | bool | `true` | 生成 error bins JSON/MD。 |
| `run_condition_diagnostics` | bool | `true` | 生成 condition diagnostics JSON/MD。 |
| `run_summary` | bool | `true` | 生成 run analysis JSON/MD。 |
| `prediction_labels` | list[string] | `[final, best]` | final/best 分开报告。 |
| `top_k` | int | `5` | hotspot overlap。 |
| `deltaT_bins` | string | `p50,p75,p90,p95` | v1 error-bin 默认。 |
| `q_power_bins` | string | `p33,p66` | condition diagnostics 默认。 |
| `metric_set` | list[string] | v1 medium metric set | run summary 使用。 |
| `field_shape_metrics` | list[string] | `field_variance_ratio`, `spatial_correlation`, `slice_level_rmse` | 草案字段，当前不接代码。 |
| `p_quantiles` | list[float] | `[0.95, 0.99]` | v2 p95/p99 diagnostics。 |

## baseline_reference section

| 字段 | 类型 | 默认建议 | 说明 |
|---|---|---|---|
| `name` | string | `frozen_v1_best_diagnostic` | frozen v1 reference 名称。 |
| `dataset` | string | `medium1024_gapA_full1024_v2` | reference dataset。 |
| `loss_mode` | string | `background_pseudo_negative` | reference loss。 |
| `pseudo_negative_loss_type` | string | `relative_l1` | reference loss type。 |
| `pseudo_negative_weight` | float | `0.10` | reference 值。 |
| `background_relative_weight` | float | `0.10` | reference 值。 |
| `lr` | float | `1.0e-2` | reference LR。 |
| `lr_schedule` | string | `constant` | reference schedule。 |
| `best_epoch` | int | `33` | reference best-valid epoch。 |
| `metrics` | object | 见 `frozen_v1_reference.yaml` | 只记录已确认指标，未知写 null。 |
| `non_claims` | list[string] | diagnostic non-claims | 防止夸大。 |

## 最小 smoke config 示例

```yaml
schema_version: heat3d_v2_config_draft_v0
config_role: smoke
dataset:
  name: medium256_debug
  subset_path: data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium256_v2
  k_encoding_mode: diag3
  sample_limit: 8
model:
  architecture: RIGNO
  node_latent_size: 16
  edge_latent_size: 16
  processor_steps: 2
  mlp_hidden_layers: 1
optimizer:
  name: manual_full_batch_gradient_descent
  lr: 1.0e-5
  lr_schedule: constant
  seed: 0
loss:
  mode: mse
run:
  mode: smoke
  epochs: 1
  allow_long_training_local: false
export:
  output_dir: output/heat3d_v2_runs/smoke_minimal_seed0
  save_final_predictions: false
  save_best_predictions: false
diagnostics:
  run_baseline_comparison: false
  run_error_bins: false
  run_condition_diagnostics: false
```

## medium1024_gapA_full1024_v2 controlled config 示例

```yaml
schema_version: heat3d_v2_config_draft_v0
config_role: controlled
dataset:
  name: medium1024_gapA_full1024_v2
  subset_path: data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2
  manifest_path: configs/heat3d_v1_physics_label_medium1024_gapA_manifest.json
  k_encoding_mode: diag3
model:
  architecture: RIGNO
  node_latent_size: 64
  edge_latent_size: 64
  processor_steps: 4
  mlp_hidden_layers: 2
optimizer:
  name: adamw
  lr: 1.0e-3
  lr_schedule: warmup_cosine
  warmup_epochs: 5
  min_lr: 1.0e-5
  weight_decay: 1.0e-4
  gradient_clip_norm: 1.0
  seed: 0
loss:
  mode: background_pseudo_negative
  pseudo_negative_loss_type: relative_l1
  pseudo_negative_weight: 0.10
  background_relative_weight: 0.10
run:
  mode: controlled
  epochs: 100
  device_policy: ssh_required_for_training
export:
  output_dir: output/heat3d_v2_runs/medium1024_gapA_controlled_seed0
  save_final_predictions: true
  save_best_predictions: true
diagnostics:
  prediction_labels: [final, best]
  run_baseline_comparison: true
  run_error_bins: true
  run_condition_diagnostics: true
  run_summary: true
baseline_reference:
  path: configs/heat3d_v2/frozen_v1_reference.yaml
```

## frozen v1 reference 记录方式

frozen v1 reference 应独立保存为 `configs/heat3d_v2/frozen_v1_reference.yaml`，并在每个 controlled config 中通过 `baseline_reference.path` 引用。只记录已有文档可确认的指标；未知字段写 `null`，并用注释说明不能编造。

## 当前只是草案、不接入代码的字段

- 所有 YAML 文件本轮均不被训练脚本读取。
- `optimizer.name=adamw`、`weight_decay`、`gradient_clip_norm` 尚未接 Optax。
- `field_shape_metrics`、`draft_peak_loss_weight`、`draft_field_shape_loss_weight` 尚未接 loss/diagnostics 实现。
- `sample_limit`、`device_policy`、`run_name`、`baseline_reference.path` 尚未接 runner。
- `report_parameter_count`、`report_memory_estimate` 尚未实现。
