# Heat3D v2 config-to-command dry-run

## 本轮目标

本轮完成 P1：把 v2 YAML config 转成现有 v1 runner / diagnostics 脚本的 dry-run command plan。它只生成和打印命令，不执行训练、不执行 diagnostics、不写 `output/`。

## 解决的问题

P0 已经能读取和校验 YAML，但还没有说明这些字段如何落到现有 v1 CLI。P1 的作用是把 v2 config 与 v1 runner 参数之间的关系显式化，让后续 P1.5 准备 frozen V1 baseline reproduction runbook 时，不再依赖手写长命令。

## 当前生成的命令

`rigno/heat3d_v2_runner_command.py` 生成：

- v1 controlled training/export command；
- final / best baseline comparison command；
- final / best error-bin command；
- final / best run summary command；
- final / best condition diagnostics command。

内部 command 保存为 `list[str]`，打印时使用 `shlex.join`。生成命令本身没有副作用。

## 已映射字段

当前已映射：

- `dataset.subset_path` 到 `--subset`；
- `run.epochs`、`report_every`、`log_mode`、`progress_log`、`progress_detail` 到 runner 日志和运行参数；
- `optimizer.lr`、`lr_schedule`、`warmup_epochs`、`min_lr`、`second_stage_epoch`、`second_stage_lr`、`seed` 到 v1 runner CLI；
- `loss.mode` 以及 background / hotspot / pseudo-negative / relative / schedule 参数到 v1 runner loss CLI；
- `export.output_dir`、`save_final_predictions`、`save_best_predictions`、`best_predictions_name`、`selection_metric` 到 v1 runner export CLI；
- `diagnostics.top_k`、`deltaT_bins`、`q_power_bins`、`prediction_labels`、`metric_set` 到对应 diagnostics 命令。

## 未映射字段

以下字段会进入 `unmapped_fields` 或 `warnings`，不会被假装生效：

- model capacity：`node_latent_size`、`edge_latent_size`、`processor_steps`、`mlp_hidden_layers`、parameter/memory report；
- optimizer 草案字段：`optimizer.name=adamw/adam`、`weight_decay`、`gradient_clip_norm`、`multi_seed`；
- v2 diagnostics 草案：`field_shape_metrics`、`p_quantiles`；
- `baseline_reference.path`，当前只作为 config validation / 说明，不传给 v1 runner；
- `dataset.k_encoding_mode`，当前仍是 v1 loader 的隐含 `diag3`；
- `dataset.sample_limit`，当前 v1 runner 无 sample-limit CLI；
- `run.device_policy`，只表达本地/SSH 策略，不传给 runner；
- `export.save_run_config` 和 `save_loss_summary`，v1 runner 当前总是写。

## 为什么这些字段暂时不生效

`optimizer.name=adamw` 需要后续 P3 接 Optax；model capacity 需要训练 runner 支持通过配置覆盖 `MODEL_CONFIG`；`field_shape_metrics` 需要 P2 只读 diagnostics 实现。P1 只负责把当前 v2 YAML 与已有 v1 CLI 对齐，不改变 runner 行为。

## 边界

P1 完成后还不能声称已经复现 frozen V1 baseline。当前只是 dry-run command plan，尚未执行训练、未执行 diagnostics，也未验证 frozen reference 指标。

## 下一步建议

下一步进入 P1.5：用 v2 config / wrapper 准备 frozen-v1-equivalent baseline reproduction runbook，仍先保持 dry-run 和短 smoke 边界。
