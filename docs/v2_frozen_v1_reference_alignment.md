# Heat3D v2 frozen V1 reference alignment

## 目标

P1.5c 的目标是把 v2 frozen-v1 reproduction 配置严格对齐到真实 V1 best diagnostic run：

`medium1024_gapA_full1024_v2_e050_pn_relative_l1_w0.10_bgrel0.10_seed0`

本轮只做配置、reference provenance 和只读 alignment check，不训练、不执行 diagnostics、不写 `output/`。

## 为什么 P1.5b 与 reference 不一致

P1.5b 使用的是早期 runbook 中的 `frozen_v1_equivalent_seed0` dry-run plan。该 plan 对齐了 subset、pseudo-negative loss、background relative loss、`lr=1e-2` 和 constant schedule，但还缺少真实 V1 best run 的完整 provenance。

已知差异：

- P1.5b 使用 `epochs=100`，真实 V1 best 是 `epochs=50`。
- P1.5b 沿用了 controlled config 中的 `hotspot_weight=0.1`，真实 V1 best 是 `hotspot_weight=0.02`。
- P1.5b 的 reference 文件当时没有明确 `seed=0`、`selection_metric=valid_loss`、`report_every=5`。

因此 P1.5b 是一次 useful reproduction attempt，但不是 strict reference alignment。

## strict config 如何对齐

新增 `configs/heat3d_v2/frozen_v1_best_e050_seed0.yaml`，显式写入真实 V1 best 参数：

- subset: `medium1024_gapA_full1024_v2`
- epochs: `50`
- lr: `1e-2`
- lr schedule: `constant`
- seed: `0`
- report every: `5`
- loss mode: `background_pseudo_negative`
- pseudo-negative loss type: `relative_l1`
- pseudo-negative weight: `0.10`
- background relative weight: `0.10`
- hotspot weight: `0.02`
- selection metric: `valid_loss`
- final/best prediction export enabled

输出目录为 `output/heat3d_v2_runs/frozen_v1_best_e050_seed0`。

## reference provenance

`configs/heat3d_v2/frozen_v1_reference.yaml` 现在记录：

- run name；
- epochs、seed、selection metric、report cadence；
- hotspot weight；
- confirmed reference metrics。

该 reference 仍是 diagnostic-stage historical V1 best，不是 formal benchmark。

## subset fingerprint 的作用

`scripts/check_heat3d_v2_frozen_v1_alignment.py` 会做轻量 fingerprint：

- subset path 是否存在；
- `sample_meta.json` 文件数量；
- split 计数；
- sample id 列表 hash；
- 抽样 sample 的 metadata / q / k / temperature 文件存在性和大小。

它能回答“当前本地/远程 checkout 是否指向同一个 subset 结构和 sample id 集合”。它不能证明数组数值完全相同，因为本轮不读取大数组内容。

## 下一步

下一步应先在 SSH 上做 strict e50 rerun，并生成 final/best diagnostics。只有 strict e50 run 与 reference provenance 对齐后，才适合决定是否进入 P2 field-shape diagnostics。
