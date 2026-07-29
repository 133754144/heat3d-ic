# V6-RandomBlock pilot128 v1 closeout

`pilot128_v1` 在 WSL2 完成 16 groups × 8 variants。128 个求解全部有限，
物理 gate、功率守恒、block/support coverage 和 group split 均通过；没有
训练、模型推理、过滤、替换或重采。

温升 gate 接近但未通过：peak ΔT 为 `43.652–159.862 K`，`123/128`
位于 30–150 K。bins 为
`{0:32, 1:32, 2:31, 3:28, outside:5}`：v5 有 1 个样本跨入 bin 3，
v6/v7 共 5 个样本超过 150 K。该尝试完整保留，不进入 formal generation。

随后仅对 v5/v6/v7 冻结一套全局候选：

- v5：`P=16.5 W, top/bottom h=2050/1000`;
- v6/v7：`P=12.5 W, top/bottom h=1000/500`。

候选在全部 16 个 pilot layouts 上统一重放，v5 的 16 个样本全部落入
bin 2，v6/v7 的 32 个样本全部落入 bin 3。校准没有逐样本反算、筛选或
替换；结果冻结在
`v6_randomblock_pilot128_v1_global_rule_calibration.json`，SHA256 为
`afcf5fd43faa349d3adbfb544f5ade93e1f89a9cab72403570732e52324f9517`。
