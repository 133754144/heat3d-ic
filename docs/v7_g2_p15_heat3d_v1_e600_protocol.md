# V7 G2-P15：Heat3D e600 convergence protocol

该协议在任何 e600 训练前冻结。它以现有 e200 Heat3D-on-DeepOHeat-v1 合同为基线，唯一
计划性变化是训练 horizon `200 → 600`；模型、loss、optimizer、B24、数据、normalization、
seed 与 valid-only checkpoint 规则不变。每个 seed fresh 初始化，禁止从 e200 resume。

## 验证和 checkpoint

每个 epoch 继续用 native-1024 `valid_iid sample_first_relative_rmse_pct` 选
`best-native`。昂贵的 U-v2 full-field evaluation 在训练前预注册为 epoch 200、400、600，
并在这三个点上选 `best-common-scheduled`；最终状态保留为 `final-e600`。U-v2 输出域固定为
DeepOHeat-v1 的 `101×101×56=571256` 点，指标在 `deltaT_K` 温度空间计算。

P14.1 的 IDW 与 U-v2 evaluator 只读 valid128，原始预测不入 Git。若 scheduled common metric
在 epoch600 仍改善，按预注册规则标记 `RIGHT_CENSORED_AT_600`，不延长训练。

## 禁止访问与 provenance

P1i `test_iid`、sealed、DeepOHeat official100 在训练和评估中均保持关闭。每个 seed 使用新的
远端输出目录 `output/heat3d_v7_g2/heat3d_on_deepoheat_v1_e600_p15/seed_<seed>`，不覆盖既有
e200 结果。receipt 必须保存 repo/runner/config/data/normalization/checkpoint hash、运行时间、
峰值显存、参数量、native/common/final 指标和 checkpoint reload 完整性。

完整机器可读字段见
[`g2_p15_heat3d_v1_e600_frozen_protocol.json`](../configs/heat3d_v7/g2_p15_heat3d_v1_e600_frozen_protocol.json)。
