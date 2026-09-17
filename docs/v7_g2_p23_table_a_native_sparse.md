# Table A — P1i native sparse/common-task

本表冻结既有 valid-only native-1024 结果，不新增训练或评估。数值为三个训练
seed 的 mean ± **SD across training seeds**；不是 test/generalization 结果。

| 模型 | sample-first relative RMSE (%) | point-global relative RMSE (%) | 参数量 | checkpoint 规则 |
|---|---:|---:|---:|---|
| Heat3D V6 | 1.6294 ± 0.0131 | 2.0273 ± 0.0947 | 892,776 | 冻结 V6 point-global-best |
| GINO | 15.0139 ± 0.9130 | 17.7315 ± 1.6551 | 13,673,988 | valid sample-first 最优 |
| Transolver | 16.0028 ± 1.0310 | 18.5565 ± 0.9266 | 716,737 | valid sample-first 最优 |

Table A 的原始 receipts 为 P13/P11/P20 文件；P23 不重新选择 checkpoint。GINO/Transolver
只在 native-1024 域出现，不与 full-field transfer 表混排。
