# V7-G2 P23-R2：Heat3D e200 与 Therm-FM 同输出分辨率推理协议

本协议冻结一个 inference-only diagnostic：将已经冻结的 V7 e200 三个
Heat3D checkpoint 应用于 P1i `valid_iid=128`，以 U-v2 的 native-1024
conditioning 直接查询 P1i 的完整 `65×65×57=240,825` 网格。Therm-FM 的
P1i 输出也是该网格，因此结果可以回答“是否有同输出分辨率的推理结果”；
它不是同域训练或 same-information-budget 比较。

执行前后均禁止训练、checkpoint 重选、`test_iid`、sealed IID 和
DeepOHeat official100。输入只由 P1i case definition 的 coordinates、材料、
热源和 Robin/边界字段确定性构造，不使用温度作为输入、不插值 1024 点、
不添加 learned adapter、不 padding 或 resample 65×65 网格。U-v2 算法和
DeepOHeat-v1 e200 的 train-only normalization 不变；由于 P1i 几何尺度与
DeepOHeat-v1 训练域不同，结果必须标为 zero-shot cross-benchmark transfer
diagnostic，不能写进同域公平排行榜。

每个 seed 单独输出 valid-only receipt，记录 checkpoint SHA、runner/config/
数据 SHA、U-v2 图审计、有限性和统一 full-field metric。聚合时使用
`mean ± SD across three training seeds`；不把 3×128 行当独立重复。

机器可读冻结合同见
`configs/heat3d_v7/g2_p23_r2_heat3d_e200_p1i_u_v2_240825_protocol.json`，
runner 为
`scripts/evaluate_v7_g2_p23_r2_heat3d_e200_p1i_u_v2_240825.py`。
