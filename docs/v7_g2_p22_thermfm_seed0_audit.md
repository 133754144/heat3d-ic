# P22 Therm-FM seed0 completion audit

审计时间：2026-09-17（devbox）。seed0 已完成冻结的 200 epoch valid-only 训练；没有访问
`test_iid`、sealed 或 DeepOHeat official100。运行使用 `PDEFormer`、RTX 5070、PyTorch
2.9.0+cu128，训练 768 样本、20 个 batch/epoch、batch=40，共 4000 updates。

## 完整性结论

- history 为 200 行，epoch `1..200` 连续；每 epoch 均为 768 train samples / 20 train batches。
- history 最低 valid normalized loss 在 epoch 184（`0.007783112610923126`），与 completion receipt 一致。
- best/final/latest checkpoint、run config、history 和日志均已计算 SHA256；checkpoint reload state equal。
- 训练耗时 `4812.742 s`（约 80.21 min），峰值显存 allocated `5.095 GB`、reserved `6.451 GB`。
- best valid-only full-field：sample-first relative RMSE `2.5107%`，point-global `3.1424%`，RMSE `2.4955 K`，MAE `1.6027 K`。
- final valid-only full-field：sample-first relative RMSE `2.5098%`，point-global `3.1475%`，RMSE `2.4995 K`，MAE `1.6034 K`。

seed0 通过完整性审计，可作为 P22 seed cohort 的第一个 valid-only 成员。seed1、seed2 输出目录
当前不存在；后续仅在同一 execution/config/data 合同下串行启动，绝不覆盖 seed0。

机器可读审计见 `docs/v7_g2_p22_thermfm_seed0_audit.json`。
