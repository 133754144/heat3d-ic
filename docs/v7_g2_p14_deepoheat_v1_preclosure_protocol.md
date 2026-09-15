# V7 G2-P14 DeepOHeat-v1 pre-closure

状态：`PROTOCOL_FROZEN_MATCHED_COHORT_QUEUED_SERIAL_GPU`。

本轮把 DeepOHeat-v1 分成两个不混用的比较层级。所有证据只使用
`fs_train_volume.npy`、冻结的 768 train/128 valid case IDs 以及已生成的
train/valid reference labels；P1i `test_iid`、sealed、DeepOHeat official100
均保持锁定，G1 未修改。

## A. method-native reference

DeepOHeat-v1 seed42 已按作者的 100000-function physics-informed protocol
完成：上游 commit `3ef3d9c41666a56b5940b39a61166ccaa5aaedb2`，101×101×56
全 mesh、50 functions/iteration、Optax Adam `1e-3`、`0.9/1000` 指数衰减，
最终 checkpoint 与 training receipt 的 SHA 见
`v7_g2_p14_deepoheat_v1_native_reference_audit.json`。Heat3D 的 200e 三 seed
冻结汇总见 `v7_g2_p13_heat3d_v1_3seed_valid_only_aggregation.json`。

映射审计确认：Heat3D 的 128 valid source IDs 和每行 canonical
`source_input_sha256` 都与官方 `fs_train_volume.npy` 一致，但 native seed42
使用了这个 100000 行池，而且官方 release 没有 valid split。因此这 128 行
全部是 native train/eval overlap。虽然 train/valid label cache 提供了同一
101×101×56 温度域，直接把 native checkpoint 在这些行上评分会产生泄漏，故
native-reference 层级为 **`NOT_DIRECTLY_COMPARABLE`**，不计算共同 valid 分数，
也不解锁 official100。DeepOHeat 的 physics loss 不与 Heat3D 的
temperature-space relative RMSE 直接比较。

## B. same physical-case budget

机器可读合同为
`configs/heat3d_v7/g2_deepoheat_v1_same_physical_case_budget_contract.json`。
唯一计划性变化是把训练 source pool 从官方 100000 functions 改为冻结的
768 train IDs；保持上游 `DeepOHeat_v1` 架构（3,303,680 参数）、完整
PDE/Robin/adiabatic BC、101×101×56 mesh、50 functions/iteration、100000
iterations、Optax Adam 与指数衰减、JAX `random.choice(replace=False)` 顺序。
这是 **`SAME_PHYSICAL_CASE_BUDGET`**，不是 `SAME_INFORMATION_BUDGET`：
DeepOHeat 仍使用完整 101×101 input function 与 full-mesh physics objective，
Heat3D 使用 1024 sparse `coords+k+q+BC` supervised observations。

训练前冻结：case IDs、seed `0/1/2`、上游验证无关的 PDE 语义、valid-only
temperature-space evaluator、每 10000 iterations 的 validation cadence、
最小 valid `sample_first_relative_rmse_pct` 及 earliest tie-break。每个 GPU
任务按 seed0→seed1→seed2 串行、fresh start；任何非空 output directory 都拒绝
覆盖。训练池不读温度 labels，valid labels 仅用于评估/选模。

每个 matched run 的 receipt 要记录：runner/upstream/data SHA，参数量，
PDE/collocation 预算（`50 × 100000 = 5,000,000` function evaluations），
best/final iteration 与温度指标、runtime/peak memory、checkpoint SHA 及
reload integrity。最终汇总只在 valid 上计算 mean ± sample SD、best→final
差异和异常标记。

## 停止条件与问题回答

- Q1（200e small-data Heat3D 对 frozen native DeepOHeat 的竞争力）：当前
  **不可验证**，因为 native candidate 没有非重叠 valid split；不得以重叠
  valid 分数替代公平证据。
- Q2（相同 768/128 physical-case budget）：待 matched DeepOHeat 三 seed
  完成后，用完全相同的 128 valid full-field temperature evaluator 比较；报告
  precision 与 PDE/solver/训练代价，不把它写成相同信息预算。

训练完成后更新 taxonomy 并停止，不启动 Heat3D 600e、Therm-FM、multi-HTC、
任何 test/sealed/official100 evaluation。
