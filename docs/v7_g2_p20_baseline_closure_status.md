# V7 G2 P20 baseline closure status

状态：`P20_AUDIT_COMPLETE_VALID_ONLY_NO_NEW_TRAINING`（2026-09-16）。本轮完成 V6 P1i canonical provenance、Therm-FM/HCP compatibility 和 accuracy-first matrix；没有重训、没有新评估、没有读取 test/sealed。

## Full-resolution 默认策略

跨分辨率的正式 full-field 方案现冻结为 **`U-v2 direct-query dense inference`**：
以冻结的 1024 个 physics-layout-aware conditioning points，在
`101×101×56=571,256` 个 DeepOHeat-v1 官方查询坐标上直接查询。该路径不增加
learned parameter；端到端 latency 必须包含 query-graph construction、direct-query
forward 和必要的 dense postprocess。native-1024 仍作为独立 sparse view 报告。

IDW 只保留为 Heat3D V6/P1h 的 historical diagnostic，不是 full-resolution 默认方案，
也不进入 P18/P20 正式比较行。V6 P1i 的 240,825-node canonical evaluator 继续原样
冻结；本策略的适用范围是 DeepOHeat-v1 的 571,256-point cross-benchmark domain。
机器可读冻结文件为 [`g2_full_resolution_u_v2_policy.json`](../configs/heat3d_v7/g2_full_resolution_u_v2_policy.json)。

## 已闭合

- Heat3D P1i canonical：仅使用 V6_06/07/08 三 seed family；valid-only evaluator 与 checkpoint reload 证据通过，V7 e200/e600 不进入 P1i canonical。
- DeepOHeat-v1 native full：保留为 `NATIVE_REFERENCE_NOT_DIRECTLY_COMPARABLE`，因为 100,000-function training pool 与候选 valid128 重叠。
- P1i common-task matrix：Heat3D V6、GINO、Transolver 的 native-1024 valid-only rows 分开列示。

## 独立未闭合

- Therm-FM：已取得并核验 selective Poseidon-T（不下载大型 Therm-FM archive），状态更新为 `READY_FOR_P1I_FEASIBILITY`。P1i 的 `240,825-node full_fields.h5` 是 V6 P1i dense-label contract；它与 DeepOHeat-v1 的 `571,256-node` label cache 是两个不同数据源，不能混写。正式运行仍需 materialize train-only stats 和完成 pretraining-overlap audit。
- HCP：native source smoke 已通过，状态拆分为 `TRAINING_READY` 与 `OFFICIAL_ACCURACY_ASSETS_BLOCKED / REFERENCE_ONLY`。官方代码使用 inline structured uniform mesh、power/BC 配置；Git 仓库不含 benchmark data/checkpoint/raw outputs，且未证明原实现原生接收 P1i variable geometry/material/source/BC。

devbox 已恢复可访问；P20 时点只读 recheck 证明远端已有 768 train + 128 valid 的完整标签目录，
且当时未启动新任务、未生成本地大副本、未发现 Therm-FM/HCP 所需上游资产；P21 的
selective Poseidon-T 与 HCP native smoke 更新见 `docs/v7_g2_p21_*`。详见
[`v7_g2_p20_devbox_recheck.json`](v7_g2_p20_devbox_recheck.json)。

## 治理修正

V6 `test_iid=128` 已于历史路线中打开一次作为 corrected legacy confirmatory holdout，且未用于选择/调参；当前真正 untouched 的最终 holdout 是 `sealed IID`。今后 receipt 只声明本任务是否新访问 test_iid/sealed，不能将 test_iid 全局写成“未打开”。旧 receipt 不覆盖、不删除。

完整机器可读证据见：

- `docs/v7_g2_p20_p1i_heat3d_v6_canonical_audit.json`
- `docs/v7_g2_p20_thermfm_compatibility_audit.json`
- `docs/v7_g2_p20_hcp_compatibility_audit.json`
- `docs/v7_g2_p20_benchmark_matrix.json`
- `docs/v7_g2_p20_governance_amendment.json`

P21 独立 smoke、资产和 adapter 证据见 `docs/v7_g2_p21_*`。
