# V7 G2 P20-A：V6 P1i canonical Heat3D audit

状态：`PASS_AUDIT_VALID_ONLY_THREE_SEED_RELOAD_EVIDENCE`。本轮没有重训，也没有新打开任何测试角色；内容复用远端 Git 已关闭的 V6 valid-only 三 seed evaluator/重载证据。

分辨率治理补充：DeepOHeat-v1 `101×101×56=571,256` cross-benchmark full-field 的默认跨分辨率策略为 **U-v2 direct-query dense inference**。这不回写 V6 P1i 的冻结 `240,825`-node canonical evaluator；后者继续使用既有 V6 语义。策略文件为 [`g2_full_resolution_u_v2_policy.json`](../configs/heat3d_v7/g2_full_resolution_u_v2_policy.json)。

## Canonical identity

P1i 主结果只使用 `V6_06_V5best_P1i_seed0_reliable_B24`、`V6_07...seed1`、`V6_08...seed2`。V7 e200/e600 是后续 cross-benchmark/运行研究，不替代 V6 P1i canonical family。三份配置的 SHA、训练 commit、数据 manifest、split manifest、full-field sidecar 和 checkpoint SHA 见同名 JSON。

V6 contract 是 768 train / 128 `valid_iid` / 128 `test_iid`，1024 source-aware support，full sidecar 240,825 nodes；RIGNO latent 96、6 processor steps、B24、validation B32、AdamW 5e-4、wd 1e-4、10-epoch warmup cosine、600 epochs，保留四项冻结 normalized-ΔT objective。三 seed 仅同步 seed/model/batch-order/graph seed，不改变科学配置。

## Valid-only unified evaluator

V6 closeout 已用同一 evaluator 对 support 与 full-field temperature-space 视图重算，并由三个独立进程加载 best/sample-first/base/final/latest；reload PASS。当前报告的 full-field primary（V6 冻结 `point_global_best`）为：

| seed | point-global best epoch | full point-global [%] | full sample-first [%] | full raw CV RMSE [K] | checkpoint SHA256 |
|---:|---:|---:|---:|---:|---|
| 0 | 559 | 3.459884 | 3.964290 | 2.985701 | `51567afe…b90e` |
| 1 | 455 | 3.490488 | 3.963523 | 2.970243 | `71971579…e71f` |
| 2 | 587 | 3.377505 | 3.947065 | 2.947569 | `d67e0dac…2ab49` |
| mean ± SD across training seeds | — | **3.442626 ± 0.058435** | **3.958293 ± 0.009731** | **2.967838 ± 0.019180** | — |

按 P20 要求，`sample-first relative RMSE` 同时作为当前 native-1024 primary view 报告：support 为 `1.629402 ± 0.013132%`，full-field 为 `3.958293 ± 0.009731%`。这里的 sample-first 数值来自同一 V6 family 的既有 evaluator rows；V6 canonical checkpoint identity 仍保留 point-global-best 选择语义，避免把不同历史选择规则混写。

## Native-reference comparison boundary

DeepOHeat-v1 seed42 native full-pool receipt、checkpoint 与训练 recipe 已冻结在 `configs/heat3d_v7/g2_semiconductor_remote_launch_manifest.json`（native 100,000 input functions、full 101×101×56 physics mesh、Optax Adam 1e-3、100,000 iterations × 50 sampled function instances，final-only checkpoint）。但其训练池包含当前 128 个候选 valid rows，因此不能在这些 valid rows 上形成 held-out native accuracy ranking。P20-A 将该行保留为 `NATIVE_REFERENCE / NOT_DIRECTLY_COMPARABLE`，不把 DeepOHeat physics loss 与 Heat3D relative RMSE 直接比较。

相同 valid physical cases 上可直接对齐的 accuracy 证据应使用已经冻结的 `DeepOHeat-768` 与 Heat3D-768 cohort；这只能称 `SAME_PHYSICAL_CASE_BUDGET`/data-regime comparison，不能称 `SAME_INFORMATION_BUDGET`。

## Provenance boundary

本轮 devbox 只读探测因 SSH timeout 未执行新 evaluator；上述数字来自不可变的 `origin/research/v6-p1i-training` closeout blobs（source commit `ade9a7c…`）。没有访问 test labels、sealed IID 或 DeepOHeat official100，也没有修改 G1。
