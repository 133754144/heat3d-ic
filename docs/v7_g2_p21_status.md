# V7 G2 P21：Therm-FM / HCP competitor execution qualification

状态：`P21_COMPETITOR_EXECUTION_QUALIFICATION_COMPLETE_VALID_ONLY`（2026-09-16）。本轮仅在 devbox 做了独立环境审计和 bounded smoke，没有启动正式长训练；P1i `test_iid`、sealed IID、DeepOHeat official100 均未用于数值评估。G1 未修改，multi-HTC 未启动。DeepOHeat-v1 跨分辨率 full-field 的默认方案仍是 **U-v2 direct-query dense inference**。

## P20 amendment

- 三 seed 离散度的叙述统一为 **SD across training seeds**；旧 JSON 中的 `sample_sd` 字段名为兼容历史 schema 保留，不再解释为统计抽样假设。
- V6 P1i `full_fields.h5` 是 `240,825` 节点；DeepOHeat-v1 G2 volumetric label cache 是 `571,256` 节点。两者 provenance 分开，禁止混写。
- Therm-FM 状态推进为 `READY_FOR_P1I_FEASIBILITY`；HCP 拆分为 `TRAINING_READY` 与 `OFFICIAL_ACCURACY_ASSETS_BLOCKED`。

## Therm-FM（独立 transfer track）

官方来源冻结为 [Therm-FM repository](https://github.com/haiyangxin/Therm-FM) commit `1c338d0fbe0dca25311eb896a9ea136a4f3d3cb1`（Apache-2.0）。Poseidon-T 使用 pinned revision `93adcbf10f75b45ac3bca3939cc3d3f239e1e663`；config 与权重分别为 `940 B` / `83,390,397 B`，SHA 见 asset receipt，并已在本机和 devbox 双端核验。此次按用户要求再次以 HF CLI 单线程尝试本机下载，metadata 校验仍失败；没有替换已有完整副本，也没有留下可用 partial。此次失败不改变已验证资产状态，也不触发 Therm-FM 大型 archive 下载。

P1i dense adapter 在授权 train/valid fixture 上验证了共享网格严格为 `65×65×57`、C-order `x,y,z`。输入由 case definition 确定性栅格化为每层 13 channels（坐标、`k`、`q`、四个边界 mask、top/bottom `h`），flatten 后 741 channels；65×65 仅右/下 deterministic pad 到模型 128×128，不 resample、不用 `sparse1024 → interpolation`。这是一种与 Heat3D sparse support 不同且更 dense 的信息表达，不能声称 same-information 或 same-budget。

train fixture `v6p1if1_0000`、valid fixture `v6p1if1_0003` 均通过 pretrained load、`--replace_embedding_recovery` 语义、finite forward/backward、finite loss、valid forward 和 checkpoint reload。smoke target 是 synthetic finite unit target，仅验证闭环，不是 accuracy 证据。环境为 devbox PDEFormer（Python 3.12 系列、PyTorch 2.9.0、Transformers 4.57.1、Accelerate 1.11.0、RTX 5070），属于重建兼容环境而非官方 pinned environment。

当前结论：`READY_FOR_P1I_FEASIBILITY`；正式训练前还需 materialize train-only normalization、完成 pretraining overlap/data-leakage audit，并冻结独立 launch JSON。Therm-FM 仍定位为 `PRETRAINED_TRANSFER / FOUNDATION-MODEL BASELINE`，不进入 scratch common-task 表。

## HCP（独立 native/reference track）

官方来源冻结为 [HCP-enhanced DeepONet repository](https://github.com/LIMIGwenshao/HCP-enhanced-DeepONet) commit `c7d856f9b665e29b64d9faf931f86db1d6a1f26c`（MIT）。按官方 single-HTC native 配置抽取五个 fixed cuboidal domains、structured mesh/LHS、HCP projection、PDE/BC/interface loss，完成一 epoch forward/backward、Adam step、finite physics loss 和 checkpoint save/reload；75,042 参数，约 2.54 s/epoch。只增加了 Python 3.12 `imp` import shim；native 路径未调用的 `gstools` 以 import stub 隔离，未修改 projection、模型或 loss。

因此：

- native source 可运行：`TRAINING_READY`；
- 官方 checkpoint、raw outputs 与 benchmark data bundle 未随仓库提供：`OFFICIAL_ACCURACY_ASSETS_BLOCKED`；
- 原实现固定 structured cuboid、inline power/BC/material，不能原生接收 V7 variable-geometry/material/source/dual-Robin point cloud，因此 P1i direct adaptation 关闭；
- 在官方 bundle 到手前，HCP 只能作为 native reference-only track，不能宣称论文 accuracy 或加入 P1i 公平排行榜。

## Boundary and next action

本轮无 GPU formal training、无新数据集生成、无 test/sealed/official100 访问、无 multi-HTC、无 G1 变更。独立 receipt、protocol、adapter 与 smoke 日志路径均写入 JSON；大型模型/数据包未提交 Git。Therm-FM 的最小下一步是只用 train/valid 完成统计与 overlap audit；HCP 的最小下一步是取得官方 native bundle 并先做 native evaluation。详见 [`v7_g2_p21_status.json`](v7_g2_p21_status.json)。
