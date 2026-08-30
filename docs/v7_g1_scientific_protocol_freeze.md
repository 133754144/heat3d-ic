# V7 G1 scientific protocol freeze

当前协议是 `scientific_protocol_frozen_pre_G1`；正式 G1 multi-seed 仍未启动。
本文件冻结科学比较所需的 variants、support 语义、seed contract、e200 budget
和统计分析，不把 non-publication qualification 当作科学结果。

## 历史动力学与预算

直接相关的 V6 Full 参考为 `V6_06_V5best_P1i_seed0_reliable_B24`：e600
warmup-cosine、base LR `5e-4`、min LR `5e-5`、warmup 10、AdamW、weight
decay `1e-4`、gradient clipping `1.0`、B24；point-global 最佳 epoch 559，
sample-first 最佳 epoch 572。seed1/seed2 的曲线未定位，因此不推断其最佳 epoch。

不能把 e600 的前 200 个 epoch 当成 e200 收敛证据：cosine horizon=600 在
epoch 200 仍处于中段，而 cosine horizon=200 在 epoch 200 已到达 min LR，
两者的 learning-rate trajectory 不同。新的 e200 budget qualification 使用
完整新 schedule、随机初始化、seed 0；Full 最佳 epoch 173，Vanilla 最佳
epoch 164，均无边界 undertraining signal，故冻结 e200。qualification 仍是
`budget_qualification_only`，不进入正式 G1 evidence。

## Frozen model, seed and support contract

所有正式 variants 共享 V6/P1i Full parent 的 model/loss/normalization/graph/
batch contract。seed contract 只有一个：
`model=optimizer=batch_build=batch_order=graph=run_seed`，seed set 为
`{0,1,2}`，B24/B32 和 `sample_first_relative_rmse_pct`、earliest-tie checkpoint
selection 不变。

历史 `retry_deterministic_geometry_only_v1` 的 native support 是
source-layout-aware，而不是 source-amplitude-aware：`local_regions` 是 block
quota 别名；selector 使用 q/k block layout masks、geometry、layer boundaries、
control-volume weights、group ID 和 seed，不使用 numeric q、temperature、labels、
solver output 或 model error。H2 因此只允许 source-layout attribution claim。

正式 support attribution 的两个新 provider 已注册为 geometry-only、label-
independent definitions：

- `generic_stratified_v2`：保留 Full 的 boundary/interface/surface/CV coverage，
  block quota `0`；interface `128`、top `64`、bottom `64`、CV-weighted interior
  volume `768`，固定顺序为 interface/top/bottom/volume；移除 q/k
  block-layout-aware quota，不读取 q/k layout masks。
- `cv_only_v1`：CV-weighted interior volume `1024`；block/interface/top/bottom
  quota 全为 `0`。

两者的算法、输入排除项、stratum order、seed binding 和 support-index SHA 均由
[support provider contract](../configs/heat3d_v7/v7_g1_support_provider_contract.json)
冻结。它们已完成 seed0、1-epoch、CUDA、train/valid_iid-only 的 bounded
non-publication qualification。

No-FiLM 不是旧的 zero-context ablation。它只有一个 model delta：
`global_context_mode: film -> none`；冻结的 train-standardized 24-D physical
context tensor、native shape-scale、scale-context path、q/k features 和 decoder
bypass 都保留。`physics_scale_only` 保留 physics scale，只关闭 learned scale
residual correction，不退化为 direct-output architecture。

## Frozen formal matrix and hypotheses

正式矩阵为以下 7 variants × 3 seeds × 200 epochs = 21 runs：

1. Full Heat3D
2. canonical Vanilla RIGNO
3. capacity-matched Vanilla RIGNO
4. layout-agnostic stratified support
5. CV-only support
6. No-FiLM
7. physics-scale-only

Full 与 Vanilla 原始参数差距 7.4486%，超过 5% trigger；capacity-matched
Vanilla 使用 node/edge latent width 100。所有 variants 只通过同一个 registry-
driven entrypoint，variant 只表达相对 Full parent 的 delta。

统计 preregistration v3 冻结：

- H1 Full vs canonical Vanilla 的 primary 为 `point_global_relative_rmse_pct`；
- H1b Full vs capacity-matched Vanilla 的 primary 相同；
- H2 Full vs layout/CV support 的 common-domain primary 为 `source_region_RMSE_K`；
- H3 Full vs No-FiLM 的 primary 为 `sample_first_relative_rmse_pct`；
- H4 Full vs physics-scale-only 的 primary 为 `raw_K_CV_RMSE_K`；
- 报告每 seed、mean ± sample std、paired sample effects、median/P90/P95，以及
  128 个 valid_iid 中预注册的 worst-10；使用两层 bootstrap 95% CI；n=3 seed 的
  p-value 不是主要证据；禁止 posthoc 删除 seed、切换 metric 或调整 budget。

完整机器可读 preregistration 及唯一 SHA 见
[v7_g1_statistical_preregistration.json](../configs/heat3d_v7/v7_g1_statistical_preregistration.json)。

## Qualification boundary

generic/CV/No-FiLM 三个新 provider/delta 的 1-epoch receipts、旧 native
physics-scale-only 与 width-100 capacity control 的可追溯 lineage 见
[variant qualification receipt](v7_g1_variant_qualification_receipt.json)。这些
运行的 `publication_evidence`、`scientific_evidence_eligible` 和 `g1_formal`
均为 false；它们用于 implementation/provider qualification，不用于 headline
metric、model selection 或正式 G1 统计。

Full V6↔V7 semantic anchor 与同步 profiling receipt 仍保留；任何历史 V1–V6
runner、smoke、development 或 monkey-patch path 都只是 read-only historical
oracle，不属于 V7 production training graph。V6 frozen evidence、test_iid、
sealed、solver 和正式 G1 multi-seed 均未被触碰。

因此，在当前资格证据完整且所有 hard checks 通过时，控制面可标记
`V7_G1_SCIENTIFIC_READY=PASS`；这只表示协议已冻结并可供后续明确授权的正式
实验使用，不会自动启动 G1。
