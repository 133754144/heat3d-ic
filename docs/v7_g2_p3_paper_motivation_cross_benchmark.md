# V7 G2-P3 paper motivation and cross-benchmark preparation

状态：`LOCAL_PREPARATION_COMPLETE_NOT_LAUNCHED`。机器可读总合同为 `configs/heat3d_v7/g2_formal_preparation_protocol_v4.json`。本轮只在当前 Mac/G2 worktree 读取允许的 P1i train 与 `valid_iid`；没有 SSH/devbox、GPU、formal/长训练、`test_iid`/sealed，也没有下载 Therm-FM 24 GB checkpoint。

## GINO full geometry freeze

完整 768 train + 128 `valid_iid` coordinate-only audit 冻结 `r_in=0.15, r_out=0.033`、32³ latent grid。input GNO 的 zero-neighbor fraction 为 train 0.2386% / valid 0.2345%，query coverage 为 99.7614% / 99.7655%；output GNO zero-neighbor 为 0、query coverage 100%。asymmetric graph 总 edge 数 338,675,426，是 `(0.15,0.15)` 的 50.6003%。没有读取 target、loss 或 accuracy，因此半径不允许按 formal valid error 再调。完整分位数、source coverage 与 split edge counts 见 `docs/v7_g2_p3_gino_full_radius_receipt.json`。

train-only normalization 已跨 768×1024 points 以 float64/population `ddof=0` 物化：coordinate min/max、11 个 physical-feature mean/std 与 target global mean/std 都有逐数组 SHA256，core payload SHA256 为 `554ef44e093e60a2a45cff88e74d488a982fa69d1e227e9f7d43427cf3e0406a`。valid 不参与拟合，禁止 per-node-index target normalization。凭证见 `docs/v7_g2_p3_p1i_train_statistics.json`。

## P1i common-task freeze

正式 rows 固定为 Vanilla RIGNO、GINO、Transolver、Heat3D；全部只获得 `coords + 11 physical features`，不含 extra prior、solver、dense temperature field 或 learned adapter。GINO/Transolver immutable launch manifests 已准备但未授权、未启动；本地 qualification 分别为 `PASS_1_epoch_train_valid_checkpoint_reload` 与 `PASS_1_epoch_train_valid_checkpoint_bitwise_equal_reload`。Vanilla RIGNO/Heat3D 使用 G1 frozen formal artifacts，不根据正在运行的 G1 interim result 改 G2。

Primary checkpoint rule 对四个模型一致：每个 model×seed run 在 `valid_iid` 上选择最小 `sample_first_relative_rmse_pct`，并以 earliest epoch tie-break；同时保存 upstream final epoch 作 sensitivity。禁止跨模型/seed 选择、test/sealed 选择或看结果后更换 budget/metric。

## DeepOHeat-v1 data status

官方 Google Drive 的 7 个 `.npy` 文件已完整核对，共 8,710,901,856 bytes：surface 10,000 train inputs / 10 test inputs / 10 reference fields，volumetric 100,000 / 100 / 100。文件名、shape、dtype、byte size 与 SHA256 见 `docs/v7_g2_p3_deepoheat_v1_data_receipt.json`。

作者训练是 physics-informed，官方 code 不需要 train temperature labels；完整 train input functions 与 test reference fields 已足够按作者语义从头训练。因此状态升级为 `upstream_faithful_from_scratch_semiconductor_baseline_candidate`。没有 checkpoint 不再是 scientific blocker，但这还不是 original reproduction PASS；显式 license、dependency lock、生成 metadata/power scaling 与同硬件长训练仍未解决。

## Heat3D cross-benchmark boundary

在现有 `coords + k + q + BC` schema 中，可无损重表达：DeepOHeat `single_htc_bc`、`multi_htc_bc` 与 DeepOHeat-v1 volumetric（最后一项相对于 released discrete residual）。DeepOHeat `2d_power_map` 与 DeepOHeat-v1 surface 的 branch 是 top Neumann surface flux，不能冒充 volumetric q；在显式 deterministic flux-BC schema 资格验证前不可直接接入当前 11-channel Heat3D。转换器只物化数据，不求解 PDE、不训练模型。

即使 input representation lossless，Heat3D 是 supervised，官方 physics-informed train inputs 没有 train temperature truth；未来仍需用冻结的同 fidelity solver 生成 labels，单独记录 solver calls/cost。不同 benchmark 的 RMSE 不得同列排名，runtime 必须在同一硬件、精度、batch/query count 和 I/O boundary 下重新测量才可直接比较。

## Paper logic

`Why Heat3D` 的严格缺口固定为：**sparse irregular physical observations + explicit heterogeneous k/q/BC + geometry-aware operator + resolution-decoupled inference**。不得宣称“现有方法不支持 heterogeneous materials”。`Why RIGNO` 只给出 backbone 假设：G1 回答 Heat3D 新机制是否有效；G2 用实际结果回答 RIGNO 相对 GINO/Transolver 是否合理，不预设胜者。论文级 capability/evidence matrices 分别见 `docs/v7_g2_p3_why_heat3d_matrix.md` 与 `docs/v7_g2_p3_why_rigno_matrix.md`。

## Remaining gates after G1

远程 formal 前仍需新的明确授权、确认 G1 结束且目标 GPU 空闲、冻结远程 dependency/runner SHA、对 GINO/Transolver 做 one-batch GPU memory 与 epoch-walltime resource preflight，再启动预注册三 seeds。native Heat3D rows 另需生成 train reference temperatures；DeepOHeat-v1 需解决 license/dependency provenance。Therm-FM 仍是 transfer/data-efficiency track，DeepOHeat-v2 与 Geo-FNO 只保留 related-work/capability comparison。
