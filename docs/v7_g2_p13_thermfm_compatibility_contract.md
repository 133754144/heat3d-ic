# Therm-FM V7 compatibility contract

状态：`NEEDS_AMENDMENT`；track 固定为
`PRETRAINED_TRANSFER / FOUNDATION-MODEL BASELINE`。本阶段没有下载 24 GB
checkpoint archive、没有下载大型 benchmark，也没有访问 V7 test/sealed。

## Official source

- Paper: *Therm-FM: Foundation Model is ALL YOU NEED for 3D-ICs Thermal
  Simulation*, arXiv:2605.22663。
- Official repository: `haiyangxin/Therm-FM`。
- Frozen upstream commit: `1c338d0fbe0dca25311eb896a9ea136a4f3d3cb1`，license
  Apache-2.0。
- Existing evidence: official quick-demo loader/training/evaluation smoke passed
  on a tiny synthetic demo；这不是 V7 物理 accuracy 证据。

## Upstream contract

官方 `model_T` 是基于 Poseidon/scOT 的预训练 grid operator（约 21M 参数）。
steady benchmark loader 的输入是 `(N, P, L, H, W)`，其中通道编码 x/y/z 或
layer、power-density 等场；进入模型前按 layer-major flatten 成
`(N, L*P, H, W)`，输出为 `(N, L_out, H, W)` temperature field。已审计的
HS_SC release shape 为 input `[5000, 4, 2, 87, 87]`、output `[5000, 2, 87, 87]`。

正式 inference 至少需要 `config.json`、`pytorch_model.bin` 和
`normalization_constants.json`。normalization 是每个 flattened input channel
及 output layer 的 mean/std；缺少其中任一项不能声称 upstream-faithful evaluation。
官方 metric 由 release/evaluate 脚本定义，未来必须同时保留原生 grid metric 与
V7 common physical-domain metric，不能直接把不同 resolution 的 RMSE 排在同一列。

## V7 compatibility

V7 输入是 irregular point `coords + k + q + BC`，不是 Therm-FM 原生 dense grid。
因此不存在直接 drop-in：若未来执行，必须冻结 deterministic 的

1. geometry/material/BC metadata 到 grid 的 point-to-grid rasterization；
2. source/power channel 编码及 layer ordering；
3. grid prediction 到同一物理 query domain 的 grid-to-point/domain extraction；
4. train-only normalization constants 与单位换算。

这些转换不能加入额外 solver、temperature prior 或 learned encoder；转换后应
单独记录 input-information budget。若 V7 的 sparse BC/material 不可无损表达在
官方 channel contract 中，应记录 exclusion，而不是自行改造 Therm-FM。

## Provenance / resource / leakage

已知 release 将 `model_T`、`model_B`、`model_L` 及多个 benchmark 打包在约
24,100,784,430-byte 单体 checkpoint archive，steady dataset archive 约
4,467,245,278 bytes，当前未下载。预训练 Poseidon/scOT 数据与 V7 分布的潜在
重叠必须在任何结果前审计；在审计完成前不能把 transfer 结果解释为 same-budget
supervised baseline。由于缺少 selective `model_T` artifact 与 V7-compatible
benchmark manifest，本阶段不执行 pretrained valid-only evaluation。

## Minimum next step

先获得一个明确授权且可校验的 selective `model_T`、对应 config/normalization 和
一个不含 V7 test/sealed 的 benchmark；冻结上述 deterministic conversion 后，
在独立 environment 执行 valid-only evaluation。若无法获得 selective artifact 或
point-to-grid 表示不满足信息合同，Therm-FM 保留为 related-work/transfer
comparison，不进入 common-task 主表。

