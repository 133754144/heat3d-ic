# V7 G2 P20-B：Therm-FM compatibility/data audit

状态：`NEEDS_AMENDMENT`；资产决策：`BLOCKED_BY_UPSTREAM_ASSETS`。本轮只做接口和 provenance 审计，没有训练、没有下载 24 GB checkpoint archive，也没有访问任何 test/sealed。

## P1i labels

V6 tracked full-field sidecar 的 `status=complete`、`sample_count=1024`、`solver_node_count=240825`，并为每个样本记录 temperature/deltaT SHA；角色计数为 train 768、valid_iid 128、test_iid 128。因此已有 manifest 足以证明 train768 具有冻结的 240,825-node dense-label contract。压缩 archive bytes 未挂载到本 worktree，本轮没有读 labels、生成 labels 或触碰 test role；详情与 SHA 见 JSON。

## Interface

Therm-FM 官方 `model_T` 是约 21M 参数的 Poseidon/scOT dense-grid operator：输入 `(N,P,L,H,W)`，按 layer-major flatten 后进入网络，输出 dense temperature grid；faithful evaluation 还需要 `config.json`、`pytorch_model.bin` 和 `normalization_constants.json`。V7 的 `coords+k+q+BC` 是 irregular point representation，不能直接 drop-in。若未来执行，只能冻结 deterministic point-to-grid rasterization、channel/layer ordering、单位/训练集统计和 grid-to-query extraction；不得修改 backbone、添加 learned encoder 或温度先验。

## Asset and fairness decision

官方代码在 [Therm-FM repository](https://github.com/haiyangxin/Therm-FM) 已冻结到 commit `1c338d0…`（Apache-2.0），但当前没有可校验的 selective `model_T`、匹配 config/stats 或 V7-compatible conversion artifact。公开 archive 约 24.1 GB，steady dataset archive 约 4.47 GB，本轮不下载。预训练 Poseidon/scOT 数据与 V7 分布的重叠也尚未审计，所以不能把它写成 same-budget supervised baseline。

结论是 Therm-FM 独立 `BLOCKED_BY_UPSTREAM_ASSETS`，不阻塞其他 workstream。取得最小 model_T/config/stats 后，先在不含 test/sealed 的 benchmark 上冻结转换和泄漏审计，再考虑 valid-only transfer 结果。
