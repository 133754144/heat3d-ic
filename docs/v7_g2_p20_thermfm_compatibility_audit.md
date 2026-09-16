# V7 G2 P20-B：Therm-FM compatibility/data audit

状态：`READY_FOR_P1I_FEASIBILITY`（P21 selective Poseidon-T smoke 已通过）；大型 Therm-FM archive 仍未下载。本轮只做接口/provenance 与 bounded smoke，没有正式训练，也没有访问任何 test/sealed。

## P1i labels

V6 tracked full-field sidecar 的 `status=complete`、`sample_count=1024`、`solver_node_count=240825`，并为每个样本记录 temperature/deltaT SHA；角色计数为 train 768、valid_iid 128、test_iid 128。因此已有 manifest 足以证明 train768 具有冻结的 **240,825-node `full_fields.h5`** dense-label contract。它不是 DeepOHeat-v1 **571,256-node** label cache；两者 provenance 始终分开。devbox 只读 recheck 进一步确认后者目录已有 768 train + 128 valid（约 2.09 GB），label receipt SHA 为 `a4bb9963…fe4fd`，normalization payload SHA 为 `3a0273bb…ed0db`。本地 worktree 仍不复制该大目录。

## Interface

Therm-FM 官方 `model_T` 是约 21M 参数的 Poseidon/scOT dense-grid operator：输入 `(N,P,L,H,W)`，按 layer-major flatten 后进入网络，输出 dense temperature grid；faithful evaluation 还需要 `config.json`、`pytorch_model.bin` 和 `normalization_constants.json`。V7 的 `coords+k+q+BC` 是 irregular point representation，不能直接 drop-in。若未来执行，只能冻结 deterministic point-to-grid rasterization、channel/layer ordering、单位/训练集统计和 grid-to-query extraction；不得修改 backbone、添加 learned encoder 或温度先验。

## Asset and fairness decision

官方代码在 [Therm-FM repository](https://github.com/haiyangxin/Therm-FM) 已冻结到 commit `1c338d0…`（Apache-2.0）。P21 已取得并核验 Poseidon-T config/weights（83.4 MB），并在授权 train/valid fixture 上验证 direct dense rasterization 与 replacement-layer smoke；公开 Therm-FM archive（约 24.1 GB）和 steady dataset archive（约 4.47 GB）仍不下载。匹配 thermal `model_T`/stats 和预训练重叠审计仍是正式 accuracy 前置条件，因此不能把它写成 same-budget supervised baseline。

结论是 Therm-FM 独立 `READY_FOR_P1I_FEASIBILITY`，不阻塞其他 workstream。下一步是在不含 test/sealed 的 benchmark 上物化 train-only stats、完成预训练泄漏审计，并在独立环境中冻结 valid-only transfer 运行；若 thermal `model_T`/stats 无法获得，则回退为明确的 upstream-asset blocker。
