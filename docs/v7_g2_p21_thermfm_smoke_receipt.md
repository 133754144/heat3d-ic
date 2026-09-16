# P21 Therm-FM smoke receipt

结论：`PASS_BOUNDED_SMOKE_NOT_FORMAL_ACCURACY`，因此 Therm-FM 进入 `READY_FOR_FORMAL_TRAINING` 的工程候选状态，但仍需先物化完整 train-only normalization constants 并冻结正式 launch JSON。

在 devbox 独立 PDEFormer 环境（PyTorch 2.9.0+cu128、Transformers 4.57.1、Accelerate 1.11.0、RTX 5070）使用 Therm-FM commit `1c338d0fbe0dca25311eb896a9ea136a4f3d3cb1` 与 Poseidon-T revision `93adcbf10f75b45ac3bca3939cc3d3f239e1e663`。train fixture `v6p1if1_0000` 与 valid fixture `v6p1if1_0003` 均完成：

- 预训练权重加载；
- `--replace_embedding_recovery` 语义下的 741 输入通道/57 输出层替换；
- direct case-definition dense rasterization（65×65×57，确定性填充到 128×128）；
- finite forward/backward、有限 loss；
- valid forward；
- checkpoint save/load 后 state exact equality。

smoke target 是 finite synthetic unit target，仅用于检查梯度/闭环，不能作为物理精度证据。输入没有使用温度，也没有读取 test/sealed。Transformers 4.57 对 `ConvTranspose2d` shape-mismatch recovery 的默认初始化存在兼容性问题；仅按 scOT 自身 initializer 语义显式初始化 replacement recovery，并将该 compatibility patch SHA 写入 JSON。未修改 Poseidon/scOT backbone、loss 或数据物理定义。

指标和哈希详见 `docs/v7_g2_p21_thermfm_smoke_receipt.json`；大 checkpoint/dataset archive 未下载。正式工作仍保持 pretrained/transfer 分轨，不能写成 same-budget supervised baseline。
