# V7 G2 P21 Therm-FM P1i dense adapter

状态：`READY_FOR_P1I_FEASIBILITY`。该文件冻结的是可行性路径，不是正式 accuracy 结果。

## 冻结的表示

Therm-FM/scOT 的官方 steady-state loader 要求 `(N,P,L,H,W)` dense tensor。P1i 的共享几何仅在授权 train/valid fixture 上验证：240,825 个节点严格按 C-order `x,y,z` 排列，唯一网格为 `65×65×57`，而不是从节点数反推。输入由 case-definition metadata 和共享网格确定性栅格化，绝不从 1024 个 sparse 点插值，也不读取温度作为输入。

每个 z layer 的 13 个通道固定为：`x_norm,y_norm,z_norm,kx,ky,kz,q,boundary_front,boundary_back,boundary_left,boundary_right,h_top,h_bottom`。因此 scOT 输入通道数为 `13×57=741`，输出为 57 个 temperature layers。65×65 物理切片使用确定性的右/下零填充到 Poseidon-T 的 128×128 image size；不做物理重采样，输出再裁回 65×65 评价域。

归一化只允许由 768 个 train case 拟合，valid 不参与拟合，test/sealed 不读取。完整 train-only statistics 尚未物化，因此尚未启动正式训练。

## Smoke 证据

在 devbox 的独立 `PDEFormer` 环境，以官方 Therm-FM commit `1c338d0fbe0dca25311eb896a9ea136a4f3d3cb1` 和 Poseidon-T revision `93adcbf10f75b45ac3bca3939cc3d3f239e1e663` 做了两个 bounded fixture：train `v6p1if1_0000` 与 valid `v6p1if1_0003`。两者都完成 pretrained load、embedding/recovery replacement、finite forward/backward、有限 loss、valid forward 和 checkpoint reload。输入/输出只用于 smoke，target 为 synthetic finite target，不构成 accuracy 证据。Poseidon 权重及 config 的 SHA 见 JSON receipt。

`transformers 4.57` 对 shape-mismatched `ConvTranspose2d` recovery 未初始化；smoke 仅用 scOT 自身 initializer 语义显式初始化 replacement recovery。这是兼容性修复，不是 backbone 或 loss 改动；正式运行前应在隔离环境固定并记录该 patch。

## 公平性边界

这是 dense-input pretrained transfer baseline：dense channels 暴露的空间信息与 Heat3D sparse support 不同，且 Poseidon 预训练先验不等于 from-scratch 训练预算。即使后续 valid-only 训练可行，也只能作为 transfer/foundation-model 分轨。

## 证据文件

- `configs/heat3d_v7/g2_p21_thermfm_p1i_feasibility_protocol.json`
- `docs/v7_g2_p21_thermfm_smoke_receipt.json`
- `docs/v7_g2_p21_thermfm_asset_receipt.json`
