# V7 G2 P22 Therm-FM native-65×65 adapter

状态：`FROZEN_P22_NATIVE65_REAL_TARGET`（正式训练仍须通过 T4 真实 target qualification）。

该 adapter 严格保留 frozen P1i split：train 768、valid_iid 128；不调用 Therm-FM 原始
`train_ratio` 重划数据，也不访问 `test_iid`、sealed 或 DeepOHeat official100。输入来自每个
case 的 physics metadata 和共享网格的确定性 dense rasterization，禁止 sparse-1024 插值。

## Native spatial contract

P1i 共享坐标已由 archive 验证为 C-order `x,y,z` 的 `65×65×57` 网格。每个 z layer 的
13 个通道固定为：

`x_norm, y_norm, z_norm, kx, ky, kz, q, boundary_front, boundary_back, boundary_left, boundary_right, h_top, h_bottom`。

因此 Therm-FM/scOT 接收 layer-major 的 `(741,65,65)`（13 channels × 57 layers），输出
`(57,65,65)`。不再进行外部 `65→128` zero padding，也不 resample；ScOT 自身在 patch/window
嵌入阶段处理非 patch-divisible 的 65×65，并由 recovery crop 回 65×65。监督 loss 只覆盖真实
物理像素，不含人工 padding pixel。

target 是 archive 中的真实 `deltaT_K`，按 `(z,x,y)` reshape；归一化是 train-only 的
741 input-channel 与 57 output-channel mean/std，统计域仅为真实 65×65 pixels。

## Physics completeness

13-channel contract 显式携带坐标、各向异性 `kx/ky/kz`、体热源 `q`、四侧 boundary masks、
top/bottom Robin `h`。固定 geometry 与 ambient 是 case-level constants，并由共享 geometry、
metadata 和 deterministic channel construction 共同确定；没有额外 learned encoder 或 solver
输入。

## Upstream semantics retained

Poseidon-T 是 initialization（不是 released thermal `model_T`），并使用
`--replace_embedding_recovery` 语义。ScOT architecture/backbone 不改；P22 protocol 冻结
AdamW、embedding/recovery parameter group learning rate、weight decay、cosine schedule、200
epochs、valid normalized p=2 loss 的 lower-is-better checkpoint selection。该 baseline 属于
`PRETRAINED_TRANSFER_FOUNDATION_MODEL_BASELINE`，不宣称 same-information 或 same-from-scratch
training budget。

对应机器可读合同：
`configs/heat3d_v7/g2_p22_thermfm_native65_protocol.json`。
