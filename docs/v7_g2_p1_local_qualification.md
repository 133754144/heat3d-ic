# V7 G2-P1 本地资格验证与 protocol v2

日期：2026-08-30。冻结比较参考为 `G1_FORMAL_CODE_SHA=191a7a06a681556f575a1c04e2b61cb13363efe1`。本轮只在 Mac CPU 执行；未 SSH、未访问 devbox、未读取 P1i `test_iid`/sealed、未启动 formal/multi-seed/长训练，也未依据 G1 中间结果改变 G2。

机器可读协议见 `configs/heat3d_v7/g2_p1_protocol_v2.json`，逐项运行证据见 `docs/v7_g2_p1_local_qualification_receipt.json`。

## 结论

| track | 模型 | 本机 gate / feasibility | 能否进入 P1i 同表 |
| --- | --- | --- | --- |
| common task | GINO | **PASS / A**：1 epoch train → valid → checkpoint → reload | 是；保持 GINO latent-grid/GNO/FNO 语义 |
| common task | Transolver | **PASS / A**：1 epoch train → valid → checkpoint → bitwise-equal reload | 是；保持 Physics-Attention recipe |
| conditional | Geo-FNO | **NOT RUN / E for P1i**：released IPHI/FNO 是 2D，42-D `rr` 是 Elasticity 专属 shape code | 否；当前强接会改变算法 |
| semiconductor native | DeepOHeat | 三个官方 pretrained/native inference 均 **PASS / C native, E for P1i** | 否；P1i 无可辨识的原生子空间 |
| transfer | Therm-FM | **C transfer, D packaging**：official demo PASS；未下载 24.10 GB 单体包 | 只能独立 transfer/data-efficiency 表 |
| design loop | DeepOHeat-v2 | **D code, E for P1i**：`official_code_not_found` | 只能 paper/design-loop track |

正式远程训练候选仅为 **GINO、Transolver**。Geo-FNO 保留 conditional，但在找到上游忠实的 3D pointwise-function 实现前不排任务。所有远程动作仍需 G1 结束后的新授权，本轮没有启动。

## 最小 P1i 数据与信息边界

从 Hugging Face `133754144X/heat3d-thermal-simulation@7b3af69e2164ad06d1c079fbde4d6cbd50183c9a` 选择性下载 4 个 train 与 2 个 `valid_iid` 样本。48 个必要文件全部与 frozen manifest SHA 对齐；没有下载 `temperature.npy`、full-field、test 或 sealed 内容。缓存仅位于 `/tmp/heat3d-g2-p1-qualification.*`，未提交 Git。

共同输入是同一 physical sample 的 `coords + kx/ky/kz + q + boundary flags + top/bottom HTC + top ambient offset`，target 是同 1024 个点的 `deltaT_K`。GINO 和 Transolver 只调整官方已有 input/output channel dimension，没有新增 learned encoder、branch 或 Heat3D-specific conditioning。坐标与 feature normalizer 只用这 4 个 train 样本拟合；valid 不参与拟合。

本机一轮 loss 只验证 forward/backward/optimizer/evaluator 是否闭环，样本太少且只跑一轮，**不是 accuracy 或 convergence claim**。

正式 common-task 评价统一对 denormalized `deltaT_K` 使用 `configs/heat3d_v7/v7_metric_contract.json`，同时把各模型的 upstream relative-L2 training loss 单列为训练诊断。checkpoint 使用 upstream final epoch，不做事后 valid-best 选择；指标名称继续遵守“不把不同 RMSE 定义笼统写成 RMSE”的约束。

## DeepOHeat 三个官方实验 contract

共同训练目标是由 PDE、边界条件和功率项组成的 data-free physics loss，`loss_fun_type="norm"`；不是 P1i supervised temperature regression。三个脚本均使用 Adam。最重要的 schedule 语义已冻结：每 500 epoch 执行 `lr *= 0.9`，随后用当前模型参数**重新创建 Adam optimizer**，因此 optimizer moments 同时重置。这一行为正式复现时不得改成普通 scheduler step。

### `2d_power_map`

- architecture：DeepONet；trunk `3 → 128`，3 个 hidden layers；branch `441 → 256`，7 个 hidden layers；inner product 128；SiLU；Fourier frequency `2π`、std 1、frequency trainable；总参数 689,346（trainable 689,154）。
- representation：`21×21=441` top-surface 2D power sensors作为 branch；trunk 查询 3D 坐标。几何为 `[0,1]×[0,1]×[0,0.5]`，作者训练离散间隔 `20×20×10`。
- physics：homogeneous conductivity background 1，PDE/HTC coefficient `k=0.2`；四侧 adiabatic、bottom HTC、top surface-power Neumann 项。
- sampling：每个参数 draw 2,000 PDE points、每个单独 BC 200 points；GRF `var=1`、length scale 0.3；每 epoch 50 power-map draws。
- optimizer/schedule：Adam，lr `1e-3`；10,000 epochs；每 500 epoch 0.9 decay并重建 Adam；每 1,000 epoch checkpoint；每 50 epoch validation。
- eval：官方 `model_epoch_10000.pth`，输出 `T_K=293.15+25u`。10 个 paper-showcase power maps 全部通过 pretrained inference。

### `single_htc_bc`

- architecture：DeepONet；trunk `3 → 128`，branch `1 → 20`，inner product 50，branch/trunk 均 3 hidden layers；SiLU，frequency `π` trainable；总参数 75,042（trainable 74,850）。
- representation：branch 唯一参数是 top HTC，连续范围 `(0.1,0.3)`；3D trunk 查询。几何 `[0,1]×[0,1]×[0,0.55]`，训练间隔 `20×20×11`。
- physics：固定 homogeneous k；中间层固定 volumetric power；四侧 adiabatic、固定 bottom HTC、可变 top HTC。
- sampling：4,000 PDE points、每 BC 500 points、每 epoch 20 HTC draws。
- optimizer/schedule：Adam，lr `1e-3`；5,000 epochs；每 500 epoch 0.9 decay并重建 Adam；每 200 epoch checkpoint；每 50 epoch validation。
- eval：官方 `model_epoch_5000.pth`，low/middle/high beta = 0.1/0.2/0.3；`51³=132,651` query points；三例 inference 均通过。

### `multi_htc_bc`

- architecture：MIONet；3D trunk width 128；两个独立 scalar branch nets（top、bottom HTC），branch width 20，inner product 50，3 hidden layers；SiLU，frequency `π` trainable；总参数 77,392（trainable 77,200）。
- representation：two-entry beta 表示 top/bottom HTC，二者均为 `(0.1,0.3)` 连续变量；其他 geometry/k/q 固定。
- physics/sampling：与 single HTC 相同的固定中间层 volumetric power、PDE/adiabatic/HTC loss、4,000 PDE points、每 BC 500 points、每 epoch 20 parameter draws。
- optimizer/schedule：Adam，lr `1e-3`；5,000 epochs；每 500 epoch 0.9 decay并重建 Adam；每 200 epoch checkpoint；每 50 epoch validation。
- eval：官方 `model_epoch_5000.pth`，low/middle/high beta = `[0.1,0.1]`/`[0.2,0.2]`/`[0.3,0.3]`；三例 inference 均通过。

本轮没有在 CPU 上执行 DeepOHeat 的 full-semantics 一轮训练：原脚本的一轮会一次性拼接 20 或 50 个完整 physics draws 并计算二阶导数。将 draws/PDE/BC points 缩小会违反本轮“保留 sampling/training semantics”的要求，因此宁可保留 pretrained/native inference 证据，也不制造一个语义不同的 tiny training gate。

## DeepOHeat 对 P1i 的可表示性

审计 frozen input-definitions CSV 的全部 1,024 行：top HTC、bottom HTC、total power、q proxy、local-k statistics，以及每一个 background `kz` 字段都各有 1,024 个唯一值；source count 有 8 种，k-region count 有 7 种，source placement/area 同时变化。

因此不存在可证明的：

1. fixed geometry/k/q、仅 top HTC 变化；
2. fixed geometry/k/q、仅 top+bottom HTC 变化；
3. fixed geometry/k/BC、q 可无损压缩为官方 top 2D power-map branch。

正式结论为：`P1i_direct_comparison_not_identifiable_without_algorithm_change`。没有继续 DeepOHeat→P1i 工程适配，也没有忽略 k/q/BC 变化来人为制造子集。

## GINO gate 与正式配置

最接近 P1i 的官方 recipe 是 3D CarCFD `GINO_Small3d`。冻结：GNO input/output radius 0.033，input mean reduction/output sum reduction；`32³` latent grid；3D FNO modes `16³`、4 layers、64 channels、instance norm、channel MLP、Tucker rank 0.4；AdamW lr `1e-3`/weight decay `1e-4`；StepLR(50, 0.5)；301 epochs；relative `LpLoss(d=2,p=2)`；batch 1。

P1i 只把官方 data channels 从 0 调为 11、latent-feature SDF 路径关闭，并将 linear input-integral 对应的 `fno_in_channels` 调为 11；输出仍为 1 channel。Mac 缺 Open3D/torch-scatter，使用 upstream 自带 pure-PyTorch neighbor/reduction fallback，radius 和 reduction 语义未改。

参数量 13,673,988。4 个 optimizer steps 后 checkpoint 的全部有状态参数 step 均为 4；2 个 valid 样本可推理，checkpoint 可重新装载并再次 forward。本轮 valid relative-L2 仅作 smoke diagnostic，不得作为性能结果。

上游当前 frozen commit 的 config 与 constructor 有轻微漂移：config 仍列 `fno_domain_padding`/旧 coordinate-embedding 字段，而当前 `GINO.__init__` 不接受前者。protocol v2 使用当前官方 constructor 可表达的等价字段；没有私自向模型增加 padding 层。

## Transolver gate 与正式配置

冻结官方 Elasticity shell recipe：8 layers、hidden 128、8 heads、64 slices、MLP ratio 1、dropout 0、unified position off、ref 8；AdamW lr `1e-3`/weight decay `1e-5`、gradient clip 0.1；CosineAnnealingLR `T_max=500`；500 epochs；batch 1；`TestLoss(size_average=false)` relative L2。P1i 仅把 `space_dim` 2→3、`fun_dim` 0→11，output 仍为 1。

参数量 716,737。1 epoch 的 train、valid、checkpoint、new-model reload 全部通过，reload 前后首个 valid prediction bitwise equal。输入 point-token 只包含共同 1024-point information，无 learned adapter、dense field 或 external prior。

## Geo-FNO 条件结论

原仓库已 deprecated。released Elasticity Geo-FNO 的 learned deformation `IPHI` 和 spectral path 是 2D；`fc_code=Linear(42,width)` 中的 42-D `rr` 是该 Elasticity 数据集专属全局 shape code。P1i 是 3D 并需要 pointwise k/q/BC functions。仅修改 input/output channel 无法把该实现变成 P1i 模型；需要重写 3D IPHI/Fourier path 或新增 learned conditioning，均属于算法改变。

所以本轮不运行虚假的 Geo-FNO P1i gate，状态为 `NOT_RUN_scientific_incompatibility_without_algorithm_change`。原论文 reproduction 证据仍保留，但不得与 modern NeuralOperator implementation 混称。

## Therm-FM 与 DeepOHeat-v2

Therm-FM 保留已有 official quick-demo training/evaluation PASS。真实 `model_T` 是约 21M 参数的 Poseidon/scOT grid model。steady loader 原始输入为 `(N,P,L,H,W)`，其中 `P` 是 x、y、z/layer、power density；进入模型前按 layer-major flatten 为 `(N,L*P,H,W)`，输出是 `(N,L_out,H,W)` temperature。HS_SC release shape 为 input `[5000,4,2,87,87]`、output `[5000,2,87,87]`。`normalization_constants.json` 必须提供每个 flattened input channel 与每个 output layer 的 mean/std，official evaluation 从 `model_path` 自动读取；`config.json`、`pytorch_model.bin`、normalization JSON 三者缺一不可。

真实 checkpoint 被打包在含 T/B/L 和所有 benchmark 的 24,100,784,430-byte 单体 tar，steady dataset 也是 4,467,245,278-byte 单体 tar。本轮不下载。它的定位固定为 `pretrained / transfer / data-efficiency competitor`，并明确：`pretrained prior != same from-scratch training budget`。

DeepOHeat-v2 截至 2026-08-30 仍没有找到可验证的 official code/data；未启动 independent reimplementation。保留论文中的 discrete energy-form loss、preconditioning/Muon2、hotspot residual trust gate、AMG-warm GMRES、solver-refined self-improvement 与 design-loop protocol，等待官方发布后再建任务。

## G1 结束后的远程候选清单

需要另行授权，且按以下次序推进：

1. GINO：先做单 seed resource pilot，确认 `32³` latent grid 的 GPU memory/wall-time，再按官方 301 epochs、AdamW+StepLR 执行 preregistered formal seeds。
2. Transolver：先做单 seed resource pilot，再按官方 500 epochs、AdamW+Cosine schedule 执行 preregistered formal seeds。
3. Therm-FM：仅在批准大文件下载后，独立 transfer/data-efficiency track 获取 model_T + 一个 steady benchmark + normalization constants。
4. DeepOHeat：只做原生 semiconductor physics comparison，不排 P1i main-table 训练。
5. Geo-FNO：等待 upstream-faithful 3D representability 解锁；否则不排任务。
6. DeepOHeat-v2：等待 official code/data；未来以 solver calls、verified design quality、wall time 和 trust-gate behavior 为指标，而非 P1i 同表 field error。
