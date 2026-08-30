# G2-P2 DeepOHeat-v1 feasibility audit

日期：2026-08-31。范围仅为本地 literature/code freeze；没有 SSH、长训练、P1i learned adapter 或 formal G2 run。

## Provenance 与可复现边界

- paper：[arXiv:2504.03955](https://arxiv.org/abs/2504.03955)，*DeepOHeat-v1: Efficient Operator Learning for Fast and Trustworthy Thermal Simulation and Optimization in 3D-IC Design*；Xinling Yu、Ziyue Liu、Hai Li、Yixing Li、Xin Ai、Zhiyu Zeng、Ian Young、Zheng Zhang；IEEE Transactions on Components, Packaging and Manufacturing Technology（online 2025 / volume 2026）。
- official repo：[xlyu0127/DeepOHeat-v1](https://github.com/xlyu0127/DeepOHeat-v1)，冻结 commit `3ef3d9c41666a56b5940b39a61166ccaa5aaedb2`（无 tag）。论文与仓库相互链接。
- license：仓库没有 `LICENSE`、SPDX header 或 README license grant，故冻结为 `NO_LICENSE_FILE_NO_EXPLICIT_REUSE_GRANT`，不得推定为 MIT。
- release completeness：仓库只有 9 个 Python/notebook 源文件和 README；没有 dependency lock、训练数据、normalization manifest、`.eqx` checkpoint 或 recorded evaluation CSV。README 的 Google Drive 只标为 “Data”，未给文件清单/hash；本轮未下载。
- local reproduction：没有 official checkpoint，无法执行 native pretrained inference；没有 data/lock，且 full-semantics 一轮 surface/volume physics training 均需作者完整 batch/mesh，故没有用缩小 sampling 制造 smoke。状态为 `NOT_RUN_resource_contract_incomplete`。

因此本轮 classification 为 **D, formal_semiconductor_native_candidate**：科学任务适合 semiconductor-native 独立表，但在 checkpoint/data manifest、依赖锁和 license 未解决前，尚不能冻结为 `formal_semiconductor_native_baseline`，也不进入 P1i common-task 表。

## Architecture 与训练语义

DeepOHeat-v1 是 physics-informed DeepONet 变体。branch 编码 surface power map 或 volumetric floorplan；三个独立的一维 KAN trunks 分别处理 x/y/z；各轴 basis 与 branch coefficients 用外积/einsum 组成低秩 3D field。trunk 是 4-layer、order-3 Chebyshev KAN（hidden 64），每轴独立；rank 128。branch 是 441/10,201 → 256 的 9-layer MLP（代码参数 `branch_depth=8`，Equinox 实际含 input/output 共 9 个 Linear），swish，输出 128。

用冻结源码与 `equinox==0.13.8` 仅构造模型并按上游 `eqx.is_array` 计数：surface 805,168 parameters，volumetric 3,303,728。该动作没有 forward、数据或训练。

### Surface power

- physical case：1 mm × 1 mm × 0.5 mm homogeneous cuboid，k=0.1 W/(m K)，侧面 adiabatic，bottom convection HTC=500 W/(m² K)、ambient=298.15 K，top 2D power map。
- representation：branch 441 = 21×21 sensors；separable axes 21×21×11（4,851 full-mesh physics locations）。test generator 为 axes 101×101×51。
- physics objective：interior steady Laplace residual；top Neumann power residual；bottom Robin residual；四侧 adiabatic derivative residual；`lam_b=1`。无 supervised temperature loss。
- sampling：每 iteration 从 GRF surface maps 中 without-replacement 抽 50；GRF length scale 0.3；loss 在完整 separable mesh 计算。
- optimizer：Adam，lr 1e-3；10,000 iterations；Optax exponential decay 0.9 / 500 steps；seed 42。
- normalization：源码没有 dataset-fitted standardization。坐标已经无量纲化到 `[0,1]×[0,1]×[0,0.5]`；网络输出 u 在 evaluation 用 `T_K=293.15+25u` 转物理温度。power arrays 的生成尺度未由 release manifest 说明。
- checkpoint/eval：训练结束仅序列化 `DeepOHeat_v1_trained_model.eqx`，不是 valid-best；evaluation 保存 `u_pred_heat3d.npy`，报告 per-case mean/std 的 relative L2、normalized-u root mean squared error、peak absolute difference、physical-temperature MAPE/PAPE。repo 未发布该 checkpoint/output。
- paper result：10 个 structured showcase maps，MAPE 0.049%；RTX 3090 training 0.0084 h（30.24 s）、peak GPU memory 0.56 GB。它是 native-domain paper result，不是本机 reproduction。

### Volumetric power

- physical case：1 mm × 1 mm × 0.55 mm three-layer stack，101×101×56 mesh；10 个矩形 components 位于 middle layer并允许 overlap，powers 为 0.5/1/2 mW；侧面 adiabatic，top/bottom HTC=500 W/(m² K)、ambient 298.15 K。
- representation：branch 10,201 = 101×101 floorplan/power map；separable trunks on 101×101×56 axes。
- physics objective：piecewise-k Laplacian residual、middle-layer volumetric source residual、top/bottom Robin BC 与侧面 adiabatic BC；完整约 570K-point mesh，未缩采样。
- sampling/optimization：每 iteration 50 random floorplans；100,000 iterations；Adam lr 1e-3，0.9 / 1,000-step exponential decay；seed 42。
- checkpoint/eval：同样是 final serialized leaves；100 random test floorplans，对 PC-GMRES+AMG references 报相同 metric family。repo 未发布 data/checkpoint/result CSV。
- paper result：MAPE 0.035%，RTX 3090 training 0.52 h，12.10 GB；属于论文报告。

## Confidence 与 hybrid GMRES

线性热系统为 `A T=b`。confidence score 是 relative algebraic residual `r=||A T0-b||₂/||b||₂`；`r<α` 时直接接受 operator prediction，否则把 `T0` 作为 GMRES warm start。它不需要 reference temperature，因此能在 design loop 中做 trust gate，但依赖与训练 problem 完全一致的离散 A/b。

released notebook 固定 101×101×56 finite-difference matrix，CuPy GMRES `restart=200, maxiter=20000, tol=0.05`；`precondition=False` 的 hybrid refinement 不用 AMG，PC-GMRES+AMG 另作 reference。论文报告 random-init GMRES 20,000 iterations 后 27.54 s，而 operator warm start 0.55 s 达 relative residual 0.04（约 50×）；optimization 中 threshold α=9，1,000 steps 有 299 次 refinement。论文表格给出 PC-GMRES 4.94 h、operator-only 0.05 h、hybrid 0.07 h，hybrid 相对 PC-GMRES 约 70.6×，这些都不是本机测量。

## Formal gate

允许进入 formal semiconductor-native 独立表之前必须同时满足：明确的 code license/reuse permission；可校验的 official surface/volume data 与 checkpoint manifest；可重建的 Python/JAX/Equinox/Optax/CuPy/PyAMG lock；至少一个 official native inference/evaluation receipt；明确 paper code 与 released checkpoint 的 normalization/version对应。P1i learned adapter、修改 physics loss 或缩减核心 50-function/full-mesh semantics均不能用来绕过该 gate。
