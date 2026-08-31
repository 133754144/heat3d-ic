# G2-P3 “Why Heat3D?” 论文证据矩阵

状态：`paper_motivation_frozen_before_G2_results`。符号：**是**表示论文或官方代码明确支持；**固定**表示物理量存在但不是逐 case 显式条件；**条件**表示框架能表达但公开实验/代码范围更窄；**未报告**表示不能从一手证据推出。这里比较的是 capability，不是 accuracy 排名。

一手来源：[DeepOHeat/DAC 2023](https://arxiv.org/abs/2302.12949)、[DeepOHeat-v1/TCPMT](https://arxiv.org/abs/2504.03955)、[DeepOHeat-v2](https://arxiv.org/abs/2608.16080)、[ARO/ICCAD 2024](https://doi.org/10.1145/3676536.3676713)、[T-Fusion/ASP-DAC 2025](https://doi.org/10.1145/3658617.3697749)、[SAU-FNO/DAC 2025](https://arxiv.org/abs/2510.15968)、[Therm-FM](https://arxiv.org/abs/2605.22663)。Heat3D 一列只陈述本仓库 frozen P1i contract，不使用 G1 interim result。

## Representation 与显式物理信息

| 方法 | geometry representation | grid dependence | irregular-point support | heterogeneous k(x) | q variation | BC conditioning | geometry/layout variation | input/output resolution coupling |
|---|---|---|---|---|---|---|---|---|
| DeepOHeat | cuboid/multi-domain configuration + coordinate trunk | branch sensors 固定；query 可连续 | output query 条件支持；input 非 point-set native | 条件：框架可编码材料配置，released 三任务多为固定 k | 是，surface/volume power | 是，HTC/BC branches | 条件：框架广于 released fixed-cuboid tasks | input sensors 固定；output trunk 解耦 |
| DeepOHeat-v1 | 固定 cuboid；x/y/z separable KAN axes | power branch 固定 21²/101²；输出为 Cartesian product | 否，separable axes 不是任意点集 | 固定 piecewise coefficient；不是 case-wise k input | 是，surface/volumetric maps | 固定 Robin/adiabatic | 否 | input 固定；output Cartesian resolution 可变 |
| DeepOHeat-v2 | 已知 FVM operator 上的 300×300×23 chiplet grid | 强 | 否 | 固定在离散 operator 中，不是显式 case input | 是，placement/power | 固定在 operator 中 | placement/layout 变化，geometry stack 固定 | 耦合于固定 FVM grid |
| ARO | power trace → spatial-temporal field 的 autoregressive operator | 论文 benchmark 为 well-defined field domains | 未报告 point-native | 未报告显式 k input | 是 | 未报告显式 BC input | 是，主张 unseen circuits/transfer | 未报告独立 query resolution |
| T-Fusion | tensor arithmetic + Bayesian autoregression 的 multi-fidelity fields | 是 | 否 | 未报告显式 k input | 是 | 未报告显式 BC input | 多种 chip designs，但不是坐标条件 operator | 同一 benchmark grid 内耦合 |
| SAU-FNO | self-attention U-Net FNO grid field | 是 | 否 | 固定/隐含于 benchmark，不是显式 k input | 是 | 固定/隐含 | 多个设计 case；非任意 geometry point set | 标准 grid-to-grid |
| Therm-FM | multilayer `(N,P,L,H,W)` grid，layer-major flatten | 是 | 否 | benchmark 可含异质结构，但 model_T contract 不是 P1i pointwise k | 是，power-density channel | package BC 隐含/benchmark-specific | cross-chip adaptation | grid-to-grid，依赖 benchmark normalization |
| Heat3D | sparse physical points + regional graph + independent query points | 否 | **是** | **是：pointwise kx/ky/kz** | **是：pointwise q** | **是：flags + top/bottom h + ambient offset** | **是：coords 与显式 fields** | **解耦：input observations 与 output queries 可不同** |

## Training、transfer 与 deployment

| 方法 | supervised / physics-informed | pretraining | multi-fidelity | cross-chip transfer | solver coupling | design-loop capability |
|---|---|---|---|---|---|---|
| DeepOHeat | physics-informed/data-free PDE+BC residual | 否 | 否 | 未报告系统性 cross-chip | 训练用 autodiff residual；inference 无 solver | 是，论文定位热设计快速调用 |
| DeepOHeat-v1 | physics-informed full-mesh residual | 否 | 否 | 否 | **hybrid DeepOHeat→GMRES** | solver warm-start/certification，非完整 placement loop |
| DeepOHeat-v2 | discrete energy-form physics loss + online self-improvement | 否 | online solver labels | placement trajectory 内更新 | **AMG-warm GMRES + trust gate** | **是** |
| ARO | supervised operator learning + active learning | 否 | **是** | **是** | data generation 依赖 simulators；inference 无 solver | 可用于 design exploration，非 solver-in-loop trust gate |
| T-Fusion | supervised Bayesian multi-fidelity fusion | 否 | **是** | 多 design 实验；系统性 cross-chip claim 未冻结 | low/high-fidelity simulators 生成数据 | 未报告 |
| SAU-FNO | supervised | low-fidelity pretrain→high-fidelity fine-tune | **是** | transfer learning；cross-chip 范围未单独证明 | 数据生成 solver only | 未报告 |
| Therm-FM | pretrained Poseidon/scOT + thermal fine-tune | **是** | **是** | **是，10–30 target samples track** | 数据生成 solver only | 未报告 solver-in-loop design loop |
| Heat3D | supervised P1i | 否 | 当前否 | 当前未宣称 | inference 无 solver | 当前未宣称 |

## 严格 gap statement

Heat3D 的研究缺口不是“现有方法不支持 heterogeneous materials”。DeepOHeat framework、DeepOHeat-v1/v2 的固定离散物理以及 Therm-FM 的工业 benchmark 都能涉及异质材料。更窄、可检验的组合缺口是：

> 在同一 operator 输入中联合处理 **sparse irregular physical observations + explicit heterogeneous k/q/BC + geometry-aware operator + resolution-decoupled inference**。

这个 statement 只说明输入/算子接口的组合差异，不预设 Heat3D accuracy、runtime 或 transfer 优于任何方法。ARO/T-Fusion/SAU-FNO 的未报告项保持 unknown，不能把 absence of evidence 改写成不支持。
