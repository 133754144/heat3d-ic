# MASS-HBM 原始数据集与 Heat3D/V7 物理兼容性审计

日期：2026-09-19

状态：`AUDIT_AND_PROTOCOL_COMPLETE_NO_TRAINING`
机器可读版本：[v8_mass_hbm_dataset_compatibility_audit.json](v8_mass_hbm_dataset_compatibility_audit.json)

## 0. 结论先行

**总判定：当前 V7 + dataset adapter 为 `NO_GO`；独立 V8 physics contract 为 `CONDITIONAL_GO / MODEL_SCHEMA_CHANGE_GO`。** RIGNO 主体可以先保留，但 V7/P1i 的 11 维 local input、perfect-contact 界面、仅顶/底 Robin 边界与固定 P1i 几何契约不能表达 MASS-HBM 的界面热阻、内部液冷、温度反馈功率和可变架构。数值范围的局部重叠不等于物理兼容。

两条路线必须隔离：

- Track A 可读取收敛后的 `k/q/Rint` 与最终网格，仅作为 **oracle-field representation-capacity diagnostic**；不能声称替代求解器。
- Track B 是最终 accuracy-speed 主路线，只能使用 pre-solve/deployable 输入，直接预测收敛温度场。

当前不允许开始 smoke training。进入 Phase 2 前必须先完成 Phase 1 的只读 V8 adapter、字段来源验证、泄漏 gate、group manifest，并再次取得明确训练授权。

### 证据类别约定

- **[PAPER]**：来自 IEDM 2026 MASS-HBM 四页论文原文。
- **[RAW]**：来自 `/Users/xuyihua/Downloads/GlobalThermal_AI_Dataset` 的文件、数组或 metadata 的只读审计。
- **[REPO]**：来自当前 Heat3D V6-P1i/V7/G1/G2 仓库契约。
- **[INFERENCE]**：基于上述证据的判断或方案；不是作者事实。
- `UNKNOWN`：现有论文、数据交接包和仓库均不足以确认，禁止猜测。

## 1. 审计边界、方法与可复核性

本次只读审计未运行 MASS-HBM solver、Heat3D 训练、checkpoint 选择或 inference；未访问 sealed split；未修改 G1/G2 evidence、原始数据、`data/`、`output/`、`checkpoints/` 或 `logs/`。没有 silent clipping、unit guessing 或 normalization refit。

论文：`finalfinal.pdf`，题目为 *MASS-HBM: A Multiphysics Atom-to-System Simulation Tool for Thermal-Reliability-Aware System-Technology Co-Optimization of HBM*，共 4 页。PDF 经文本与页面渲染双重核对。

数据审计使用新增的只读脚本 `scripts/audit-mass-hbm.py`：

- 对 314 个 case 的索引、case summary 和 manifest 做全量 metadata 扫描；
- 对 32 个按 architecture / experiment / sweep / workload / pitch / slab / cooling 分层选出的代表 case 抽取数组；
- 每个数组最多确定性等距抽样 100,000 个值；`.npy` 使用 mmap；只对代表性 `.npz` 解压；
- 输出写入 `/tmp`，原始数据目录写入次数为 0。

三项根 metadata SHA256：

| 文件 | SHA256 |
|---|---|
| `dataset_index.csv` | `c5e0a85ed164f094dcfcb1985aac429f991000ccdf3014c9e1523fcc40566554` |
| `dataset_metadata.json` | `d86df19866022b35e416afe8a4678dc00a4db1b4d713d7f5171f54334e0f1fc3` |
| `files_manifest.csv` | `fe67a43f1033de13472b07cbfa58d7a3d536bb4ecd1113bcc3affba9de391157` |

## A. Dataset inventory

### A.1 根目录与总体规模 [RAW]

根目录包含：

```text
GlobalThermal_AI_Dataset/
├── cases/
├── dataset_index.csv
├── dataset_metadata.json
├── files_manifest.csv
├── load_dataset_example.py
├── README.md
├── shared_configs/
└── training_eligible_index.csv
```

- schema：`global-thermal-ai-handoff-v2`
- source Git HEAD：`6c167dc88104f52beedf3a20c837a1d3771c5c1c`
- 314 个 `FINAL_CONVERGED`、`full_coupling` case；3 个未收敛、physical-limit 或不完整 case 未打包。
- `cases/` 约 3.78 GB；manifest 共 6,940 个文件、3,782,471,081 bytes。
- 格式：3,456 CSV、1,884 JSON、942 NPY、628 NPZ、28 YAML、1 Python、1 Markdown。
- 交接包自述 scope 为 `reduced-order global steady-state electrothermal model`；LC-V 是 reduced-order distributed sink，不是 CFD 或厂商标定模型。

### A.2 case 数量与设计族 [RAW]

| 维度 | 数量 |
|---|---:|
| 2.5D-HBM | 27 |
| 3D-P-HBM | 26 |
| 3D-V-HBM | 29 |
| LC-V-HBM | 232 |
| Baseline | 4 |
| Exp0 SolverAcceleration | 6 |
| Exp1 CapacityScaling | 12 |
| Exp2 ModelWorkload | 79 |
| Exp3 SizeSweep | 105 |
| Exp4 STCOOverview | 108 |

扫描族共 28 类。主要 sweep 包括 `capacity`、`bandwidth`、`capacity_bandwidth_sweep`、`group_count_sweep`、`layer_count_sweep`、`thickness_sweep`、多个 software codesign / LC-V placement / DVFS 系列，以及 2.5D、3D-P、3D-V、LC-V 架构族。

可见参数覆盖：capacity 48–828 GB，requested bandwidth 2.015–44.056 TB/s，pitch 50/100/150/200/250/300/400/500 μm，group count 1–10，frequency factor 0.5–1.0。工作负载含 Llama-3.1-70B/8B、Llama-3.2-3B、Qwen2.5-14B/32B、reference prefill 和 high-capacity bandwidth stress；phase 为 prefill/decode。部分列缺失，因此不能把这些字段当作每个 case 都完整可用。

### A.3 每个 case 的目录与字段 [RAW]

```text
case/
├── structure/
│   └── solver_geometry.json, component placement/rules, interface manifest
├── thermal_parameters/
│   └── final_material_k_maps.npz, final_interface_tbr_map.npz,
│       materials.csv, interfaces.csv
├── boundary_conditions/
│   └── boundary_conditions.json, lc_sink_audit.csv,
│       lc_region_mask_audit.csv
├── power_input/
│   └── power_density_map.npy, component_power_by_die.csv,
│       power_feedback.csv, power_conservation_audit.csv
├── temperature_output/
│   └── temperature_map.npy, temperature_map_C.npy,
│       stack_temperature_summary.csv, convergence_history.csv,
│       case_summary.json
├── provenance/
│   └── case_manifest.json, latest_state.json
└── labels.json
```

| 要求字段 | 取证结果 |
|---|---|
| coordinates | 未存独立坐标数组；可由 structured geometry 重建 cell centers |
| mesh/connectivity | x/y 等距、z 非均匀；无显式 connectivity 文件，可从 shape/spacing 推导 6-neighbor |
| cell/control-volume size | `solver_geometry.json.z_cell_thicknesses_m`；x/y 由 extent 与网格数获得；必须用于体积加权 |
| material/region | `materials.csv`、component spatial placement、region/interface manifests |
| k | `final_material_k_maps.npz`，`kx/ky/kz`，W/(m K) |
| q/P | `power_density_map.npy`，W/m³；另有 component、feedback、conservation 表 |
| Rint | `final_interface_tbr_map.npz`，m²K/W；另有 `interfaces.csv` 与 calibration metadata |
| BC | external HTC/ambient JSON；LC-V 有 internal sink audit/mask |
| temperature | `temperature_map.npy` K；`temperature_map_C.npy` 为便利副本 |
| stress/strain | manifest 与 case 文件中没有对应场数组；`UNKNOWN / ABSENT_FROM_HANDOFF` |
| cooling channels | reduced-order distributed internal sink 的 region/mask/HTC/gap/audit 信息 |
| workload/activity | index、shared config、component power、model/phase/frequency 字段 |
| solver iterations | `convergence_history.csv`、`power_feedback.csv`、materials/interfaces iteration rows |
| runtime/timing | case wall time、assembly/solve/coupled timing与 backend metadata |

## B. Physics semantics audit

### B.1 论文所述物理 [PAPER]

1. MASS-HBM 接收 architecture、geometry、material、thermal BC 和 workload，进行跨原子—器件—系统的稳态 electrothermal 计算。
2. 自洽循环更新 temperature-dependent conductivity、interface resistance、DRAM leakage/refresh power，直到温度与功率收敛。
3. 论文将导热率描述为 `k(T,σ)`，界面热阻描述为依赖温度/应力。
4. adaptive mesh 初始由结构和界面决定，随后根据演化温度场调整：低梯度区粗化、hotspot 附近保留细化。
5. 论文包含 2.5D、3D-P、3D-V，以及组间液冷 channel 优化；75°C/100°C 分别是 warning/critical 阈值。
6. 论文展示 stress/strain/reliability 结果，但论文展示不等于本交接包包含对应场。

### B.2 原始交接包语义 [RAW]

| 变量 | 物理含义 | 单位 | 位置/形状 | tensor | 生命周期 |
|---|---|---|---|---|---|
| geometry/coords | structured cell geometry / cell center | m | cell，`[Nz,Ny,Nx]` | — | pre-solve 可得，但须核验是否为最终 adaptive mesh |
| volume | control-volume measure | m³ | cell | scalar | pre-solve 可推导 |
| material/region | 材料与组件归属 | categorical | cell/region | — | pre-solve |
| final `k` | 收敛态轴向导热率 | W/(m K) | cell，3 个同形数组 | diagonal3 | **converged** |
| nominal/baseline `k` | reference-temperature 材料参数 | W/(m K) | material/iteration table | diagonal3 | initial/nominal，需 adapter 重建 |
| final `q` | 自洽反馈后的体热源 | W/m³ | cell | scalar | **converged** |
| initial power/activity | 初始 workload/component 功率 | W；空间化后 W/m³ | global/component/cell | scalar | pre-solve，但需 adapter |
| final `Rint` | 收敛态 TBR | m²K/W | z-face 或 homogenized representation | scalar face / effective tensor | **converged** |
| nominal `Rint` | reference/calibration TBR | m²K/W | interface/global | scalar | pre-solve 参数 |
| external BC | ambient 与外表面对流 | K、W/(m²K) | boundary face | scalar + normal | pre-solve |
| LC sink | 内部 distributed sink | K、W/(m²K)、W/K | internal cells/faces | scalar | case design + audit；控制器依赖须再核验 |
| temperature | 收敛稳态温度 | K primary | cell | scalar | label/output |
| stress/strain | 机械状态 | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | 原始交接包缺失 |
| iteration/runtime | 收敛与耗时轨迹 | residual/iteration/s | global/iteration | — | metadata only |

代表性 `materials.csv` 同时含 baseline k、raw `k(T)`、updated k 和 temperature mean，并按 iteration 记录；HBM cell 的 temperature-dependent fraction 在 314 个 case 中均为 1.0。最终 `k` 因而不是部署时已知常量。

`power_feedback.csv` 同时包含 initial fixed-power 与后续 temperature-feedback 记录；leakage/refresh 分量随温度更新。最终 `power_density_map.npy` 不能默认视为 pre-solve 输入。

`interfaces.csv` 含 interface temperature 与 intrinsic/bond/effective resistance；最终 TBR 随自洽状态变化。交接包没有 stress field，故无法从现有数据独立复原 `Rint(T,σ)` 中的应力贡献。

### B.3 强制 oracle 标记 [INFERENCE]

以下量一律标为 `TARGET_DEPENDENT / ORACLE_ONLY`：

- `final_material_k_maps.npz`
- feedback-updated `power_density_map.npy`
- `final_interface_tbr_map.npz`
- final leakage / refresh / component power
- 任何由温度梯度或 hotspot 决定的 final adaptive/refined mesh、support 或 workload placement

它们可进入 Track A，但不得进入 Track B。即使不把 `T` 数组直接传给模型，`T -> mesh/k/q/Rint -> input` 仍是信息泄漏。

## C. MASS-HBM → P1i Heat3D compatibility

当前 P1i local input 是：

```text
[k_x, k_y, k_z, q,
 is_top, is_bottom, is_side, is_interior,
 top_h, bottom_h, top_T_inf_minus_T_ref]
```

P1i 还假定 structured orthogonal mesh、diagonal `k`、perfect contact、顶/底 Robin、侧壁 adiabatic；V7 decoder bypass 只取前 8 个 local feature，另有固定 24D global context。loader 固定 1024 physics-layout-aware support。

### C.1 字段映射矩阵

| MASS-HBM 字段/物理 | Track A | Track B | 说明 |
|---|---|---|---|
| cell-center coordinates | `ADAPTER_ONLY` | `ADAPTER_ONLY` | 从 geometry 重建并转 m，不裁剪 |
| structured connectivity | `ADAPTER_ONLY` | `ADAPTER_ONLY` | 无显式 connectivity；推导 6-neighbor |
| cell volume | `ADAPTER_ONLY` | `ADAPTER_ONLY` | 非均匀 z；loss/metric 必须体积加权 |
| scalar k | `LOSSY_MAPPING` | `LOSSY_MAPPING` | 仅 metadata 证明 isotropic 时才能复制为 diag3 |
| diagonal3 k | `DIRECT_COMPATIBLE` | `ADAPTER_ONLY` | final 数组数值形状兼容但 oracle-only；Track B 重建 nominal k |
| full symmetric-6 k | `MODEL_SCHEMA_CHANGE_REQUIRED` | `MODEL_SCHEMA_CHANGE_REQUIRED` | 当前 handoff 未见 off-diagonal；P1i 也不能表达 |
| final q | `DIRECT_COMPATIBLE` | `UNSUPPORTED` | 数值单位兼容，但为反馈收敛态 |
| initial workload/power | `ADAPTER_ONLY` | `MODEL_SCHEMA_CHANGE_REQUIRED` | 需 component/local activity schema |
| arbitrary material interfaces | `MODEL_SCHEMA_CHANGE_REQUIRED` | `MODEL_SCHEMA_CHANGE_REQUIRED` | 11 维输入无 material/interface topology |
| Rint | `MODEL_SCHEMA_CHANGE_REQUIRED` | `MODEL_SCHEMA_CHANGE_REQUIRED` | 需 interface-aware edge/face feature |
| compatible external top/bottom Robin | `ADAPTER_ONLY` | `ADAPTER_ONLY` | 物理类型相同，但 MASS HTC 多为强 OOD |
| internal liquid-cooling Robin/sink | `MODEL_SCHEMA_CHANGE_REQUIRED` | `MODEL_SCHEMA_CHANGE_REQUIRED` | 不能伪装成 top/bottom flag |
| arbitrary boundary normals | `MODEL_SCHEMA_CHANGE_REQUIRED` | `MODEL_SCHEMA_CHANGE_REQUIRED` | 需 normal + BC type + parameters |
| adaptive/irregular mesh | `MODEL_SCHEMA_CHANGE_REQUIRED` | `UNSUPPORTED` | point operator可处理坐标不等于当前 contract 已支持；T-guided final mesh 禁入 Track B |
| 2.5D/3D-P/3D-V variable geometry | `MODEL_SCHEMA_CHANGE_REQUIRED` | `MODEL_SCHEMA_CHANGE_REQUIRED` | 远超固定 P1i stack/footprint |
| power-temperature feedback | `ORACLE_ONLY` | `MODEL_SCHEMA_CHANGE_REQUIRED` | Track B 学 initial-input → converged-T，不接收 final power |
| temperature-dependent conductivity | `ORACLE_ONLY` | `MODEL_SCHEMA_CHANGE_REQUIRED` | Track B 接 nominal/law parameter，不接收 final evaluated k |
| 59,150–490,100 point full field | `ADAPTER_ONLY` | `ADAPTER_ONLY` | 1024 support 可保留；full query 必须稀疏/缓存/分块 |
| converged T target | `DIRECT_COMPATIBLE` | `DIRECT_COMPATIBLE` | Kelvin、cell alignment 后直接监督 |
| stress/strain | `UNSUPPORTED` | `UNSUPPORTED` | handoff 缺场、单位和位置 |
| solver/runtime | `METADATA_ONLY` | `METADATA_ONLY` | 只用于审计/benchmark |

### C.2 重点十项结论

1. **scalar/diag3/full-6 k**：raw final field 是 diag3；scalar 压缩会丢失 anisotropy；full-6 未在 handoff 中出现，但未来若提供，V8 必须扩 schema。
2. **arbitrary interfaces**：53 个 case 使用 `explicit_series_z`，261 个使用 `rotated_homogenized_tensor`；不能都折叠成 local cell k 后宣称界面物理等价。
3. **Rint**：P1i 固定 perfect contact；MASS-HBM TBR 非零且状态相关，必须进入 edge/face contract。
4. **internal liquid cooling**：105 个 case 有内部 coolant HTC/gap/local refinement 字段；这不是顶/底 Robin。
5. **arbitrary normals**：P1i 只有 top/bottom/side 类别，缺少法向量、面面积和通用 BC 类型。
6. **adaptive/irregular mesh**：raw 交接数组是 dense structured grid，但论文的 mesh adaptation 依赖温度；final geometry 是否已受 target 影响必须由作者确认。
7. **variable geometry**：65×65 mm footprint、0.442–5.760 mm z extent、不同架构/层数，不属于冻结 P1i 10×10 mm 固定 stack。
8. **P(T)**：final q 含 leakage/refresh feedback，只能用于 Track A。
9. **k(T,σ)**：final k 含温度状态；stress 贡献无法从 handoff 复原。
10. **point count**：full field 最大 490,100；V7 的 1024 是 conditioning support，不是 full-field 上限，但 direct-query construction 是实际瓶颈。

## D. Quantitative range comparison

### D.1 MASS-HBM 实测 [RAW]

| 量 | 实测范围/分位数 |
|---|---|
| x/y physical extent | 65.0 / 65.0 mm，314/314 一致 |
| z extent | 0.442–5.760 mm，14 个取值 |
| field points | 59,150 / q25 236,600 / median 236,600 / q75 236,600 / 490,100 |
| shape | 14×130×130 占 260；17×130×130 占 40；另有 12 类 |
| final k，32-case 抽样 | 各轴观察 envelope 0.6–174.554 W/(m K) |
| final q，32-case 抽样 | 0–1.11875×10¹⁰ W/m³；per-case q99 envelope 1.542×10⁹–5.227×10⁹ |
| total power，314 cases | 184.712 / 388.057 / 570.735 / 791.671 / 1141.241 W（min/q25/q50/q75/max） |
| temperature field extrema | 全集最低 20.0000°C；全集最高 145.0484°C |
| ΔT per case | 15.5525 / 32.2275 / 46.8858 / 57.6154 / 125.0484°C |
| effective internal coolant HTC | 2,222.58–10,000 W/(m²K)，105 cases |
| minimum / mean coolant gap | 0.42–6.9 / 0.84–13.8 mm |
| final local Rint | 1.5801×10⁻⁷–1.6158×10⁻⁷ m²K/W，105 cases |
| calibrated per-interface R | 2.2505×10⁻⁶–3.3544×10⁻⁶ m²K/W，53 cases |
| coupled iteration count | 2 / 4 / 5.5 / 7 / 29 |
| case wall time | 7.789 / 17.868 / 30.752 / 71.912 / 418.902 s |

数组分位数是代表性 case 的确定性抽样，不是 314 case 全 voxel pooled quantile；因此报告明确写作“observed representative-sample envelope”，不伪装为总体分布。

### D.2 frozen V6-P1i [REPO]

| 量 | V6-P1i contract | 对比 |
|---|---|---|
| footprint | 10×10 mm | MASS 固定 65×65 mm，非同 domain |
| layer stack | 9 层，总厚 4.175 mm | MASS z 0.442–5.760 mm 且结构可变 |
| dense full field | 65×65×57 = 240,825 nodes | MASS 59,150–490,100 points |
| conditioning | 1,024 source/layout-aware support | 可作为 V8 起点，但 quotas 必须扩展 |
| local k envelope | 0.16–400 W/(m K) | MASS 抽样 0.6–174.554，数值落入不代表语义兼容 |
| q envelope | 1×10⁸–8×10¹⁰ W/m³ | MASS 抽样最高 1.119×10¹⁰，仍有 P(T) 语义差异 |
| BC | dual Robin + adiabatic side | MASS 含内部 distributed liquid sink |
| top/bottom h | top 500–1600；bottom 20–200 W/(m²K) | MASS external top 默认可达 30,000；internal HTC 2,223–10,000 |
| contact | perfect，R=0 | MASS Rint 非零且状态相关 |

结论：范围 overlap 只能说明数值尺度未必完全离谱，不能证明 geometry、BC、interface 或 nonlinear state semantics 兼容。

## E. Leakage audit

### E.1 泄漏链路判定

| 链路 | 存在性 | Track B |
|---|---|---|
| `T -> adaptive mesh -> input` | [PAPER] 明确存在；[RAW] final geometry 是否已适配需作者确认 | 禁止；须使用 target-independent base mesh |
| `T -> k(T,σ) -> input` | [PAPER]+[RAW] 存在，HBM temp-dependent cell fraction=1 | final k 禁止 |
| `T -> P(T) -> input` | [PAPER]+[RAW] power feedback/leakage/refresh 存在 | final q/P 禁止 |
| `T/stress -> Rint -> input` | [PAPER]+[RAW] final interface state 存在；stress field 缺失 | final Rint 禁止 |

### E.2 可用数据四分法

1. **deployable / pre-solve inputs**：target-independent geometry；material/region identity；reference/nominal k 或 law parameters；initial workload/activity/component power；external BC；液冷设计参数；nominal/calibration Rint；architecture/pitch/slab/group/thickness。
2. **converged solver-state inputs**：final k、final q、final Rint、final leakage/refresh/component power、T-guided mesh/placement。仅 Track A。
3. **labels / outputs**：Kelvin temperature field；Celsius 副本；peak/stack temperature scalar 可作辅助标签。
4. **metadata only**：iteration、residual、assembly/solve/runtime、backend、provenance/checksum。不得作为模型输入。

## F. Dataset split risk

314 条不是 314 个独立同分布随机样本，而是 geometry/pitch/slab/group/capacity/bandwidth/workload/cooling 的相关 sweep。Exp0 repeat 和同一 case 的 iteration state 尤其不能跨 split。随机 sample-level split 会让近邻设计点、重复 geometry 或同一 solver trajectory 泄漏到不同集合；随机 voxel split 更是禁止。

### F.1 group-aware 主方案

原子 group key：

```text
(source_case_identity,
 architecture, scan_family, geometry_case_id,
 pitch_um, slab_count, group_count,
 model_name, phase,
 cooling_policy, cooling_parameter_bin)
```

先构造完整 group，再对 group 做 deterministic hash，目标约 60/20/20 train/valid/test；必须先报告实际 group/case 数及架构/实验平衡，不能为凑比例拆 group。同一 solver case 的所有 iteration、repeat 派生和 field 必须在同一 split。

### F.2 OOD 套件

- **unseen geometry**：完整 hold out `geometry_case_id` / thickness family。
- **unseen pitch**：hold out 完整 pitch 值；边缘 50/500 μm 样本极少，应单独标注低支持，不混作稳定统计。
- **unseen slab count**：hold out 完整 layer/slab count。
- **unseen workload**：完整 hold out `model_name + phase`。
- **unseen cooling condition**：按 HTC/gap bins 分组；conventional 与 internal-liquid 分开。
- **cross-architecture**：leave-one-architecture-out；LC-V 必须视为独立 cooling/topology domain。

## G. Proposed experiment

### Track A — Oracle-field diagnostic

**目的**：检验模型在已给定收敛物理场时的 representation capacity。**禁止**用于 solver-replacement、deployability 或端到端 speedup claim。

- input：cell coordinates/volume、converged diag3 k、converged q、converged Rint edge/face field、external/internal BC、final mesh。
- target：converged temperature field K。
- adapter/model changes：V8 adapter；interface-aware edge；generic BC；variable-geometry batching；volume-weighted loss；oracle provenance mask。
- support：1024 support 可作为初值，但 quotas 至少覆盖 power source、material/interface、internal coolant、external boundary、bulk。
- query：所有 cell full-field；Sparse KD-tree 建邻域，按 geometry 缓存，可控 chunk 查询。
- split：与 Track B 共用 group-aware IID/OOD，便于测 oracle gap；绝不按 voxel 随机拆。
- metrics：J 节全部 accuracy 指标；efficiency 只描述模型表示路径，不与 MASS 完整求解链作替代性结论。

### Track B — End-to-end surrogate

**目的**：只凭部署前物理输入，直接预测 MASS-HBM converged temperature field；这是最终 accuracy-speed benchmark 主路线。

- input：target-independent geometry/volumes；material/region；reference/nominal k 或 `k(T,σ)` 参数；initial workload/activity/component power；nominal/calibration Rint；external BC；internal liquid cooling design/sink parameters。
- forbidden input：final k/q/Rint、final leakage/refresh power、T-guided final mesh/support/placement、convergence history。
- target：converged temperature field K。
- changes：新 V8 physics contract；component/workload schema；interface-aware edge/face feature；generic BC/internal sink；variable geometry；leakage validator。
- support/query：只依据 pre-solve geometry/physics 做 quotas；相同 geometry 多 workload 可缓存 neighborhood；full query 必须 Sparse KD-tree + cached neighborhood + chunking。
- split：group-aware IID + 六个 OOD suite。
- metrics：J 节全部 accuracy 与 efficiency 指标。

Track B 第一版无需显式复现自洽 iteration；它学习从 initial/deployable state 到 converged T 的映射。若以后需要可解释 iteration rollout，应另立 contract，不能把 final state 偷塞入输入。

## H. Architecture decision

五个候选中选择：

**5. 需要新的 V8 physics contract。**

这不意味着立即更换 RIGNO 主体。推荐保留 backbone，先更换物理 schema 与 adapter：

1. local node/cell：nominal diag3 k、initial q/activity、material/region、cell volume、pre-solve flags；若将来确有 full tensor，再扩 symmetric-6。
2. interface edge/face：`Rint`、normal、area、interface/material-pair ID、explicit vs homogenized representation。
3. generic BC：normal vector、BC type mask、h、T∞、internal sink conductance/temperature；不再只用 top/bottom/side。
4. global context：architecture、extents、pitch/slab/group/workload/cooling summary；重新冻结 V8 normalization，不沿用或暗中 refit V7 normalization。
5. provenance：每个 feature 带 `PRE_SOLVE / ORACLE / LABEL / METADATA` source class，loader fail-closed。
6. sampling/query：physics-stratified support、Sparse KD-tree、identical-geometry cache、chunked full-field query。

V7 contract 与 G1/G2 evidence 均保持不变；不得为 MASS-HBM 回写 frozen V7。

## I. Efficiency benchmark design

### I.1 完整计时边界

MASS-HBM：

```text
mesh/preprocess
+ multiphysics/self-consistent update
+ thermal assembly/solve
```

Heat3D：

```text
unit/schema validation + preprocess
+ support selection
+ graph/query construction
+ network forward
+ full-field query/postprocess
```

只报 model forward 会系统性低估 Heat3D 成本；只报 linear solve 会低估 MASS-HBM 成本。

### I.2 测量协议

- **cold-start**：新进程、无 graph/cache/JIT；分别列出加载、compile、graph 和 forward。
- **warm/cached**：JIT 已完成；相同 geometry cache 命中与新 geometry cache miss 分开报告。
- **throughput**：固定时间窗 cases/s，按 geometry reuse ratio 分层。
- **memory**：host RSS 与 accelerator live peak；不得用进程历史 cumulative peak 代替 phase live peak。
- **training cost**：GPU-hours、wall-hours、峰值/累计能耗（可得时）、试验总次数；失败 run 也计入工程成本。
- **break-even**：

```text
N_break_even = C_training / (t_MASS,warm - t_Heat3D,warm)
```

另报 cold-start 和含 adapter/工程投入的 sensitivity，不把开发成本混成单一精确数字。

### I.3 高 N 查询要求 [REPO + INFERENCE]

G2 已冻结 full-query latency boundary，必须包含 query-graph construction 与 direct-query forward。在 571,256 query 点的同机 profile 中，query graph construction 为 cold 4.445 s、steady median 3.923 s、p95 4.658 s；P15 valid128 整体约 1,008 s。它证明 graph/query 不是可忽略 overhead。

对 MASS-HBM 59k–490k 点：

- Sparse KD-tree：**必需**，避免 dense pairwise construction。
- cached neighborhood：**必需但只对完全相同 geometry 有效**；不得把 cache 命中混入 uncached latency。
- chunked query：**必需**，控制峰值内存与大图编译形状。

## J. Evaluation metrics

所有空间误差默认以 physical cell volume 加权；同时给出 per-sample distribution，避免大网格 case 支配 pooled 指标。

1. **sample-first relative RMSE**：

   ```text
   100 * mean_i sqrt(
       sum_j w_ij (T_pred - T_true)^2 /
       sum_j w_ij (T_true - T_ref_i)^2
   )
   ```

   必须完整写名，不简称为“RMSE”。`T_ref_i` 由预先冻结的 ambient/reference contract 给出。

2. **RMSE K**：每 sample volume-weighted Kelvin RMSE；明确 sample-first aggregate 与 pooled 值。
3. **MAE K**：每 sample volume-weighted Kelvin MAE。
4. **peak-temperature error**：最大温度的 signed 与 absolute K error。
5. **hotspot top-1% RMSE**：按真值 hottest 1% physical volume 定义区域，避免用预测选择区域。
6. **hotspot-location error**：真/预测最高温坐标欧氏距离 mm；并可报 top-1% centroid distance。
7. **75°C threshold**：point-level 与 case-level confusion、precision、recall、F1、margin error。
8. **100°C threshold**：同上；若正样本少，明确 support，不用 accuracy 掩盖 class imbalance。
9. **latency / throughput / peak memory**：按 I 节完整链、cold/warm/cache regime 报告。

## 2. 最终 compatibility matrix 与 GO 判定

| 路线 | 判定 | GO 类型 | 边界 |
|---|---|---|---|
| current V7 + adapter | **NO_GO** | none | 无法表达 Rint、internal cooling、generic normal、variable geometry 和 deployable nonlinear state |
| Track A oracle diagnostic | **CONDITIONAL_GO** | **model-schema-change GO** | 仅 V8 oracle-field diagnostic；必须有 claim firewall |
| Track B end-to-end | **CONDITIONAL_GO** | **model-schema-change GO** | 仅在 pre-solve contract、作者澄清、leakage gate、group split receipt 完成后 |
| formal training now | **NO_GO** | — | 当前任务只到 audit + protocol |
| Phase 2 smoke now | **NO_GO** | — | 先完成 Phase 1，再取得新授权 |

不存在 adapter-level GO。RIGNO backbone “可保留”与“模型 schema 无需变化”不是同一件事。

## 3. 需要向 MASS-HBM 作者确认的 UNKNOWN

1. 提供 stress/strain 场、单位、node/cell/face 位置，以及 initial 还是 converged；当前 handoff manifest 中没有。
2. 提供 `k(T,σ)`、`Rint(T,σ)`、leakage `P(T)`、refresh feedback 的函数形式与参数，以便构造 deployable nominal inputs。
3. 确认每个 `power_density_map.npy` 是否都是 iteration-final；为每个 case 指定 canonical iteration-0 volumetric map。
4. 确认 `solver_geometry.json` 是 pre-solve base mesh 还是 temperature-guided adaptation/local refinement 后的 final mesh；若不同，提供 target-independent base mesh。
5. 说明 `final_interface_tbr_map.npz` 的 face indexing/orientation，尤其是 `rotated_homogenized_tensor` case 如何映射到物理界面。
6. 确认 LC-V sink field 是纯 design input，还是包含 converged-temperature controller update；提供 pre-solve sink contract。
7. 解释唯一 `interface_review_status=REVIEW_REQUIRED` case；澄清前不得进入任何 split。
8. 提供可比较的 mesh、coupled update、thermal solve 分阶段计时与硬件信息。

## 4. 下一阶段最小实施方案

### Phase 1 — 数据 adapter（不训练）

- 只读 index/case loader；显式单位与 axis/location schema。
- 构造 pre-solve 与 oracle 两份互斥 view；feature provenance fail-closed。
- 重建 cell centers、volume、structured edges、interface faces、generic BC、initial power。
- 生成 group manifest 与 OOD membership；验证 iteration/case 不跨 split。
- 1–3 case 的纯数据 round-trip、功率守恒、体积加权和 target-alignment 检查。

退出 gate：0 个 unit guess、0 个 clipping、0 个 oracle feature 进入 Track B、0 个 dataset write。

### Phase 2 — 1–3 sample end-to-end smoke

需新的明确训练授权。只验证 forward/loss/gradient finite、full-field chunk query 和 provenance gate；不选 checkpoint、不报告 accuracy claim。

### Phase 3 — small pilot

小规模 group-disjoint Track A/Track B；完成 metrics 与 oracle gap，验证 V8 schema 是否足够。

### Phase 4 — formal training

冻结 V8 contract、split receipt、normalization、seed、resource budget 后才允许正式训练。

### Phase 5 — accuracy–latency benchmark

在 matched hardware 上比较完整 MASS-HBM 与 Heat3D 链，给出 cold/warm/throughput/memory/training cost/break-even。

## 5. 最重要的五个兼容性结论

1. final `k/q/Rint` 与可能的 final adaptive mesh 都含 target/solver-state 信息；Track B 使用即泄漏。
2. MASS-HBM 的 nonzero interface Rint 与 LC-V internal distributed sink 无法映射到 P1i 的 perfect contact + dual Robin flags。
3. raw final k 是 diagonal3，当前样本未见 full-6；但 arbitrary interface normal 和 homogenized orientation 仍要求 edge/face schema。
4. MASS-HBM 65×65 mm、59k–490k points、四架构/多 sweep 与固定 10×10 mm P1i domain 不同；范围重叠不是 physics compatibility。
5. 最小正确路线是保留 RIGNO backbone、建立独立 V8 physics contract，并把 Sparse KD-tree、geometry cache、chunked query 纳入正式 benchmark。
