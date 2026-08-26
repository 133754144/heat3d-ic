# Heat3D-IC 面向论文发表的审稿评估与 V7 整改计划

> 审稿视角：计算机科学与集成电路/EDA 领域
>
> 审计日期：2026-08-26
>
> 审计基线：`0812d282b46d69de4b75fea86afa153dff98bdc6`
>
> 审计范围：当前 `main`、研究分支中的 V6/P1i 证据、历史文档、训练/推理调用链，以及 devbox 的同版本代码。本文为只读审稿审计，不包含训练、推理、求解、数据生成或结果重算。

## 结论先行

如果按当前代码和证据直接投稿，建议判断如下：

| 方向 | 当前结论 | 审稿人最可能的拒稿理由 |
|---|---|---|
| NeurIPS / ICML / ICLR 等计算机顶会 | Reject，约 3/10 | 主要贡献仍表现为 RIGNO 的 3D IC 领域适配；缺少通用 PDE 证明、强基线、离散收敛、OOD 与独立方法创新 |
| DAC / ICCAD | Weak Reject，约 4/10 | 问题高度相关，但缺少同口径竞品、工业/公开案例、设计优化及物理一致性验证；严格端到端加速约 2× |
| DATE / IEEE TCAD / TCPMT | Major Revision 后有现实机会 | V6/P1i 已有较完整的全场、热点、界面、运行时和实验治理证据，但代码分层和证据发布仍不足 |
| 一般工程或应用会议 | 可投稿 | 但会掩盖项目已形成的方法和工程价值，不建议在当前状态急投 |

核心判断不是“模型效果差”，而是“研究已经进入 V6，论文和代码的可信表达还停留在历史 V4/V1”。当前最需要解决的是：

1. 将研究贡献从版本历史中提炼为可检验的科学问题；
2. 统一一套论文主数据、指标和端到端计时口径；
3. 清除正式训练/推理对 smoke、check、development 脚本的运行时依赖；
4. 补齐横向基线、几何/物理 OOD、界面物理指标和真实设计应用。

## 1. 审计基线与当前项目状态

当前本地工作树的 HEAD、`origin/main` 和 devbox 代码均为 `0812d28`。本地工作树没有未提交修改；devbox 另有未跟踪的 `configs/heat3d_v6_supplemental/`，不能把它作为 `main` 的版本化论文证据。

README 仍然将项目描述为“V4 closeout”，并把 V5 写成未来路线，见 [README.md](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/README.md:5)。实际 `main` 已包含 V6/P1h、V6/P1i 及 supplemental publication closeout，因此 README、入口命令和当前研究状态不一致。

当前项目可以准确描述为：

```text
(k_x, k_y, k_z, q, 双 Robin 边界, 几何/层信息)
    -> 稳态温升场 DeltaT(x) / 温度场 T(x)
```

其主要路线是：

```text
1,024 source-aware 条件锚点
    -> RIGNO 图算子和 global/shape-scale context
    -> 16,384 查询点
    -> layer/interface-aware reconstruction
    -> 240,825 节点的 FVM 对齐全场
```

V6 阶段索引已经记录了从 P1g 几何去混淆、P1h 共享支持到 P1i 连续物理 random-block 的演进，[见阶段索引](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/docs/v6_phase_index.md:7)。

## 2. 当前可用于论文的研究成果

### 2.1 P1h/P1i 数据与模型角色

V6 不是一个全局唯一的数据集，而是角色分层：

| 角色 | 数据集 | 模型/检查点 | 论文定位 |
|---|---|---|---|
| V6 layer | `heat3d_v6_p1h_shared_support1024_v0` | `V6_03_V5best_P1h`，seed 0，e111 | 分层共享支持和高分辨率重建 |
| V6 formal random-block | `heat3d_v6_p1i_continuous_physics1024_v1` | `V6_06_V5best_P1i_seed0_reliable_B24`，e559 | 正式随机块物理分布、主论文候选 |

P1i 的正式 split 为 `768/128/128`（train/valid/test），并假设完美界面接触 `R_contact=0`。当前证据没有覆盖可变接触热阻、实验封装、普适几何 OOD 或校准不确定性。[V6 scientific closeout](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/docs/v6_p1i_closeout.md:9)

### 2.2 建议作为主表的数据

当前最适合作为论文主结果的是 P1i、`model_seed0`、E16384 reconstruction。指标必须保留原名称，不能把 point-global relative RMSE、sample-first CV-relative RMSE 和 raw CV RMSE 混写成一个“RMSE”。

| 角色 | point-global relative RMSE | sample-first CV-relative RMSE | raw CV RMSE | source RMSE | peak RMSE | interface RMSE |
|---|---:|---:|---:|---:|---:|---:|
| valid32 | 2.7023% | 2.7385% | 2.2848 K | 3.8721 K | 4.0178 K | 0.3857 K |
| test128 | 2.9920% | 2.9485% | 2.3891 K | 3.9405 K | 5.7263 K | 0.3555 K |

严格公共性能边界是“内存中的 `k/q/BC` → 同步得到 240,825 节点结果”。WSL2 Attempt 4 的主要时序为：

- fresh median/p95：`0.8832/0.9830 s`；
- Q2 吞吐：`1.7263 sample/s`；
- 相对 persistent CPU FVM 的配对 fresh/Q2 加速：`2.001×/2.005×`。

完整主表、测试结果和指标口径见 [P1i P6-A 主表](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/docs/v6_p1i_p6a_publication_tables.md:7)。

### 2.3 次级证据

- P1h 三 seed、16,384 点重建：全场 raw RMSE `1.2236±0.0934 K`，全场相对误差 `3.0174±0.2304%`。[P1h inference closeout](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/docs/v6_production_inference_final_closeout.md:5)
- 研究分支保留了 P1i 三 seed valid128 证据，全场 point-global relative RMSE 为 `3.442626±0.058435%`。但 `main` 的 publication compact table 仍全部是 `model_seed0`，这必须在投稿前统一。
- 固定拓扑、新 `k/q` 的 supplemental 结果为 E/U reconstruction 约 `8×—9×` runtime 加速；它只覆盖四个 train-only 几何、一次随机工作负载顺序、没有标签和准确率。[supplemental closeout](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/docs/v6_supplemental_known_topology_publication_closeout.md:5)

### 2.4 当前可以主张与不能主张的内容

可以主张：

- 在冻结的 P1i generator、材料/边界/热源范围和 reconstruction 协议内，E16384 在 test128 上获得约 `2.99%` point-global relative RMSE；
- 在指定 WSL2/FVM 公共边界内，E16384 有约 `2×` 配对端到端加速；
- source-aware anchors、查询分辨率和界面感知重建是可重复研究对象；
- 峰值误差尾部已经被识别并公开，而不是被平均指标掩盖。

不能主张：

- 普适任意几何泛化或工业 sign-off accuracy；
- `8×—9×` 的全场准确仿真加速；
- 优于 FVM 的物理精度；
- 已解决 OOD、可变接触热阻、实验封装验证或 uncertainty quantification；
- 已经优于 DeepOHeat、ARO、GINO、Transolver 或 vanilla RIGNO。

## 3. 分领域审稿评价

### 3.1 计算机科学顶会

RIGNO 已经是 NeurIPS 2025 的任意域图神经算子方法，并主张点云、跨分辨率和复杂域泛化。[RIGNO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dcb91f43033bb1d367d1848806dee98d-Abstract-Conference.html)

因此，以下内容本身不能构成新方法：

- 在点云上使用 GNN；
- 处理 3D 几何；
- 从较低点数推理到较高点数；
- 在单一热传导数据集上取得较低误差。

如果要冲 NeurIPS/ICML/ICLR，V7 必须把问题提升为：

> 面向不连续导热系数、稀疏局部源和界面条件的椭圆型 interface PDE，构造 source-aware、measure-consistent 且跨离散分辨率稳定的图神经算子。

这要求多 PDE 基准、GINO/Transolver/FNO/DeepONet/vanilla RIGNO 等基线、采样误差收敛曲线、算子误差与 reconstruction floor 分解，以及界面/守恒性质的理论或严格数值解释。否则审稿人会把工作归类为“RIGNO 的高质量领域工程应用”。

### 3.2 集成电路与 EDA 顶会

EDA 方向更适合当前项目，因为问题、指标和 FVM 对照都具有明确芯片热分析语义。但 DAC/ICCAD 近年的直接竞争已经包括：

- DeepOHeat：面向功率图、边界条件和换热系数变化的 3D IC 热算子，并报告了工业设计和大幅加速；[DeepOHeat](https://arxiv.org/abs/2302.12949)
- ARO：强调跨电路 transfer、稳态/瞬态、多保真和主动学习，并在 unseen circuits 上与 MTA 比较；[ARO](https://eprints.whiterose.ac.uk/id/eprint/225574/)
- DeepOHeat-v1：增加多尺度表达、可信度评分以及与 FD/GMRES 的混合优化；[DeepOHeat-v1](https://arxiv.org/abs/2504.03955)

当前项目的优势是 source-aware 稀疏支撑、异质场、全场 reconstruction、分区/界面指标和严格时序边界；不足是没有同口径 SOTA 对照、真实/公开设计验证、热感知设计闭环和接触热阻实验。

## 4. 距离发表还欠缺的内容

以下不是“可选增强”，而是审稿人判断论文是否成立所需要的证据。

### P0：必须在投稿前完成

1. **统一论文主数据和主指标**

   - 指定 P1i 为主实验，P1h 作为分层/重建补充；
   - 将 seed 0/1/2 的均值、标准差、置信区间纳入主表；
   - 统一 point-global、sample-first、raw、source、peak、interface 的定义；
   - 将历史 V4/V5 结果标为跨数据集历史证据，禁止直接横向比较。

2. **补齐同口径基线**

   至少包含 vanilla RIGNO、random/uniform/volume-only anchors、FNO、DeepONet/DeepOHeat、GINO、Transolver 和一个 ML chip thermal solver。所有基线必须使用同一 P1i split、标签预算、训练预算、参数量报告和 wall-clock 边界。

3. **补齐方法消融**

   - 去掉 source-aware anchors；
   - 以均匀、volume-only 或随机 anchors 替代；
   - 去掉 global physics context；
   - 去掉 shape-scale 分支；
   - 去掉 layer/interface-aware reconstruction；
   - 直接高分辨率输出；
   - 对 `k` 各向异性、稀疏热源和边界条件分别做消融。

4. **补齐泛化和失效实验**

   - 几何 OOD：层数、长宽比、source 数量/位置、random-block 拓扑；
   - 物理 OOD：导热率、热功率、Robin 系数范围外推；
   - 可变 `R_contact`、TSV/microbump/BEOL 或其受控代理；
   - 预注册的 sealed IID 终检；
   - 报告失败案例、峰值尾部和适用域，不得只给平均误差。

5. **补齐物理指标**

   除对 FVM 的温度误差外，至少报告离散能量残差、界面热流连续性、Robin 边界残差、总功率守恒和最大温度误差。否则论文仍是“监督回归”，而不是可信的物理 surrogate。

6. **清理正式运行链中的 smoke 语义**

   正式训练、推理的代码、日志、checkpoint provenance 必须全部写成 publication/production 语义，不能继续出现 “smoke”、“not formal model performance” 的自相矛盾记录。

### P1：强烈建议在投稿前完成

1. 增加一个公开或工业风格的 3D IC/package case；
2. 增加热感知布局、功率分配或热约束优化案例；
3. 报告参数量、显存、CPU/GPU、编译时间、缓存/非缓存时间和 amortized cost；
4. 对 1024/4096/8192/16384/240825 点数给出精度—延迟—内存 Pareto 曲线；
5. 区分“一次新案例端到端成本”和“已知拓扑新物理条件成本”；
6. 在干净 checkout 和新环境中提供一条可复现的最小路径。

### P2：可作为后续工作

- 瞬态热传导、温度依赖材料参数和 leakage feedback；
- uncertainty quantification 与 hotspot trust gate；
- 与 FVM/GMRES 的选择性混合修正；
- 面向多芯片/多封装的 transfer learning 或少样本适配；
- 工艺级真实 TSV、microbump、界面热阻和冷却结构。

## 5. 待解决问题清单

| ID | 问题 | 审稿风险 | 解决标准 | 优先级 |
|---|---|---|---|---|
| R-01 | README、入口命令仍描述 V4/V5 | 审稿人无法判断当前 canonical experiment | README 与 V6 phase index、主配置、主表一致 | P0 |
| R-02 | 正式训练仍调用 V1 smoke runner | provenance 与论文语义冲突 | 独立 V6 training engine，运行元数据不再出现 smoke | P0 |
| R-03 | production inference 导入训练巨型脚本和 V3 smoke hook | 代码复现脆弱，私有 API 改动会静默破坏 | runtime 只依赖 `rigno/` 稳定库模块 | P0 |
| R-04 | P1i 主表为 seed0，三 seed 证据分散在研究分支 | 统计显著性和可重复性不足 | 主表报告三 seed mean±std/CI，明确 model seed 与 lifecycle seed | P0 |
| R-05 | 没有 vanilla RIGNO/GINO/Transolver/DeepONet 等同口径比较 | 无法证明方法贡献或 SOTA 地位 | 固定 split、预算、硬件和指标重跑 | P0 |
| R-06 | 只有 IID 和有限 random-block，缺少几何/物理 OOD | “泛化”结论不成立 | 预注册 OOD 矩阵，报告平均和尾部 | P0 |
| R-07 | `R_contact=0` 固定 | 物理适用范围过窄 | 加入接触热阻 sweep 和界面热流误差 | P0 |
| R-08 | 没有能量/热流/Robin 残差 | 只能称回归误差，不能称物理可信 | 发布守恒和边界 residual 指标 | P0 |
| R-09 | 2× E2E 与 8×—9× known-topology runtime 容易被混写 | 速度主张可能被审稿人判定为口径不公 | 主文只放公共 E2E，supplemental 明确 runtime-only | P0 |
| R-10 | 无设计优化闭环 | EDA 价值停留在预测器 | 完成布局/功率/热约束优化并与 FVM 比较 | P1 |
| R-11 | synthetic-only | 工业有效性不足 | 增加公开 HotSpot/PACT/3D-ICE 或工业风格案例 | P1 |
| R-12 | 没有标准测试、CI、依赖边界 | artifact 审查风险高 | `pyproject.toml`、tests、CI、import policy | P1 |
| R-13 | V1 数据类被 V6 adapter 继续使用 | 版本边界模糊，维护成本高 | 使用版本无关领域对象，V1 仅留兼容 adapter | P1 |
| R-14 | 模型默认 `qk_region_feature_version='bugged_v1'` | 漏配配置时静默进入错误语义 | 未显式声明直接 fail-closed | P0 |
| R-15 | 源码中仍跟踪 `output/heat3d_ic/heat3d_operator_best.pkl` | 代码与 artifact 管理政策不一致 | 移出源码发布路径，使用 release/artifact manifest | P1 |

## 6. V7 阶段建议与验收门槛

V7 不应只是“再训练一个更大的模型”，而应是 publication-readiness phase。建议拆成以下五个 gate。

### V7.0：论文问题与证据冻结

**目标**：让论文只有一条主叙事。

交付物：

- 一页 paper contract：问题、输入/输出、贡献、主数据、主指标、不可主张内容；
- P1i 主表和三 seed 汇总；
- P1h、旧 V4/V5、supplemental 的角色标签；
- README、phase index、配置、论文表格一致。

验收：任何主文数字都能追溯到唯一 dataset/checkpoint/code/protocol hash。

### V7.1：V6 runtime 解耦

**目标**：清除正式路径中的 V1/V3 smoke 依赖。

建议新建或整理：

```text
rigno/training/engine.py
rigno/runtime/checkpoint.py
rigno/runtime/features.py
rigno/runtime/grouping.py
rigno/eval/metrics.py
rigno/eval/reconstruction.py
scripts/train_heat3d_v6.py
scripts/infer_heat3d_v6.py
```

验收：

- production/import graph 不再依赖 `scripts/check_*`、`*_smoke.py`、`*_development.py`；
- 不再通过 `sys.path` 注入脚本目录解决包导入；
- 不再跨脚本调用私有 `_...` API；
- 同一 checkpoint 在旧兼容路径和新 V6 路径上的输出误差低于预注册阈值；
- 训练/推理 provenance 使用 `experiment_role=publication_training` 或 `production_inference`。

### V7.2：基线与机制消融

**目标**：证明性能来自 source-aware/context/reconstruction 机制，而不是训练预算或数据便利。

验收：

- vanilla RIGNO、random/uniform/volume-only anchors、FNO/DeepONet/GINO/Transolver 均有结果；
- 所有模型使用同一 P1i split、相近参数量和统一计时边界；
- 每个新模块都有去除实验；
- 给出 anchor/query 数量的精度—延迟曲线和 sampling floor。

### V7.3：物理与 OOD 可信度

**目标**：把“热场回归”提升为“受物理约束的热算子”。

验收：

- 几何 OOD 和物理 OOD 均有预注册 split；
- 可变接触热阻至少有一组受控实验；
- 报告总功率守恒、界面热流、Robin 残差、peak error tail；
- sealed IID 在所有选择和调参后才打开；
- 每个失败案例有解释，不用平均指标掩盖尾部错误。

### V7.4：EDA 应用闭环

**目标**：说明加速确实改善芯片设计流程。

验收：

- 至少一个热感知布局、功率分配或热约束优化任务；
- 与 FVM/PACT/HotSpot/3D-ICE 或公开 benchmark 的误差和 wall-clock 对比；
- 报告一次推理、批量 amortized 和优化全流程三种成本；
- 显示 peak temperature、hotspot location、source-region error 与设计目标之间的关系。

### V7.5：可复现发布

**目标**：使论文 artifact reviewer 能在干净环境中复核主结果。

验收：

- `pyproject.toml` 或等价依赖锁定；
- 标准 `tests/` 与 CI；
- 一条最小 dry-run 和一条 checkpoint inference 命令；
- 所有外部数据/大文件只通过哈希 manifest 和下载说明管理；
- 不提交 `data/`、`output/`、`checkpoints/`、`logs/` 和预测大文件；
- 主结果表由 machine-readable artifact 自动生成，避免手工抄写。

## 7. 正式代码调用链审稿意见

### 7.1 训练链

当前正式路径是：

```text
scripts/run_heat3d_v4_config.py
    -> rigno/heat3d_v2_runner_command.py
    -> scripts/run_heat3d_v4_controlled_training.py
    -> scripts/run_heat3d_v1_medium_controlled_training_export.py
```

关键证据：

- `run_heat3d_v4_config.py` 省略 `--config` 时默认使用 V5 `V4P5_42_canonical.yaml`，[见启动器](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/scripts/run_heat3d_v4_config.py:30)；
- command builder 将 V1 controlled training export 设为训练脚本，[见 builder](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/rigno/heat3d_v2_runner_command.py:18)；
- V4 wrapper 明确写着保留 legacy V1 runner，最终调用 `legacy_runner.main()`，[见 wrapper](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/scripts/run_heat3d_v4_controlled_training.py:2)；
- V1 runner 文件头、CLI、启动日志和 `run_config` 都把自身描述为 smoke/diagnostic，而不是正式模型训练，[见文件头](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/scripts/run_heat3d_v1_medium_controlled_training_export.py:1)、[启动日志](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/scripts/run_heat3d_v1_medium_controlled_training_export.py:6086)、[run_config](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/scripts/run_heat3d_v1_medium_controlled_training_export.py:9142)；
- V1 runner 还从 `check_heat3d_v1_small_train_valid_smoke.py` 导入 `MODEL_CONFIG` 和私有函数，[见导入](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/scripts/run_heat3d_v1_medium_controlled_training_export.py:38)。

因此这不是“旧脚本尚未删除”，而是正式 publication training 仍经过 smoke 语义。它会使审稿人无法判断某次运行究竟是正式训练、诊断运行还是历史兼容路径。

### 7.2 推理链

V6 production inference 直接导入多个 evaluator 脚本、V1 训练 runner 和 V3 checkpoint smoke hook，并调用 V1 runner 的私有 `_make_*`、`_attach_*`、`_model_apply` 等函数，[见推理入口](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/scripts/run_heat3d_v6_production_highres_inference.py:23)。

Supplemental runner 又依赖名为 development 的 high-N 脚本和其私有函数，[见 supplemental runner](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/scripts/run_heat3d_v6_supplemental_publication_known_topology.py:23)。V6 数据 adapter 继续使用 V1 数据类和 legacy bridge，[见 V6 adapter](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/rigno/heat3d_v6_dataset.py:18)。这些兼容关系可以在 V7 暂时保留，但必须被收口在清晰的 adapter 层，不能成为 production runtime 的主接口。

另外，RIGNO 默认 `qk_region_feature_version='bugged_v1'`，[见模型默认值](/Users/xuyihua/.codex/worktrees/b2f3/3D%20IC%20Heat/rigno/models/rigno.py:1313)。正式配置虽然覆盖它，但漏配时应直接失败，而不是静默回退到已知错误语义。

## 8. 建议的论文行文逻辑

### 8.1 研究问题

建议围绕四个问题组织：

1. source-aware anchor 是否比均匀/volume-only 支撑更能覆盖稀疏热源？
2. global physics/shape-scale 分解是否降低跨样本温升幅值误差？
3. layer/interface-aware reconstruction 能否在不重训时稳定扩展至 240,825 节点？
4. 在统一端到端边界下，准确率—延迟 Pareto 是否优于数值 solver 和神经算子基线？

### 8.2 论文贡献

可以压缩为四个可检验贡献：

1. 标签无关的 source-aware 条件锚点；
2. 面向异质温升幅值和形状的物理尺度分解；
3. 查询点—层/界面重建的全场部署路径；
4. 带误差尾部、物理指标和统一时序边界的可信部署协议。

### 8.3 论文中必须避免的表达

- “arbitrary geometry”除非真正加入未见几何实验；
- “industrial sign-off”除非有工业/实验验证；
- “8×—9× end-to-end speedup”除非同时包含准确率和新案例公共边界；
- “SOTA”除非完成直接基线；
- 将 P1h 和 P1i 的数值直接比较；
- 将 point-global relative RMSE、sample-first CV-relative RMSE、raw CV RMSE 简化称为同一个 RMSE。

## 9. 可用于论文背景的综述性总结

三维集成通过垂直堆叠提高器件密度并缩短互连，但也增加单位面积功率密度、纵向热阻及层间热耦合，使热点、温度梯度和界面散热成为设计可靠性的关键约束。HotSpot 代表了适合早期架构探索的紧凑热模型；3D-ICE 将紧凑建模扩展到带层间微流道冷却的 3D IC；PACT 则支持从标准单元到体系结构层级的稳态、瞬态及多种新型冷却技术。这些工作说明，工程热分析必须同时考虑速度、物理可信度、封装结构和冷却边界，而不只是预测平均温度。[HotSpot](https://www.cs.virginia.edu/~skadron/Papers/hotspot_tvlsi06.pdf)、[3D-ICE](https://research.ibm.com/publications/3d-ice-fast-compact-transient-thermal-modeling-for-3d-ics-with-inter-tier-liquid-cooling)、[PACT](https://ieee-ceda.org/media/pact-extensible-parallel-thermal-simulator-emerging-integration-and-cooling-technologies)

神经算子把单个 PDE 解的拟合提升为输入函数到解函数的映射。DeepONet 和 FNO 奠定了数据驱动算子学习框架；GINO、Transolver 和 RIGNO 进一步处理复杂几何、点云和跨分辨率问题。3D IC 专用方法已经形成直接竞争：DeepOHeat 学习功率图、边界条件和换热系数到温度场的映射；ARO 强调跨电路迁移、稳态/瞬态、多保真和主动学习；DeepOHeat-v1 又加入多尺度表达、可信度评分及与数值求解器结合的优化流程。[FNO](https://openreview.net/pdf?id=c8P9NQVtmnO)、[GINO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)、[Transolver](https://proceedings.mlr.press/v235/wu24r.html)、[RIGNO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dcb91f43033bb1d367d1848806dee98d-Abstract-Conference.html)

因此，当前真正有价值的研究缺口不是“能否用神经网络预测温度”，而是在异质、各向异性和材料系数不连续的多层结构中，如何用有限且标签无关的条件支撑覆盖稀疏热源，如何将低维算子表示稳定重建到高分辨率全场，以及如何在统一端到端边界下同时报告热点、热源区、界面误差、物理残差和运行时间。Heat3D-IC 应围绕这一缺口定位，而不是泛泛地声称“神经算子比 FVM 快”。

2026 年的 Therm-FM 和 DeepOHeat-v2 预印本进一步把门槛推向跨芯片少样本适配、高对比度界面、可信度门控和设计优化，因此 V7 不能只继续增加模型容量。[Therm-FM](https://arxiv.org/abs/2605.22663)、[DeepOHeat-v2](https://arxiv.org/abs/2608.16080)

## 10. 参考文献方向

1. Huang et al., “HotSpot: A Compact Thermal Modeling Methodology for Early-Stage VLSI Design,” IEEE TVLSI, 2006. DOI: `10.1109/TVLSI.2006.876103`。
2. Sridhar et al., “3D-ICE: Fast Compact Transient Thermal Modeling for 3D ICs with Inter-Tier Liquid Cooling,” ICCAD, 2010。
3. Yuan et al., “PACT: An Extensible Parallel Thermal Simulator for Emerging Integration and Cooling Technologies,” IEEE TCAD, 2022. DOI: `10.1109/TCAD.2021.3079166`。
4. Lu et al., “Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators,” Nature Machine Intelligence, 2021. DOI: `10.1038/s42256-021-00302-5`。
5. Li et al., “Fourier Neural Operator for Parametric Partial Differential Equations,” ICLR, 2021。
6. Li et al., “Geometry-Informed Neural Operator for Large-Scale 3D PDEs,” NeurIPS, 2023。
7. Wu et al., “Transolver: A Fast Transformer Solver for PDEs on General Geometries,” ICML, 2024。
8. Mousavi et al., “RIGNO: A Graph-based Framework for Robust and Accurate Operator Learning for PDEs on Arbitrary Domains,” NeurIPS, 2025。
9. Liu et al., “DeepOHeat: Operator Learning-based Ultra-fast Thermal Simulation in 3D-IC Design,” DAC, 2023。
10. Wang et al., “ARO: Autoregressive Operator Learning for Transferable and Multi-fidelity 3D-IC Thermal Analysis with Active Learning,” ICCAD, 2024。
11. Yu et al., “DeepOHeat-v1: Efficient Operator Learning for Fast and Trustworthy Thermal Simulation and Optimization in 3D-IC Design,” IEEE TCPMT, 2026。

## 11. 审稿审计记录

- 本地 HEAD、`origin/main`、devbox 版本：`0812d28`；
- 本地工作树：无未提交修改；
- 远端：SSH GitHub remote，未执行写操作；
- 本次未运行训练、验证推理、FVM 求解、数据生成或结果分析脚本；
- 当前仓库没有标准 `.github/workflows`、`tests/`、`pyproject.toml` 等完整测试发布结构；
- 当前 V6 集成清单虽然有 allowlist/denylist 和 closeout checker，但尚未解决运行时仍依赖 smoke/legacy 脚本的问题；
- devbox 的未跟踪 supplemental 目录应在正式发布前清理或明确隔离；
- 该文档本身是 V7 规划和投稿准备的控制文档，不替代正式实验结果、训练日志或数据 manifest。

## 最终建议

V7 的第一目标应是“出版可信度”，第二目标才是“模型性能”。在完成 V7.0–V7.2 之前，不建议继续扩大模型或宣称新的速度纪录；在完成 V7.3–V7.5 之前，不建议以“通用神经算子”或“工业热 sign-off”名义投稿。

最现实的投稿路径是：先形成一篇以 P1i 为主、source-aware full-field thermal surrogate 为主题的 EDA 论文，目标 DATE/TCAD/TCPMT；待补齐跨域 PDE、强基线和理论/离散收敛证据后，再考虑计算机科学顶会版本。
