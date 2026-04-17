# 3D IC 热仿真数据库与 RIGNO 迁移方案

## 1. 先给结论

如果你的目标是研究 3D IC 中由于 TSV、微凸点、互连线和界面材料带来的热导率分布不均问题，同时又想把 [RIGNO](https://arxiv.org/html/2501.19205v2) 迁移到热仿真上，那么最可行的路线不是：

1. 一开始就对整片 3D IC 显式刻画每一根 TSV、每一个 micro-bump、每一条 BEOL 互连线。
2. 直接把现有 Heat3D 的简单两材料数据集放大成“更复杂一点”的版本。

更合理的方案是做成一个三级数据库，并把学习目标定义为“低保真到高保真的热场修正”：

1. `微结构等效库`：学习 TSV、micro-bump、局部互连单元的几何参数到等效热导率张量、界面热阻参数的映射。
2. `芯片级多保真样本库`：在整芯片/多层堆叠尺度上存储低保真、中保真、高保真热场，并保证它们来自同一个几何与功率配置。
3. `迁移实验索引库`：明确哪些样本属于源域、目标域、跨结构迁移、跨材料迁移、跨分辨率迁移。

推荐把最终模型做成：

`T_high = T_low + DeltaT_RIGNO(...)`

而不是直接让 RIGNO 从头预测 `T_high`。

原因是：

1. 文献已经表明，异构热导率问题的主流处理方式不是“显式解每条金属线”，而是“等效化 + 物理约束 + 多保真修正”。
2. RIGNO 的优势在任意域、点云/图表示和跨分辨率泛化，不是自带 3D IC 微结构热等效模块。
3. 低保真基线加残差修正，最适合把 ARO、T-Fusion、SAU-FNO、DeepOHeat-v1 的思路和你现有 Heat3D-RIGNO 代码基础接起来。

## 2. 现有论文到底怎么处理热导率不均

结合你当前目录里的报告和论文，现有方法大致分为五类。

### 2.1 直接把热导率场当作函数输入

这条路线以 DeepOHeat 为代表。它的核心思想是：材料导热率分布本身就是 PDE 配置的一部分，可以和功率图、边界条件一起作为神经算子的输入。

这类方法的优点：

1. 表达最直接。
2. 理论上可以处理空间变化的材料场。

这类方法的短板：

1. 如果把 TSV、bump、BEOL 线网全都显式映射到细网格，数据生成代价会很高。
2. 强跳变界面附近容易出现局部误差。
3. 对复杂几何和细尺度结构，规则网格表示不够友好。

### 2.2 先做等效热导率提取，再做大尺度热场学习

这条路线对你最重要。PI-ONet 和 TTSV bounded NN 基本都属于这一类。

它们的共同做法是：

1. 先对单个 TSV、bump 或局部单元做高保真 FEM 仿真。
2. 提取水平/垂直等效热导率，必要时形成各向异性热导率张量。
3. 在更大尺度的芯片仿真中，把 TSV 层或 bump 层当作等效均匀层或各向异性层。

PI-ONet 的做法尤其清晰：

1. 以单个 TSV 或 bump building block 为最小对象。
2. 用 FEM 提取 `k_xy_eq` 和 `k_z_eq`。
3. 用一个小型 ANN 学习“几何参数 -> 等效热导率”。
4. 再把整层 TSV/bump 阵列替换成等效层。
5. 在大尺度模型里再做热场预测。

TTSV bounded NN 的意义是进一步说明：

1. 对 TTSV 这种强各向异性结构，等效热导率预测本身就是一个独立且值得建模的任务。
2. 水平等效热导率往往最难学，加入物理边界约束会更稳。

### 2.3 在材料界面显式加入物理约束

PI-ONet 强调了这一点。对于多层 chiplet、TSV 层、bump 层，材料交界面上至少要保证：

1. 温度连续。
2. 热流连续。

如果模型只学习整体温度场，但不管界面条件，那么平均误差可能不大，局部热点和界面热流却可能不可信。

这对你后面建库的启发是：数据库里不能只存温度场，最好把界面位置、材料分区、等效参数和边界条件一起存下来。

### 2.4 用几何感知算子处理复杂结构

RIGNO、GINO 这类工作在这里有价值。它们的意义不在于替代等效建模，而在于：

1. 允许输入点云、非规则网格、任意域。
2. 更适合 chiplet、异构堆叠、非规则轮廓、局部细化采样。
3. 有机会比规则网格算子更自然地表达局部结构变化。

换句话说，RIGNO 更适合作为“全局热场学习器”，而不是“微结构物理提取器”。

### 2.5 用多保真、迁移学习和混合求解降低代价

ARO、T-Fusion、SAU-FNO、DeepOHeat-v1 都说明了一个现实：

1. 高保真热仿真太贵。
2. 直接靠大量高保真样本训练不经济。
3. 多保真、主动学习、迁移学习和求解器修正会长期存在。

其中：

1. ARO 把多保真和主动学习绑在一起，减少高保真样本需求。
2. T-Fusion 说明少量高保真加大量低保真也能把误差压得很低。
3. SAU-FNO 明确使用 transfer learning，先用低保真训练，再用少量高保真微调。
4. DeepOHeat-v1 引入可信度评分，并在必要时回退到数值求解器做修正。

## 3. 对你最合适的方法定位

我建议你不要把研究问题写成：

`RIGNO 直接替代 FEM 求解包含 TSV/微凸点/互连线的全显式 3D IC 热场。`

这个目标太重，而且不符合现有文献的成熟做法。

更合理的定位是：

`基于局部等效热参数提取与多保真迁移学习的 RIGNO 3D IC 热场预测框架`

具体说，就是把任务拆成两层：

1. `局部微结构层`：解决 TSV、micro-bump、局部互连单元如何变成可学习的等效热参数。
2. `全局热场层`：解决在任意域、多层堆叠、复杂边界下，如何快速预测高保真温度场。

这样以后你的创新点会更稳：

1. 不和 PI-ONet 重复，因为你主模型用的是 RIGNO 而不是 DeepONet。
2. 不和 ARO/T-Fusion 重复，因为你强调的是任意域图神经算子和结构迁移。
3. 不和纯等效参数工作重复，因为你把等效参数进一步送入全局算子学习与多保真修正。

## 4. 最推荐的数据库结构

## 4.1 不建议用“单一大表”

热仿真数据本质上有三种东西：

1. `元数据`：材料、几何、边界条件、求解器设置、保真度等级。
2. `场数据`：坐标、温度场、功率场、热导率场、界面掩码、低保真基线场。
3. `关系数据`：同一个 case 的 coarse/fine/high-fidelity 对应关系，源域/目标域关系，训练/验证/测试划分。

所以最合理的做法不是 MySQL 一张大表，而是：

1. 用 `Parquet` 或 `SQLite` 存元数据和索引。
2. 用 `Zarr`、`HDF5` 或 `npy` 存大体积场数据。
3. 用 manifest 文件把样本关系串起来。

这是最适合机器学习训练的“文件数据库”结构。

## 4.2 三层数据库

### A. 微结构等效库

目标：解决 TSV、micro-bump、局部 BEOL/混合键合单元的等效热参数。

建议对象：

1. TSV 单元。
2. TTSV 单元。
3. micro-bump + underfill 单元。
4. hybrid bonding 局部单元。
5. BEOL 金属/介质复合单元。

每条记录至少包含：

1. `structure_type`：TSV、bump、BEOL、bonding。
2. `geometry_params`：直径、半径、pitch、liner 厚度、高度、填充率、层数、方向。
3. `material_stack`：Cu、Si、SiO2、underfill、bonding layer 等。
4. `solver_level`：FEM coarse / FEM fine。
5. `outputs`：`k_x_eq`、`k_y_eq`、`k_z_eq`、可选 `R_interface_eq`。
6. `valid_range`：适用几何范围。

这里的关键不是把每个单元都喂给 RIGNO，而是先训练一个“小模型”把微结构参数映射到等效参数。

### B. 芯片级多保真样本库

目标：给全局热场学习提供训练样本。

每个 case 描述一整个 3D stack / chiplet 结构，至少要有：

1. `stack_id`：几何与层叠结构编号。
2. `power_id`：功率图编号。
3. `bc_id`：边界条件编号。
4. `fidelity`：low / medium / high。
5. `mesh_or_points_path`：坐标点云或网格。
6. `k_tensor_path`：各向异性导热率场。
7. `q_path`：功率场。
8. `interface_mask_path`：界面掩码或材料分区。
9. `t_low_path`：低保真温度场。
10. `t_high_path`：高保真温度场。
11. `flux_path`：可选热流场。
12. `parent_case_id`：表示它和哪个低保真样本对齐。

### C. 迁移实验索引库

目标：让你的实验设计不是“随机切分”，而是真正有迁移含义。

建议单独存：

1. `source_domain`：规则层叠、简单材料、无显式 TSV 的源域。
2. `target_domain_1`：加入等效 TSV/bump 层。
3. `target_domain_2`：更复杂 chiplet 轮廓和非规则边界。
4. `target_domain_3`：材料体系变化，例如 underfill / bonding 材料变化。
5. `transfer_type`：geometry shift / material shift / resolution shift / fidelity shift。

这样你后面写论文时，迁移实验会非常清楚。

## 5. 一套真正可落地的字段设计

建议在数据库根目录下至少准备以下 manifest。

### 5.1 `materials.parquet`

字段建议：

1. `material_id`
2. `name`
3. `kx`
4. `ky`
5. `kz`
6. `rho`
7. `cp`
8. `interface_model`
9. `source_ref`
10. `temperature_range`

### 5.2 `microstructures.parquet`

字段建议：

1. `micro_id`
2. `type`
3. `geometry_json`
4. `material_json`
5. `fidelity`
6. `solver`
7. `kx_eq`
8. `ky_eq`
9. `kz_eq`
10. `r_interface_eq`
11. `sample_path`

### 5.3 `stacks.parquet`

字段建议：

1. `stack_id`
2. `stack_type`
3. `outline_json`
4. `layers_json`
5. `micro_assignment_json`
6. `cooling_json`
7. `mesh_strategy`
8. `domain_shift_tag`

### 5.4 `cases.parquet`

字段建议：

1. `case_id`
2. `stack_id`
3. `power_id`
4. `bc_id`
5. `fidelity`
6. `solver`
7. `mesh_path`
8. `field_k_path`
9. `field_q_path`
10. `field_interface_path`
11. `field_t_path`
12. `field_flux_path`
13. `parent_case_id`
14. `runtime_sec`
15. `quality_flag`

### 5.5 `splits.parquet`

字段建议：

1. `case_id`
2. `split`
3. `domain_role`
4. `transfer_group`

### 5.6 `pairs.parquet`

这个表对多保真和残差学习最有用。

字段建议：

1. `low_case_id`
2. `high_case_id`
3. `alignment_type`
4. `same_geometry`
5. `same_power`
6. `same_bc`
7. `delta_t_path`

## 6. 样本文件怎么组织

建议目录结构如下：

```text
dataset_3dic_thermal/
  manifests/
    materials.parquet
    microstructures.parquet
    stacks.parquet
    cases.parquet
    splits.parquet
    pairs.parquet
  microstructures/
    tsv/
    micro_bump/
    beol_cell/
    bonding/
  cases/
    case_000001/
      points.npy
      k_tensor.npy
      q.npy
      material_id.npy
      interface_mask.npy
      t_low.npy
      t_high.npy
      delta_t.npy
      meta.json
    case_000002/
      ...
```

如果后面数据量变大，`points.npy`、`k_tensor.npy`、`t_low.npy`、`t_high.npy` 可以逐步迁移到 `zarr`。

## 7. 推荐的建模变量

为了让 RIGNO 真的学到“异构热导率”，建议每个点至少具备以下输入通道。

### 7.1 几何与位置

1. `x, y, z`
2. `layer_index`
3. `distance_to_top_sink`
4. `distance_to_interface`

### 7.2 物理场输入

1. `q(x,y,z)`：功率密度
2. `k_x(x,y,z), k_y(x,y,z), k_z(x,y,z)`：各向异性导热率
3. `material_id` 或其 embedding
4. `interface_mask`
5. `boundary_type_mask`
6. `h_conv`、`T_amb`

### 7.3 低保真辅助输入

1. `T_low`
2. `grad_T_low` 或可选热流近似
3. `fidelity_level`

这里最关键的一点是：

不要只存一个标量 `k`，而要尽量存 `k_x, k_y, k_z`。

因为 TSV、micro-bump、BEOL 复合层最自然的结果就是各向异性，而不是单一均匀热导率。

## 8. 最推荐的学习目标

我建议你把 RIGNO 的主任务定义成残差学习，而不是绝对温度回归。

### 方案 A：直接学习高保真温度场

形式：

`(k, q, bc, geometry) -> T_high`

优点：

1. 形式最干净。

缺点：

1. 对高保真样本量依赖大。
2. 对跨结构泛化压力大。

### 方案 B：学习高低保真修正量

形式：

`(T_low, k_tensor, q, interface, bc, geometry) -> DeltaT`

然后：

`T_pred = T_low + DeltaT`

这是我最推荐的方案，原因有三：

1. 低保真求解器已经学到了大部分平滑热扩散趋势。
2. RIGNO 只需要修正 TSV、bump、界面和热点附近的误差。
3. 最符合多保真文献的工程逻辑。

### 方案 C：局部可信度 + 回退求解

这是后续增强路线，不是第一阶段必做。

形式：

1. RIGNO 先给出 `T_pred`。
2. 再输出一个 `confidence score`。
3. 若局部区域置信度低，则调用高保真求解器增量修正。

这条路线与 DeepOHeat-v1 的混合框架最接近。

## 9. RIGNO 在这里应该怎么用

RIGNO 的优势有三个：

1. 任意域和点云表示。
2. 多尺度区域节点传播。
3. 对分辨率和采样变化更友好。

但它并不直接解决：

1. TSV/bump 微结构等效参数提取。
2. 多保真配对逻辑。
3. 3D IC 的界面物理约束设计。

所以最合理的角色分工是：

1. `微结构等效模型` 负责局部参数提取。
2. `低保真热求解器` 负责提供 baseline。
3. `RIGNO` 负责在复杂几何与任意采样下学习全局热场或修正场。

### 9.1 和你现有代码的衔接

你现在的 `rigno/dataset_Heat3D.py` 与 `rigno/heat3d_pipeline.py` 已经支持“坐标 + 系数场 + 输出场”的基本模式。

建议后续把当前两通道系数：

1. `k`
2. `q`

扩展成多通道：

1. `T_low`
2. `q`
3. `k_x`
4. `k_y`
5. `k_z`
6. `interface_mask`
7. `boundary_mask`
8. `material_embedding_index`

按你当前 pipeline 的结构，一个很实用的改法是：

1. 把 `T_low` 放在 `u_inp`。
2. 把其余物理量放在 `c_inp`。
3. 输出目标设为 `DeltaT` 或 `T_high`。

这样对现有代码改动最小。

## 10. 分阶段实施路线

## 第一阶段：把现有 Heat3D 升级成“异构稳态热基线”

目标不是先做真实 3D IC，而是做正确的数据抽象。

建议新增以下能力：

1. 多层材料，不再只有左右两块材料。
2. `k_x, k_y, k_z` 各向异性导热率。
3. 多种边界条件，而不是统一 Dirichlet。
4. 界面掩码。
5. 一个粗网格或紧凑模型生成的 `T_low`。

这一阶段完成后，你就有了“残差学习版 Heat3D-RIGNO”。

## 第二阶段：建立微结构等效库

建议顺序：

1. 先做 TSV。
2. 再做 micro-bump。
3. 最后做 BEOL/互连复合层。

优先级这样排，是因为：

1. TSV 和 bump 在文献中已有更成熟的等效路线。
2. BEOL 线网最难，适合放在第三步。

### 第二阶段最小可行规模

建议先做：

1. `TSV`：500 到 1000 个高保真单元样本。
2. `micro-bump`：500 到 1000 个高保真单元样本。
3. `BEOL 单元`：200 到 500 个高保真单元样本。

这已经足够训练一个等效参数预测器。

## 第三阶段：建立芯片级多保真样本库

建议保真度定义如下：

1. `Low`：粗网格 FDM/FVM 或 HotSpot/紧凑热模型/等效层模型。
2. `Medium`：较细网格，但仍使用等效 TSV/bump 层。
3. `High`：局部显式 TSV/bump 或更精细 FEM/Icepak/COMSOL 结果。

### 第三阶段最小可行规模

如果算力有限，我建议的 MVP 规模是：

1. `Low-fidelity`：5000 到 10000 个样本。
2. `Medium-fidelity`：500 到 1500 个样本。
3. `High-fidelity`：100 到 300 个样本。

先做稳态，不要一开始就做瞬态。

## 第四阶段：迁移学习与论文实验

建议至少做四组实验：

1. `ID`：同分布测试。
2. `Geometry shift`：层数、轮廓、TSV 布局变化。
3. `Material shift`：underfill、bonding、填充材料变化。
4. `Fidelity shift`：仅用少量高保真样本做微调。

比较对象建议包括：

1. 直接从头训练。
2. 低保真预训练 + 高保真微调。
3. 直接预测 `T_high`。
4. 预测 `DeltaT`。

## 11. 关于 TSV、微凸点和互连线，数据库里究竟怎么表示

这是你问题里最关键的一点。

### 11.1 TSV

建议默认不显式表示整片芯片里的每根 TSV。

建议做法：

1. 在微结构层面对 TSV 单元做显式高保真提取。
2. 在芯片级层面把 TSV 阵列映射成 `k_x, k_y, k_z` 与 `R_interface`。
3. 仅在高保真验证子集中保留局部显式 TSV 结构。

### 11.2 micro-bump

与 TSV 类似处理，但要额外注意：

1. bump 高度和直径带来的纵向与横向热阻差异。
2. underfill 对热扩散路径的强影响。
3. bump 阵列稀疏度对等效导热率的影响。

### 11.3 互连线 / BEOL

这部分最不适合直接全显式建模。

建议做法：

1. 用局部代表单元建立金属填充率、线宽、间距、方向到等效张量的映射。
2. 对不同金属层方向性，显式记录 `k_x != k_y`。
3. 若接触热阻明显，再增加一个 `R_interface_eq` 或“薄层热阻”特征。

一句话总结：

`TSV/bump/互连线最好在数据库中以“局部显式 + 全局等效”的混合方式存在。`

## 12. 我最建议你现在就开始做的版本

如果你希望尽快形成论文可跑通的路线，我建议立刻采用下面这个版本。

### 版本名

`RIGNO-MF-3DIC-v0`

### 目标

学习：

`(T_low, q, k_x, k_y, k_z, interface_mask, bc_features, points) -> DeltaT`

### 数据来源

1. `Low`：等效层 + 粗网格热求解器。
2. `High`：少量 FEM/Icepak/COMSOL 芯片级结果。
3. `Micro`：TSV/bump/BEOL 单元高保真提取结果。

### 训练顺序

1. 在简单异构立方体和多层结构上预训练。
2. 加入等效 TSV/bump 层的芯片级样本继续训练。
3. 用少量高保真 3D IC 目标样本微调。
4. 最后评估几何迁移、材料迁移和少样本高保真适配。

### 你会得到的优势

1. 能解释 TSV/bump/互连引起的热导率非均匀性是如何进入模型的。
2. 能说明为什么 RIGNO 适合复杂几何与任意采样。
3. 能借多保真减少高保真数据压力。
4. 能和当前文献清楚区分开来。

## 13. 不建议你现在做的事情

1. 一开始就做瞬态、多物理场、热-应力耦合、布局优化闭环全套。
2. 一开始就追求显式 BEOL 全芯片级解析。
3. 只存温度场，不存材料场、界面和保真关系。
4. 只做随机 train/test 划分，不做迁移划分。

## 14. 最后一句判断

如果把你当前基础、已有文献和 RIGNO 的特性放在一起看，最可行的方案不是“直接把 RIGNO 套到真实 3D IC 全显式热仿真”，而是：

`先建立微结构等效参数库，再建立芯片级多保真数据库，最后让 RIGNO 学习低保真到高保真的热场修正。`

这条路线最稳、最符合现有文献演化，也最容易在你当前代码基础上做出第一版结果。

## 15. 对应资料来源

你当前工作区里最值得和本方案一起看的材料有：

1. `报告/异构热导率导线互联结构的热仿真算子学习专题报告.html`
2. `报告/3DIC热仿真AI文献综述_增强版.html`
3. `报告/3DIC文献分类与创新点分析.html`
4. `报告/3DIC热仿真选题调研报告.html`

关键论文建议优先复读：

1. RIGNO: <https://arxiv.org/html/2501.19205v2>
2. DeepOHeat: 3D IC 热方程配置到温度场的算子学习。
3. ARO: 多保真、主动学习与可迁移 3D IC 热分析。
4. PI-ONet: TSV/bump 等效热导率提取 + 界面物理约束。
5. Fast Thermal Modeling of TTSV via Bounded Neural Networks: TTSV 各向异性等效热导率学习。
6. SAU-FNO: 低保真预训练 + 高保真微调的迁移路径。
7. DeepOHeat-v1: 可信度评分与混合求解修正。
