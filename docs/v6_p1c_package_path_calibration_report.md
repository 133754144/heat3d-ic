# V6-P1c package-path calibration

## 范围与冻结合同

P1c 只新增 B/C0 共 8 个样本；P1b-A 没有重新生成或写入。为补齐 A 的层温降和 junction 热阻，生成器对 4 个冻结 A 工况做了只读、内存内 solver replay，并逐项核对既有 peak/mean DeltaT、Rth 和上下热流占比。没有扩样、训练、模型推理、按温升筛选/重采、功率反算或材料参数调节。

共同条件为 10 x 10 mm P1b `logic_package`、top Robin `h=500 W/(m2 K), T_inf=300 K`、sides adiabatic、perfect contact，以及 P1b 的单层单热点与双层分布热源。每个 topology 仍有 64 mm2 总 source area，1/4 W 仍按 source area 分配。source 的 z-node 离散严格保留 P1b 行为，避免 package path 之外的 q 离散变化。

- A（只读基线）：300 K 直接施加在 lower die 外底面。
- B：lower die 下依次加入 75 um bump/underfill、100 um TSV silicon interposer、1 mm via-bearing BT substrate、1.6 mm FR-4 PCB；300 K 只施加在 PCB 外底面。
- C0：不加底部层，lower die 外底面改为 adiabatic。

B 的各向异性等效参数直接冻结自封装热模型表：bump/underfill `kxy/kz=0.6/4.9`、interposer `148.3/151.0`、BT substrate `0.2/0.49`、PCB `0.8/0.3 W/(m K)`。该来源明确将 bump/underfill 和布线封装部件表示为保持厚度与热路径的等效块；P1c 没有根据求解温度改变这些值。Quasi-3D package 文献也把 package stack 显式解析到 PCB 外表面后，才施加恒温或对流边界：

- [Modeling, Analysis, Design, and Tests for Electronics Packaging beyond Moore, chapter 3](https://www.sciencedirect.com/science/article/pii/B9780081025321000032), Tables 3.3.1--3.3.2.
- [Quasi-3D Thermal Simulation of Integrated Circuit Systems in Packages](https://doi.org/10.3390/en13123054).

## A/B/C0 成对结果

| topology | P (W) | path | peak DeltaT (K) | mean DeltaT (K) | Rth_peak (K/W) | top/bottom fraction | R_j-top (K/W) | R_j-board (K/W) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| single hotspot | 1 | A | 0.0098 | 0.0060 | 0.0098 | 0.0003/0.9997 | 0.0013 | 0.0075 |
| single hotspot | 1 | B | 16.0174 | 11.4669 | 16.0174 | 0.7780/0.2220 | 0.3580 | 15.9184 |
| single hotspot | 1 | C0 | 20.5713 | 20.0766 | 20.5713 | 1.0000/0 | 0.4616 | n/a |
| single hotspot | 4 | A | 0.0390 | 0.0239 | 0.0098 | 0.0003/0.9997 | 0.0013 | 0.0075 |
| single hotspot | 4 | B | 64.0694 | 45.8676 | 16.0174 | 0.7780/0.2220 | 0.3580 | 15.9184 |
| single hotspot | 4 | C0 | 82.2851 | 80.3065 | 20.5713 | 1.0000/0 | 0.4616 | n/a |
| dual-layer distributed | 1 | A | 0.1884 | 0.0785 | 0.1884 | 0.0044/0.9956 | 0.0005 | 0.0882 |
| dual-layer distributed | 1 | B | 16.1320 | 11.4443 | 16.1320 | 0.7790/0.2210 | 0.3994 | 15.9776 |
| dual-layer distributed | 1 | C0 | 20.7304 | 20.0671 | 20.7304 | 1.0000/0 | 0.5105 | n/a |
| dual-layer distributed | 4 | A | 0.7538 | 0.3141 | 0.1884 | 0.0044/0.9956 | 0.0005 | 0.0882 |
| dual-layer distributed | 4 | B | 64.5280 | 45.7773 | 16.1320 | 0.7790/0.2210 | 0.3994 | 15.9776 |
| dual-layer distributed | 4 | C0 | 82.9215 | 80.2683 | 20.7304 | 1.0000/0 | 0.5105 | n/a |

定义：junction temperature 是 source-power-weighted solver temperature；`R_j-top=(T_j-T_top_surface_mean)/P_total`，`R_j-board=(T_j-T_board_exterior_mean)/P_total`。C0 没有 board 和底部热流，因而 `R_j-board` 明确为 not applicable，而不是无穷大或零。

30--80 K 仍只是求解后的结果窗口。8 个新样本中只有 B 的两个 4 W 工况自然命中（64.07/64.53 K）；B 的 1 W 为 16.02/16.13 K，C0 的 1 W 为 20.57/20.73 K，C0 的 4 W 为 82.29/82.92 K。没有因这些结果修改或替换任何工况。

## 层与界面温降

逐样本完整值位于 `configs/heat3d_v6/v6_p1c_package_path_layer_drops.csv` 和 `v6_p1c_package_path_interface_drops.csv`。4 W 工况的跨层绝对温降最大值为：

| path | layer | max absolute axial drop (K) |
|---|---|---:|
| B | PCB FR-4 equivalent | 46.1934 |
| B | BT substrate with vias | 16.9819 |
| B | silicon interposer | 0.0272 |
| B | bump/underfill | 0.1168 |
| B | lower/upper silicon (max) | 0.1769 |
| B | TIM (max) | 0.3353 |
| C0 | lower/upper silicon (max) | 0.2273 |
| C0 | TIM (max) | 0.4309 |

由于本轮冻结 `perfect contact` 且采用 conformal shared nodes，数学上的界面温度 jump 全部严格为 0。interface CSV 同时保存界面两侧相邻 z-plane 的温差；它描述界面邻域梯度，不应误称为 contact-resistance jump。B 的主要底部路径温降来自 PCB 和 BT substrate，而不是 bump 或 silicon interposer；这正是将 300 K 从 die 底面移至板外底面后，A 的零外部底阻被显式替换的结果。

## 网格、投影与守恒

- A/C0 mesh：64 x 64 x 32 intervals，139425 nodes；B mesh：64 x 64 x 56 intervals，240825 nodes。所有新增层至少 4 个 z intervals，source 的最小 x/y 解析度和 control-volume 覆盖仍满足 P1b 下限。
- 每个新样本在温度求解前冻结 1024 个唯一 irregular points：volume/source/interface/top/bottom = 512/256/128/64/64，选择过程不读标签。
- solver peak 与 projected-point peak 的最大差为 0.01347 K。还保存了 solver CV mean 与 1024 点非加权 mean 的差；由于采样刻意过采 source/interface，该差是 sampling representativeness bias，而不是线性插值数值误差，也不参与筛选。
- 最大绝对能量守恒相对误差为 `3.45e-9`；全部必需值 finite。B 的 top+bottom fraction 与 C0 的 top-only fraction 均在该容差内闭合。

## 物理判断

**B remote Dirichlet 能形成合理的顶部主导双路径校准。** 四个 B 样本顶部承担 77.80%--77.90% 热流、底部仍承担 22.10%--22.20%；300 K 不再直接钳制 die，而位于显式 PCB 外底面。该路径既避免 P1b-A 的近端恒温短路，又保留封装到板的物理下行支路，适合作为下一步主冷却候选的校准证据。

**C0 能形成严格 top-only 极限，但不应被称为完整的 package cooling path。** 它在数值上守恒且全部热量经 top Robin 离开，适合 verification/OOD 或敏感性上界；但它删除了所有板端散热，`R_j-board` 无定义。因此 C0 可以证明当前 top Robin 在 1--4 W 下能够承担热量，却不能替代带 PCB/安装条件的生产边界。

## 工件与复现

- 数据目录：`data/heat3d_v6_p1c_package_path_calibration8_v0`（仓库 ignore；只含 8 个新样本）。
- manifest SHA256：`f24b9f320a883348e202c6dd37a20f1e683022de2989dbcfdb2361d49786b6a7`。
- 生成：`python3 scripts/generate_heat3d_v6_p1c_package_path_calibration.py`
- 校验：`python3 scripts/check_heat3d_v6_p1c_package_path_calibration.py`

本报告不建议扩样，也没有据此冻结正式 V6 主体数据分布；P1c 只完成 package-path calibration。
