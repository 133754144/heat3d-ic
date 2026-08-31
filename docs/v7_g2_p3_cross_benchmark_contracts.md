# Heat3D-on-semiconductor-native dataset contracts

转换器为 `scripts/convert_v7_g2_semiconductor_case.py`。它只重表达官方 case，不求解 PDE、不训练模型。官方绝对温度保持 `T_K=293.15+25u`。由于 released top/bottom Robin ambient 均为 `u=0.2`，Heat3D 使用 `T_ref=298.15 K` 与 `deltaT=25(u-0.2)`。为让 Heat3D 的 Kelvin-form PDE channels 量纲自洽，冻结精确的 canonical dimensionalization：一个 released length unit 记为 1 m、k 数值不变、`q_K=25q_u`、`h=k/Robin_length`。这是对同一 PDE 的固定单位变换，不增加 case 信息，也不表示 benchmark 的真实芯片尺寸。

| benchmark | 原作者输入 | Heat3D 可构造输入 | representation | released split / output | metric alignment | direct numerical compare |
|---|---|---|---|---|---|---|
| DeepOHeat `2d_power_map` | 21×21 top Neumann-flux sensors；固定 cuboid/k/bottom Robin | coords、固定 k、BC，加独立 top-flux sensors | **当前 11-channel 不无损**：surface flux 不能冒充 volumetric q | release prototype maps；query 21×21×11=4851；无 supervised train labels | absolute-T MAPE/PAPE；同 query/Celsius reference 才可对齐 | 否；先需显式 deterministic flux-BC schema 与 train references |
| DeepOHeat `single_htc_bc` | scalar top Robin coefficient；固定 k/q/other BC | 51³ coords + k=.2 + middle-layer q + derived top/bottom h | **lossless** with respect to released nondimensional PDE | continuous parameter family，无 fixed supervised split；output 51³ | 同一 u/ΔT field 的 RMSE、rel-L2、absolute-T MAPE/PAPE | 条件可；Heat3D train labels 需另由同一 solver/fidelity生成并冻结 |
| DeepOHeat `multi_htc_bc` | top+bottom Robin coefficients；其余固定 | 同上，两个 h 分别由 `h=kappa/robin_k` 确定 | **lossless** | continuous 2-D parameter family；output 51³ | 同 single HTC | 条件可；先冻结 common cases/solver labels |
| DeepOHeat-v1 surface | 10,000 train/10 test 的 21² surface-flux functions；10 reference fields | query coords、固定 k/Robin，加独立 top-flux sensors | **当前 11-channel 不无损**，原因同 2d power map | train `(10000,21,21)`；test `(10,21,21)`；reference `(10,101,101,51)` | rel-L2/RMSE/max-L1 on u；MAPE/PAPE on `293.15+25u` | 否；且 release 无 train temperature labels |
| DeepOHeat-v1 volumetric | 100,000 train/100 test 的 101² power functions；100 reference fields | 101×101×56 coords + released discrete k/q coefficients + Robin BC | **lossless with respect to released discrete residual** | train `(100000,101,101)`；test `(100,101,101)`；reference `(100,101,101,56)` | 同 v1 surface | 条件可；test 已有 truth，Heat3D supervised train truth 尚缺 |

## Lossless 的边界

`single_htc_bc`、`multi_htc_bc` 和 DeepOHeat-v1 volumetric 可以不增加物理信息地进入现有 `coords+k+q+BC` schema。前两者的 branch 量是 Robin length/coefficient，转换成 Heat3D h 使用同一个 released 方程的代数恒等式，不是 learned adapter。

两个 surface-power benchmark 的输入是 Neumann surface flux。当前 Heat3D 11 个物理 channels 只有 volumetric q 和 Robin/region flags；把 flux 填进 q 会改变 source measure 与 PDE，故 converter 只输出 `top_neumann_flux_sensors` 作为 schema evidence，不把它喂给现有模型。未来若增加显式非学习型 flux BC value/type，必须先作为新的 cross-benchmark input contract 单独资格验证，不能回写 P1i common-task budget。

## Data sufficiency 与公平性

DeepOHeat/DeepOHeat-v1 是 physics-informed from-scratch：train arrays 是输入函数，不需要 train temperature labels。Heat3D 是 supervised；若要在同一 native case 训练，需要对 frozen train functions 生成同 fidelity 的 reference temperature。这个 solver-label cost 必须单列，不能把“相同 physical cases”写成“相同 training semantics”。

Therm-FM HS_SC 的官方 input 是 `(N,4,2,87,87)`（x/y/z-or-layer/power density），output 是 `(N,2,87,87)` temperature。未来只有在 benchmark data 与材料/BC metadata 可单独取得时，才能把同一 grid cells 确定性转成 Heat3D point cloud；固定 k/BC 可以显式物化，但不能从缺失 metadata 猜测。本轮没有下载 24.10 GB checkpoint 或 4.47 GB steady archive。
