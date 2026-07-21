# V6-P1b logic_package 16-sample power calibration

## 冻结范围与结论

P1b 已按 `4 source topologies x 4 package powers` 生成固定 16 样本。没有训练、模型推理、扩样、样本替换、按温升筛选/重采或由温度/Rth 反算功率。30--80 K 只在求解完成后统计，命中数为 **0/16**。

在本轮 10 x 10 mm `logic_package`、top Robin 500 W/(m2 K)、bottom Dirichlet 300 K 合同下：

- 1 W 的 peak DeltaT 为 0.0098--0.1934 K；2 W 为 0.0195--0.3869 K，**在当前 P1b BC 下，1--2 W 不适合作为 30--80 K 数据的功率中心**。该结论不能在正式 cooling/package path 冻结前外推。
- 4 W 的 peak DeltaT 仅为 0.0390--0.7738 K，仍比结果窗口低两个数量级。它应保留为后续同几何 BC/interface paired sensitivity 的高端锚点，但在敏感性结果完成前既不晋级、也不永久排除于正式主体功率档。
- bottom Dirichlet 承担 99.5602%--99.9693% 总热流，是低热阻和低温升的主导条件。内部 TIM layer 的有限 `t/k` 热阻仍存在，但 bottom 300 K 直接施加在 lower silicon 外表面，外部底部 interface/package path 为零热阻。详见 `docs/v6_p1b_bottom_dirichlet_interface_resistance_audit.md`。正式 V6 主体应先冻结主冷却和界面合同；本轮禁止且没有反算所需功率或 contact resistance。

## logic_package 与面积功率合同

P1b 使用正式 V6 `logic_package` 材料族和 10 x 10 mm footprint。为了在一个固定 stack 内比较一层/两层 source topology，silicon-die/TIM 单元实例化为 lower die、inter-die TIM、upper die、top TIM，顶部为 copper spreader：

| layer, bottom to top | thickness | k | property role | z intervals |
|---|---:|---:|---|---:|
| silicon_die_lower | 0.15 mm | 120 W/(m K) | bulk, active | 4 |
| tim_between_dies | 0.05 mm | 4 W/(m K) | effective, interface | 4 |
| silicon_die_upper | 0.15 mm | 120 W/(m K) | bulk, active | 4 |
| tim_to_spreader | 0.05 mm | 4 W/(m K) | effective, interface | 4 |
| spreader | 1.00 mm | 400 W/(m K) | bulk, passive | 16 |

四种 topology 的 declared source planform area 都固定为 64 mm2：

| topology | active layers | sources | per-source area | per-source power fraction |
|---|---:|---:|---:|---:|
| single_layer_single_source | 1 | 1 | 64 mm2 | 1 |
| single_layer_multi_source | 1 | 4 | 16 mm2 | 1/4 |
| dual_layer_few_source | 2 | 2 | 32 mm2 | 1/2 |
| dual_layer_multi_source | 2 | 8 | 8 mm2 | 1/8 |

对每个样本，`source_power = package_total_power x source_area / 64 mm2`。所以 0.5/1/2/4 W 分别对应统一面功率密度 7.8125/15.625/31.25/62.5 kW/m2；单源功率范围 0.0625--4 W。体积 q 再按各 source 实际覆盖的 control-volume volume 归一化，范围 5.616e7--5.825e8 W/m3。package、active-layer 与 component/source power 分层保存，没有把 die/package power 无条件复制到每个局部源。

## 逐样本热结果

| topology | P (W) | peak DeltaT (K) | mean DeltaT (K) | Rth_peak (K/W) | top/bottom heat fraction |
|---|---:|---:|---:|---:|---:|
| single-layer single | 0.5 | 0.004881 | 0.002984 | 0.009762 | 0.000307/0.999693 |
| single-layer single | 1 | 0.009762 | 0.005968 | 0.009762 | 0.000307/0.999693 |
| single-layer single | 2 | 0.019524 | 0.011935 | 0.009762 | 0.000307/0.999693 |
| single-layer single | 4 | 0.039049 | 0.023871 | 0.009762 | 0.000307/0.999693 |
| single-layer multi | 0.5 | 0.005065 | 0.002984 | 0.010130 | 0.000307/0.999693 |
| single-layer multi | 1 | 0.010130 | 0.005968 | 0.010130 | 0.000307/0.999693 |
| single-layer multi | 2 | 0.020260 | 0.011935 | 0.010130 | 0.000307/0.999693 |
| single-layer multi | 4 | 0.040519 | 0.023871 | 0.010130 | 0.000307/0.999693 |
| dual-layer few | 0.5 | 0.096724 | 0.039261 | 0.193447 | 0.004398/0.995602 |
| dual-layer few | 1 | 0.193447 | 0.078521 | 0.193447 | 0.004398/0.995602 |
| dual-layer few | 2 | 0.386895 | 0.157043 | 0.193447 | 0.004398/0.995602 |
| dual-layer few | 4 | 0.773789 | 0.314086 | 0.193447 | 0.004398/0.995602 |
| dual-layer multi | 0.5 | 0.094221 | 0.039261 | 0.188442 | 0.004398/0.995602 |
| dual-layer multi | 1 | 0.188442 | 0.078521 | 0.188442 | 0.004398/0.995602 |
| dual-layer multi | 2 | 0.376884 | 0.157043 | 0.188442 | 0.004398/0.995602 |
| dual-layer multi | 4 | 0.753768 | 0.314086 | 0.188442 | 0.004398/0.995602 |

所有 topology 均呈线性功率响应；同一 topology 的 Rth_peak 在四个功率档内只呈数值舍入级变化。单层 single/multi 的 mean DeltaT 相同，fragmentation 只令 peak 改变约 3.8%。双层工况由于一半功率位于 upper die，Rth_peak 约为单层的 18.6--19.8 倍，但仍不足以让 4 W 接近 30 K。

## 网格、投影与守恒

- layer-aligned mesh：64 x 64 x 32 intervals，65 x 65 x 33 = 139425 nodes；每个物理层至少 4 个 z intervals。
- 60 个 source 的覆盖范围为 975--7803 control volumes；最差 source 仍有 12 个 x/y intervals，超过预注册下限 8。
- bottom Dirichlet nodes 上 q 的数量为 0。
- 每样本在求解前冻结 1024 个唯一 irregular points：volume/source/interface/top/bottom = 512/256/128/64/64；点选择不读取温度、梯度、热点或 solver residual。
- solver peak 与 1024 点投影 peak 的差值最大为 0.003543 K。该值只作当前点覆盖诊断，不参与样本接受、替换或重采。
- 最大绝对能量守恒相对误差为 `1.19091e-10`；全部数组和审计指标 finite。

## 工件与复现

- Dataset manifest SHA256：`74e5799f5104c729f9bf4c35d6f5979d95c18d5b6efd7328f1691c84dd3517cd`。
- 数据根目录：`data/heat3d_v6_p1b_logic_package_power_calibration16_v0`（129 files，约 2.3 MiB）。`data/` 遵循仓库既有 ignore 策略；registry、manifest mirror、逐样本/逐热源 CSV、JSON audit、generator、checker 和本报告进入 Git。
- 生成：`python3 scripts/generate_heat3d_v6_p1b_logic_package_power_calibration.py`
- 校验：`python3 scripts/check_heat3d_v6_p1b_logic_package_power_calibration.py`

生成器拒绝覆盖已有 P1b 数据目录。本轮结论只适用于冻结的 bottom-Dirichlet calibration 合同，不把该边界的低温结果外推为正式 V6 主体数据分布。
