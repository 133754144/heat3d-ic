# V6-P1b bottom Dirichlet 与界面热阻文献审计

## 核心判断

Dirichlet boundary 只定义施加边界上的温度，不自动包含 heat-sink、TIM、substrate 或 thermal contact resistance。两种模型必须严格区分：

1. **远端 Dirichlet**：固定温度施加在 PCB/cold plate/heat-sink 的外表面，die 与该边界之间仍显式保留 package/TIM/material layers；这些层的 `t/k` 热阻继续存在。
2. **die-bottom Dirichlet**：固定温度直接施加在 silicon die 外表面；若没有独立 film/contact law，则 die 到理想热浴的边界热阻为零。

P1b 属于第二种。内部有两个 50 um、k=4 W/(m K) 的 TIM material layers，但 lower silicon 的 bottom face 直接固定为 300 K，且所有材料界面采用 perfect contact。因此 P1b 只包含 TIM 层自身的有限导热热阻，不包含 Si/TIM、TIM/Cu 的零厚度 contact/TBR，也不包含 lower-die 到冷板/PCB 的底部 package/interface resistance。

## 一手论文对照

| source | 外部热边界 | package/interface 表示 | 对 P1b 的含义 |
|---|---|---|---|
| [DeepOHeat-v1](https://arxiv.org/abs/2504.03955) | 原始 DeepOHeat 系列使用 top/bottom convection，而非 die-bottom fixed temperature | 三层算例用分层 k；论文未给出独立 contact/TBR 参数 | 不能用它为 die-bottom 300 K + perfect contact 背书 |
| [HotSpot 4.0 accuracy study](https://www.cs.virginia.edu/~skadron/Papers/hotspot_tc08.pdf) | 明确指出更真实的外表面是分布式 convection，而单一 isothermal package node 是近似 | 显式加入低 k TIM layer，并比较 k=7.5 与 1.33 W/(m K) | TIM material resistance 必须保留；它仍不等同独立 contact resistance |
| [Quasi-3D package simulation](https://www.mdpi.com/1996-1073/13/12/3054) | PCB surface 可固定为 ambient，也可使用 convection | die 到 PCB 之间显式包含 bumps、substrate、interposer、microbumps 等结构层，层间垂向项由 layer k/thickness 给出；未单列 contact TBR | 文献中的 constant PCB temperature 是远端 Dirichlet，不等价于直接固定 die bottom |
| [3D-ICE](https://www.epfl.ch/labs/esl/wp-content/uploads/2018/12/3D-ICE_ICCAD2010.pdf) | 论文中的 Dirichlet 用于 microchannel coolant inlet；外露 stack surfaces 在该液冷模型中设为 adiabatic | stack 由 material layers 和冷却单元组成 | “3D-ICE 使用 Dirichlet”不能推导出“die bottom 应固定为 ambient” |
| [Fast convolution 3D-IC model](https://www.sciencedirect.com/science/article/pii/S0026269214001463) | 为速度可用 BC 隐式替代真实 package，但论文专门讨论该简化的 package impact | silicon dies 之间保留 low-conductivity interface layers | 即使 package 被边界条件等效，inter-tier material resistance 仍需显式保留 |
| [HBM anisotropic conductivity measurement](https://arxiv.org/abs/2303.06785) | 测量目标不是 bottom-Dirichlet package 仿真 | polymer/solder interfaces 使器件 through-plane effective k 显著低于单体材料 | 界面层不能由 perfect-contact silicon stack 静默替代；但 effective k 也不能冒充独立 contact TBR |
| [Irregular 3D-IC heat-source optimization](https://thermalscience.rs/pdfs/papers-2026/TSCI251130051W.pdf) | top surface 设为 298.15 K isothermal，其余绝热 | 论文明确声明各层 perfect contact，并明确忽略 adjacent-layer contact resistance | 这是“恒温边界 + 忽略 contact”的透明简化示例，不是界面热阻已被 Dirichlet 吸收 |

这些论文呈现的共同模式是：固定温度边界与内部热阻是两个独立建模选择。若论文忽略 contact resistance，通常应像最后一个案例一样显式声明；若希望保留界面影响，则以 TIM/underfill/bumps 等有限厚度材料层或独立 interface law 表示。

## P1b 数量级审计

- 每个内部 TIM layer 的 areal material resistance 为 `t/k = 50e-6 / 4 = 1.25e-5 m2 K/W`。这是材料层热阻，不是 contact TBR。
- top Robin 的 areal convection resistance 为 `1/h = 2.0e-3 m2 K/W`，但 P1b 的 bottom face 是零边界热阻的恒温面，所以 99.5602%--99.9693% 功率从 bottom 流出。
- 单层 topology 的 Rth_peak 仅 0.00976--0.01013 K/W；双层 topology 为 0.18844--0.19345 K/W。双层增量说明内部 TIM/upper-die 路径确实生效，但它不能补偿 lower-die 与 300 K 理想热浴的直接连接。

因此，P1b 的 1--4 W 结果首先是 **ideal-bottom-sink sensitivity result**，不是正式 V6 package-power envelope 的证据。

## 评估意见

1. P1b 的 metadata/status 应保持 `bottom_dirichlet_perfect_contact_verification` 语义；不能标为正式主体 cooling condition。
2. “1--2 W 不适合作为功率中心”只能限定为 **当前 P1b BC 下**。在冻结真实 package/cooling path 前，不应把该结论推广到正式 V6。
3. 4 W 应保留在已生成 P1b 中，并优先作为后续同几何 BC/interface paired sensitivity 的高端锚点；在敏感性结果出来前，不把它纳入或排除正式主体功率档。
4. 后续若获授权，应固定功率和 source topology，只改变热边界路径：
   - 当前 die-bottom Dirichlet + perfect contact；
   - 远端 bottom temperature，显式加入具名 substrate/TIM/package layers；
   - 在已引用材料界面上再单独加入 contact/TBR。
5. contact/TBR 必须保存为 `m2 K/W` 的独立 face parameter，并绑定 `interface_id/lower_material/upper_material/evidence_id`。不得通过修改 bulk/effective k 静默代替，也不得从本轮温升反算 contact 数值。

本轮只完成文献与现有 P1b 结果审计，没有新增样本或运行新的 BC counterfactual。
