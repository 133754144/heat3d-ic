# V6-RandomBlock 文献合同与分阶段生成计划

## 结论

RandomBlock 保持 V6-layer 的 10×10 mm package、九层结构、64×64×56
layer-aligned FVM 网格、dual-Robin schema、300 K ambient、side adiabatic
和 perfect contact。只改变输入场的局部结构：在两个 active silicon
layers 中加入多个轴对齐 `k`/`q` 块，并把已经有一手论文支撑的冷却和
功率包络扩展到 RandomBlock。

机器可读文献合同位于
`configs/heat3d_v6_randomblock/v6_randomblock_literature.json`。正式
数值范围只取自同行评议一手论文；预印本仅列为背景。论文未报告的值
不得进入正式生成，在直接锚点间插值或把面热流换算成体积热源时必须
携带 `RB-X*` 工程转换 provenance。

## 一手证据矩阵

| ID | 一手论文与定位 | 可直接采用的证据 | 正式用途 |
|---|---|---|---|
| RB-L01 | Petrosyants & Ryabov, *Energies* 2020, Sec. 4.1/Table 1, [doi](https://doi.org/10.3390/en13123054) | 2/20 W package；Si 150、Cu 385 W/(m·K) | power 与 k 锚点 |
| RB-L02 | Redmond et al., *IEEE TCPMT* 2013, PDF pp.1–2/Fig.1/Table I, [doi](https://doi.org/10.1109/TCPMT.2012.2226721) | 14.5/1000 W/cm²；23.9 W/chip、47.8 W/package；h=2050；chip 140、Cu 400 | 高功率/热点/强冷却 |
| RB-L03 | Wang et al., *Thermal Science* 2018, pp.2–3/Table 1/Fig.4, [doi](https://doi.org/10.2298/TSCI1804685W) | q=2e9/6e9 W/m³；总功率 8 W；h=1000；k=20/120/600 | 体积 q 与低/中/高 k |
| RB-L04 | Zhang et al., *Scientific Reports* 2018, 3D-IC case/Table 1, [doi](https://doi.org/10.1038/s41598-018-21020-w) | 1 W；h=15；TIM/chip/underfill/TSV k=10/135/50/400 | 低功率、自然对流与 k |
| RB-L05 | Liu et al., DAC 2023, Sec. III/V-A/V-B, [doi](https://doi.org/10.1109/DAC56929.2023.10247998) | 任意 heat-block/inhomogeneous-k 配置；top/bottom h=333.33–1000 | 随机块和 dual-Robin 设计依据 |
| RB-L06 | Bouden et al., *ICHMT* 2016, Eq.3, [doi](https://doi.org/10.1016/j.icheatmasstransfer.2016.09.016) | QFN/board 不同表面具有不同自然对流系数 | 只支持独立 top/bottom h，不把局部相关式当 uniform h |
| RB-L07 | Alexandrov et al., *IEEE TVLSI* 2014, package thermal setup, [doi](https://doi.org/10.1109/TVLSI.2013.2278951) | h=2500 W/(m²·K) | 保留 V6-layer 强冷却端点 |
| RB-L08 | Patil & Suma, *Heliyon* 2022, FEM/Table 1/TCN cases, [doi](https://doi.org/10.1016/j.heliyon.2022.e08719) | 局部高 k thermal collection network/TTSV | 高 k block 的物理动机 |

RandomBlock 的正式包络为：

- package total power：1–47.8 W；直接锚点为
  1/2/8/20/23.9/47.8 W；
- source surface density：14.5–1000 W/cm²；
- 直接 volumetric q：2e9–6e9 W/m³；1000 W/cm² 按冻结的
  0.15 mm active-layer 厚度换算后的上限为 6.6667e10 W/m³，明确标为
  `RB-X01`，不是论文直接 q；
- `k` block palette：20/50/120/135/140/150/385/400/600 W/(m·K)；
- top h：15–2500 W/(m²·K)；bottom h：15–1000 W/(m²·K)；
- ambient 固定 300 K。

## RandomBlock 几何合同

- 每个 layout group 含 3–10 个 q blocks、2–8 个 k blocks；
- 块均为轴对齐长方体，z 跨越一个完整 active layer；
- q block 的 x/y bbox 必须完全位于 `[0.05L, 0.95L]²`，不得接触
  package 外边界 5% 区域；
- 同类块在同一 layer 不重叠；k/q 跨类重叠只允许
  `disjoint/partial/aligned` 三种预登记关系；
- q block 至少覆盖 3×3 in-plane intervals 和 32 个 control volumes；
- k block 至少覆盖 4×4 in-plane intervals；
- scalar k 写入三条对角分量，所有值必须严格为正；base stack 的
  anisotropic k 不变；
- q 由每个 block 的功率和真实 control-volume 总量计算，严格检查
  `sum(q_i*CV_i)=package power`；
- 每组八个物理变体共享 block 数量、位置、尺寸和 overlap topology，
  只改变 k 数值、q 功率分数、package power 和 top/bottom h。

## Support 与防泄漏

每个 layout group 在任何温度求解前冻结一套 1024 solver-node support，
八个变体共享同一坐标顺序和 graph。选点只依赖 layer/interface、
top/bottom 以及该组预登记的 k/q block bboxes，不读取温度或 split
labels；配额为 volume 512、block coverage 256、interfaces 128、top 64、
bottom 64。checker 要求每个块至少有一个 support node、九层/八界面和
两个 Robin surface 均覆盖。

## 分组、split 与哈希

- 固定 seeds：layout `20260730`、physics `20260731`、split `20260732`、
  support `20260733`；
- 正式集：128 layout groups × 8 variants = 1024；
- group-level split：96 train groups、16 valid groups、16 sealed test
  groups，对应 768/128/128 samples；
- split 只由 `SHA256(split_seed, group_id)` 排序后冻结，同组不得跨 split；
- 输出 layout、physics、support、split、config、manifest 和 provenance
  SHA256。

## 温升合同

预定义 bins 为 `[30,60) / [60,90) / [90,120) / [120,150] K`，每组八个
variant 预登记为每 bin 两个 intended slots。smoke/pilot 先用论文直接
power anchors 与一组全局 BC/power 表生成，保留所有求解结果。允许在
pilot 失败后只修改**全局生成规则**并从头重建 pilot；禁止删除、替换或
按单样本温度挑选。pilot 通过后冻结同一全局规则并一次生成 1024；正式
集不得在生成后按温度或模型误差筛样本。

## 串行 gate

1. `smoke16`：2 groups × 8 variants。通过 schema、finite solve、正定 k、
   power/energy、block/support coverage 和温升范围审计后 commit/push。
2. `pilot128`：16 groups × 8 variants。审计四 bin、k/q/BC joint
   distribution、solver iteration/stability 和 split/group leakage。失败时
   只修改规则、清空独立 pilot 输出并从头开始，历史审计保留。
3. `formal1024`：协议冻结后生成 128 groups。不得在生成后按模型误差、
   test 标签或 temperature 结果重采；输出 manifest、逐样本/逐块 CSV、
   QC JSON/MD 和可复现命令。

三个阶段均为数据生成，不训练、不做模型推理，也不修改 canonical P1h。

## 可复现命令

每个阶段都先冻结协议，再生成独立数据目录，最后运行只读 checker。远端
非交互 shell 必须先加载 conda，再激活 `rigno`。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
python scripts/prepare_heat3d_v6_randomblock_protocol.py --stage smoke16
python scripts/generate_heat3d_v6_randomblock_dataset.py \
  --config configs/heat3d_v6_randomblock/v6_randomblock_smoke16.yaml
python scripts/check_heat3d_v6_randomblock_dataset.py \
  --config configs/heat3d_v6_randomblock/v6_randomblock_smoke16.yaml \
  --dataset data/heat3d_v6_randomblock_smoke16_v0 \
  --manifest configs/heat3d_v6_randomblock/v6_randomblock_smoke16_manifest.json \
  --audit configs/heat3d_v6_randomblock/v6_randomblock_smoke16_audit.json
```

`pilot128` 和 `formal1024` 只在上一阶段通过并提交后，把命令中的 stage
替换为相应名称。`--dry-run` 只验证 240825-node mesh、全部 layout、幅值、
support 和写入计划，不求解、不落数据。
