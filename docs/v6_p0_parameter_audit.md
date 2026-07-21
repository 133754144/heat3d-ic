# V6-P0：V5 物理参数与数据审计

## 范围与复现

本审计只读取 `heat3d_v4_p5_clean_nohard_v0`，不训练、不推理、不生成数据。冻结输入为：

- 数据集 `manifest.json` SHA256：`248fd8c82eac352c9c224aa30800e26e3cc5f4b869262be5098f70d7acddf4cc`
- split map SHA256：`3b16913fef187f93a300221448425b5002f404ce8159407565311b44dbd00e08`
- 1073 个样本：train 672、valid_iid 128、test_iid 128、hard train/valid/test 121/12/12
- 全部样本均为 `16 x 16 x 4 = 1024` 节点，物理域 `10 x 10 x 2 mm`
- 审计脚本：`scripts/audit_heat3d_v5_physics_distribution.py`

WSL2 复现命令（只读）：

```bash
python scripts/audit_heat3d_v5_physics_distribution.py \
  --dataset data/heat3d_v4_p5_clean_nohard_v0 \
  --split-map configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json \
  --include-sample-records --output-json /tmp/v6_p0_v5_distribution.json
```

## generator、registry 与 metadata

- P5 builder 为 `scripts/build_heat3d_v4_p5_clean_nohard_dataset.py`；新补样经 `scripts/generate_heat3d_v4_p3c_smoke16.py` 调用相同 P3c generator/solver。
- 参数真源为 `configs/heat3d_v4/p3c_parameter_registry.json`；它明确写出 `R_contact=0_perfect_contact`，有限接触热阻只是 deferred implementation。
- q 是在 solver-safe interior 节点上生成后，按离散 control volume 的实际热源体积校准至目标总功率；边界节点不沉积功率。
- 每个样本的 `sample_meta.json` 保存 source/BC 类别、目标与积分功率、q 审计、QC、网格、材料和 solver 状态；数组含 `coords/k_field/q_field/bc_features/control_volume/temperature`。
- 审计兼容 `k_field[N,1]` 与 `k_field[N,3]`；全库 235 个 scalar、838 个 diagonal-3 样本。

## 文献矩阵口径

`docs/v6_p0_literature_matrix.csv` 收录 24 篇一手论文/作者公开预印本，覆盖 DeepOHeat、SAU-FNO、Therm-FM、MFIT、HotSpot、3D-ICE、PACT、ARTSim、COMSOL/FEM/FVM。每行同时保存 DOI/primary URL、原始单位和 SI 列；`NR` 表示原文未给出或不能从公开正文可靠提取，禁止用软件默认值或相邻论文补齐。换算只使用 `mm x 1e-3=m`、`um x 1e-6=m`、`mW x 1e-3=W`、`degC + 273.15=K`，温差 `1 degC=1 K`；compact thermal resistance、effective k、bulk k 与 contact resistance 不互相换算。

## 全量分布

下表为全部 1073 个样本的 `min / median / p95 / max`；CV 统计严格使用保存的 control volume。

| 量 | 单位 | min | median | p95 | max |
| --- | --- | ---: | ---: | ---: | ---: |
| min(k) | W/(m K) | 0.2782 | 2.176 | 30.00 | 30.00 |
| max(k) | W/(m K) | 30.00 | 270.9 | 636.7 | 801.4 |
| harmonic kz | W/(m K) | 0.5656 | 11.31 | 52.04 | 151.5 |
| min nonzero q | W/m3 | 5.530e2 | 2.481e6 | 6.690e7 | 3.788e8 |
| max nonzero q | W/m3 | 1.790e6 | 1.368e8 | 7.418e8 | 1.881e9 |
| source volume | m3 | 1.185e-9 | 1.600e-8 | 1.161e-7 | 1.161e-7 |
| total power | W | 0.05048 | 0.9292 | 4.295 | 9.006 |
| footprint power density | W/m2 | 504.8 | 9292 | 42951 | 90064 |
| top h | W/(m2 K) | 200.2 | 874.1 | 2676.9 | 2998.6 |
| peak DeltaT | K | 0.02281 | 2.597 | 42.73 | 277.0 |
| CV-mean DeltaT | K | 0.003242 | 0.2533 | 2.602 | 19.63 |
| CV-RMS DeltaT | K | 0.007520 | 0.5155 | 6.383 | 41.71 |
| peak Rth | K/W | 0.07051 | 2.686 | 33.17 | 182.5 |
| top heat fraction | 1 | 0.000814 | 0.02790 | 0.1853 | 0.5156 |
| bottom heat fraction | 1 | 0.4844 | 0.9721 | 0.9968 | 0.9992 |

面积功率密度使用固定 footprint `1e-4 m2`，故对应 `0.0505-9.006 W/cm2`。功率标定最大绝对误差 `1.78e-15 W`，最大能量平衡相对误差 `2.95e-11`；1073/1073 的 `q_rescale_factor=1`，q clipping 数为 0。

### 按 split（median）

| split | n | P W | max q W/m3 | Vsrc m3 | h W/(m2 K) | peak DeltaT K | Rth K/W | top fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 672 | 0.8369 | 1.107e8 | 1.600e-8 | 857.8 | 1.959 | 2.087 | 0.02496 |
| valid_iid | 128 | 0.9004 | 1.193e8 | 1.496e-8 | 871.2 | 1.731 | 1.859 | 0.02082 |
| test_iid | 128 | 0.8471 | 1.010e8 | 1.763e-8 | 864.6 | 1.659 | 2.053 | 0.02398 |
| hard_train_holdout | 121 | 1.662 | 2.594e8 | 1.481e-8 | 874.1 | 35.18 | 21.28 | 0.08833 |
| hard_challenge_valid | 12 | 1.508 | 2.293e8 | 1.881e-8 | 1444 | 32.51 | 25.47 | 0.08188 |
| hard_challenge_test | 12 | 1.595 | 4.210e8 | 1.807e-8 | 1206 | 42.53 | 18.65 | 0.1196 |

### 按 source（median）

| source | n | P W | max q W/m3 | Vsrc m3 | peak DeltaT K | Rth K/W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compact hotspot | 156 | 1.201 | 2.279e8 | 1.556e-8 | 3.732 | 3.565 |
| dual-z | 151 | 0.8505 | 9.324e7 | 1.244e-8 | 2.272 | 2.878 |
| elongated | 155 | 0.7443 | 5.883e7 | 1.896e-8 | 1.687 | 2.823 |
| multi-block | 156 | 3.596 | 2.842e8 | 4.696e-8 | 8.136 | 2.167 |
| TSV-adjacent | 151 | 0.7183 | 1.630e8 | 7.407e-9 | 3.199 | 5.504 |
| weak-background-hotspot | 146 | 1.097 | 2.710e8 | 7.407e-9 | 3.981 | 3.960 |
| weak-background | 158 | 0.5969 | 6.162e6 | 1.161e-7 | 0.602 | 1.211 |

### 按 BC（median）

| BC | n | h W/(m2 K) | peak DeltaT K | Rth K/W | top/bottom heat fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal_package | 355 | 895.2 | 2.301 | 2.767 | 0.0314 / 0.9686 |
| strong_forced_or_effective_heatsink | 352 | 2149 | 2.048 | 2.275 | 0.0596 / 0.9404 |
| weak_effective_air | 366 | 317.2 | 3.504 | 3.070 | 0.0132 / 0.9868 |

完整的 split/source/BC 交叉统计由脚本的 `by_split_source_bc` 给出，避免在文档中复制 126 个组合。

## 参数证据判定

判定只引用 `docs/v6_p0_literature_matrix.csv` 中的一手来源；`supported` 表示值和语义均有直接锚点，`partially_supported` 表示仅范围、材料类别或边界形式有证据，`unsupported` 表示当前具体实现没有对应证据。

| V5 参数/合同 | 判定 | 证据与限制 |
| --- | --- | --- |
| 10 x 10 mm footprint | supported | HBM 实测为 6x10/7x11 mm（L20），F2F package 为 10x10 mm（L22）。 |
| 2 mm 单块厚度与 4 个 z 节点 | unsupported | 文献层厚通常为 20-700 um 并显式分层（L01/L02/L03/L20）；L11 还以 adaptive vertical partitioning 为核心。没有来源支持用 4 层节点代表 2 mm 异质堆栈。 |
| 16x16 横向网格 | partially_supported | compact/AI 工作可使用较粗网格，但直接案例为 21x21、40x40、64x64、101x101 或更高（L01-L03/L12/L15）；V5 未做 generator-native mesh convergence。 |
| k=0.278-801 W/(m K) | partially_supported | 低 k、Si、Cu/TIM 类别有锚点（L01-L03/L21）；HBM 实测明确区分 140 W/(m K) memory-layer bulk-like、100/7 device-effective、2 polymer/solder through-plane（L20）。V5 上端 801 没有矩阵内直接材料锚点。 |
| diagonal anisotropic k | partially_supported | HBM 的 100 in-plane/7 through-plane effective tensor直接支持各向异性语义（L20），但 V5 随机区域张量并不等同特定 HBM 几何。 |
| q=5.53e2-1.88e9 W/m3 | partially_supported | 文献多以 tile/block total power 或 surface map 报告（L01/L02/L08/L21），不能把这些静默视作体积 q；V5 的 q 是由离散 source volume 反算。 |
| total power=0.050-9.006 W | partially_supported | 覆盖 DeepOHeat 的 mW 案例上方（L01/L02），低于或部分覆盖 10-20 W package 案例（L19）；并非统一器件等级的已验证分布。 |
| footprint density=0.050-9.006 W/cm2 | partially_supported | BSPDN 强调 5 um 非均匀功率图与热点（L21），但 V5 指标把总功率除以整个固定 footprint，不能代替局部 source flux。 |
| source volume=1.19e-9-1.16e-7 m3 | unsupported | 文献支持薄 source layer 和像素化 source（L01/L02），但没有来源支持 V5 的离散块体积范围及其与 2 mm 单块域的组合。 |
| compact/elongated/multi-block/dual-z/TSV-adjacent/background source families | partially_supported | 非均匀、热点、multi-component 与 TSV-aware source 形式有直接先例（L02/L21），但 V5 的 bbox 尺寸、占比和 family mixing probability 没有逐项文献标定。 |
| source 仅沉积在 interior control volumes | supported | 与体积热源和守恒 FVM 语义一致（L09-L12/L15）；全量 boundary power audit 为 0。 |
| top h=200-3000 W/(m2 K) | supported | DeepOHeat 333-1000（L01/L02），BSPDN 200/2500（L21）直接覆盖主要范围；接近 3000 的尾部只属小幅外推。 |
| ambient/bottom reference=300 K | supported | 298.15 K 在 L01/L02/L21 中直接使用，300 K 是其窄幅近似。 |
| top Robin + side adiabatic | supported | DeepOHeat 明确采用 convection 与 side adiabatic（L01/L02）。 |
| bottom 300 K Dirichlet | partially_supported | 298.15 K ambient 有直接锚点（L01/L02/L21），但这些工作通常用底部 convection 或完整 package path；固定等温底面不是同一物理边界。 |
| perfect contact everywhere | unsupported | HBM 测量显示 65 um polymer/solder interface layers会把 through-plane effective k 降至 7 W/(m K)（L20），FEA homogenization也显式建模 micro-bump/underfill/SiO2（L17/L18）。没有证据支持把 perfect contact 作为 V6 唯一生产工况。 |
| CV-integrated power calibration | supported | 这是离散守恒实现；全量误差小于 `1.78e-15 W`，与 FVM/energy-balance 方法要求一致（L09-L12/L15）。 |
| q clipping policy | supported as absent | metadata 与数组审计均为 0 clipping；这只证明 V5 数据事实，不代表任意 q 物理可信。 |
| DeltaT coverage=0.0228-277 K | partially_supported | 文献案例覆盖 mK/mW surrogate 到数十至上百 K 的不同尺度（L01/L02/L11/L19/L22），但它不是来自一个统一器件分布，不能作为单一 IID 温度先验。 |
| split/source/BC stratification | supported as data protocol | 这是防泄漏的数据合同而非材料参数；1073 个 sample ID 与 manifest/split map 完全一致。其物理泛化边界仍须由 V6 的 factor-held-out OOD 定义。 |

### 必须保持的 k 语义

- **bulk k**：可归属于单一材料的测量/表值，例如 L20 memory layer 140 W/(m K)。
- **effective k**：由微结构、层叠、TSV、micro-bump 或封装路径均匀化得到，例如 L20 device 100/7 W/(m K)、L17/L18 的 anisotropic equivalent tensor。
- **contact resistance**：界面温降/面热流关系，单位 m2 K/W；不能通过把相邻 bulk/effective k 改小来无声替代。

V6 metadata 必须为每个导热参数保存 `property_class: bulk|effective`、材料/均匀化对象和文献 ID；contact resistance 必须作为独立面参数保存。

## 结论

V5 在数值守恒、split 完整性和功率标定上可靠，但其物理覆盖不是一个可直接沿用的 V6 规格：垂向分辨率、固定底部等温边界、perfect contact、source volume 与 k 高端尤其需要重构。V6 pilot 应先验证这些物理合同，再讨论全量生成。
