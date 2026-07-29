# V6-RandomBlock 1024 closeout

## Outcome

`heat3d_v6_randomblock_formal1024_v2` 已在 WSL2 完整生成：128 layout
groups × 8 variants = 1024 samples，group-level split 为 768/128/128。
运行耗时 `1607.12 s`，full-field archive SHA256 为
`83a21dd34c84047e4c14b8b63fb079333fc46a3ed13b46a4f71eb2fc12e15311`。
没有训练、模型推理、温度后筛选、样本替换或 P1h 修改。

所有 1024 个 peak ΔT 都在 30–150 K，实际范围
`43.481–146.897 K`，中位数 `89.196 K`。intended bins 精确为
`256/256/256/256`；realized bins 为 `256/256/255/257`。唯一偏差是
一个 v4 样本达到 `120.5305 K`，从 intended bin 2 跨入相邻 bin 3。

原始生成器将 formal 的“realized bins 必须精确相等”判为
`failed_temperature_gate`，该状态没有被改写。本 closeout 将数据记为
`complete_with_declared_realized_bin_deviation`：完整物理窗口通过，精确
realized-bin gate 差 1 个样本。为避免事后利用全 formal split 调参，
没有据此修改生成器或重跑数据；是否晋升 canonical 留给后续独立审计。

## Physical and distribution QC

- 最大能量守恒相对误差：`1.492e-10`；
- 最大线性残差：`1.199e-10`；
- CG iteration：median `1182`、P95 `1204`、max `1221`；
- `k` blocks：20–600 W/(m·K)，九个冻结锚点均覆盖；
- `q` blocks：3–10 个/layout，formal 最大 q `2.5599e10 W/m³`；
- support：全部 1469 个 blocks 非零覆盖，最小 2、median 19 points；
- 8 个 joint features 的标准化有效秩为 `8/8`；
- group split leakage 为零，同组八个变体共享坐标和 support。

## Reproduction

```bash
cd ~/myCodeGitOnly/heat3d-ic-randomblock
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
python scripts/check_heat3d_v6_randomblock_dataset.py \
  --config configs/heat3d_v6_randomblock/v6_randomblock_formal1024_v2.yaml \
  --dataset data/heat3d_v6_randomblock_formal1024_v2 \
  --manifest configs/heat3d_v6_randomblock/v6_randomblock_formal1024_v2_manifest.json \
  --audit configs/heat3d_v6_randomblock/v6_randomblock_formal1024_v2_audit.json \
  --allow-temperature-gate-failure
python scripts/check_heat3d_v6_randomblock_closeout.py \
  --dataset-root data/heat3d_v6_randomblock_formal1024_v2
```
