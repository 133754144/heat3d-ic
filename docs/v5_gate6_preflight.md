# V5 Gate 6A/6B preflight

状态：Gate 6A 完成；Gate 6B 三个 e100 配置已准备但均未启动。未启动正式 e600 或 multi-seed。

## 数据与 checkpoint 合同

- 仅加载 `train=672`、`valid_iid=128`；未加载或评估 test/hard。
- N3 best epoch 402，checkpoint SHA256：
  `3baebb9b751bf6054f36308444cdefe7a7b4f343665164b0aabdfe2610b5a228`。
- normalization 与 Global Context standardizer 均仅由 train 拟合。
- Gate 6A evaluator commit：`cb236abe15b122dd7d7de9aa93c52b173d8aae7d`。
- 完整逐样本、四分位、gradient cosine、Q4/top-5/top-10 数据见
  `configs/heat3d_v5/gate6a/V4P5_07_N3_best_e402_diagnostic.json`。

## Gate 6A 结果

| split | joint % | oracle-scale % | oracle-shape % | physics-scale % |
|---|---:|---:|---:|---:|
| train | 4.3067 | 3.9509 | 1.7820 | 67.4996 |
| valid_iid | 24.0756 | 17.5261 | 17.6443 | 65.1565 |

valid_iid 四项 loss 与全局梯度范数：

| loss | mean | global grad | backbone | shape decoder | scale head | FiLM | bypass |
|---|---:|---:|---:|---:|---:|---:|---:|
| shape CV | 0.027834 | 0.175057 | 0.147250 | 0.091718 | 0 | 0.016677 | 0.016497 |
| log-scale | 0.030259 | 1.134649 | 0.165882 | 0 | 1.108281 | 0.177836 | 0 |
| relative-field | 0.052483 | 0.533597 | 0.197753 | 0.086960 | 0.471661 | 0.123942 | 0.015240 |
| raw-absolute | 0.029545 | 0.328836 | 0.118274 | 0.039784 | 0.299038 | 0.055949 | 0.002769 |

loss 均值最大/最小为 1.89×，但梯度范数为 6.48×，因此 `1/1/1/1`
在优化动力学上失衡。log-scale 对 scale head 的梯度占主导，而 shape 梯度最弱。

valid_iid gradient cosine 的关键冲突为：log-scale vs relative `-0.5814`，
raw vs relative `-0.5480`；shape vs relative 为 `0.2694`。

### valid_iid true CV-RMS DeltaT 四分位

| quartile (K) | shape | scale | relative | raw |
|---|---:|---:|---:|---:|
| Q1 0.0095–0.1676 | 0.02608 | 0.02371 | 0.04615 | 0.00040 |
| Q2 0.1676–0.3569 | 0.03223 | 0.03896 | 0.06758 | 0.00437 |
| Q3 0.3569–0.7067 | 0.02455 | 0.01804 | 0.04149 | 0.01232 |
| Q4 0.7067–2.0236 | 0.02848 | 0.04032 | 0.05472 | 0.10108 |

### valid_iid total power 四分位

| quartile (W) | shape | scale | relative | raw |
|---|---:|---:|---:|---:|
| Q1 0.0835–0.5553 | 0.02977 | 0.04347 | 0.06640 | 0.00981 |
| Q2 0.5553–0.9004 | 0.02261 | 0.02277 | 0.04147 | 0.01426 |
| Q3 0.9004–1.3899 | 0.03411 | 0.02345 | 0.05217 | 0.05829 |
| Q4 1.3899–5.3572 | 0.02484 | 0.03135 | 0.04989 | 0.03582 |

true-DeltaT Q4 仅占 25% 样本，却贡献 85.53% raw loss，raw 梯度范数为全体
raw 梯度的 1.223×。top-5/top-10 分别贡献 47.78%/49.62% raw loss，缩放后的
raw 梯度范数分别为全体的 1.443×/1.416×。

## 候选权重冻结

权重顺序为 `shape / scale / relative / raw`：

| variant | weights | 依据 |
|---|---|---|
| FT-L0 | `1/1/1/1` | 继续训练控制组 |
| FT-L1 | `1/1/0.5/1.5` | 优先候选；减弱与 scale/raw 冲突的 relative 梯度，并加强有明确尾部集中证据的 raw loss |
| FT-L2 | `1.5/0.5/0.5/1.5` | 诊断支持的 shape-balanced 候选；提高最弱 shape 梯度、抑制占主导的 log-scale 梯度，同时保留尾部修正 |

未主观增加其他 shape/scale 候选。Gate 6B 只能使用 train 优化、valid_iid 选择；
test/hard 必须等候选权重冻结后再评估。

## Gate 6B 共同合同

- 从相同 N3 e402 params-only checkpoint 严格加载；optimizer state 重新初始化。
- `native_branch_mode: joint`，全参数可训练。
- e100，B28，相同 seed、batch order、graph seed 与 split。
- LR `7.5e-5`，为 N3 `5e-4` 的 0.15 倍；warmup-cosine 与 min-LR 同比例缩放。
- 每 epoch 报告核心指标。
- epoch 0（加载 N3 e402 后、首次 optimizer update 前）同时参与
  `valid_base_mse` 与 true-RMS point-global best 选择。
- `params_best.pkl` 按 `valid_base_mse` 保存；
  `params_best_valid_point_global.pkl` 按 true-RMS point-global relative RMSE 保存；
  同时保存 `params_final.pkl`。
- prediction export、final probe 与 post-training diagnostics 均关闭，避免触及 test/hard。

## 手动启动命令

以下命令仅供用户手动执行，本次未执行：

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_08_gate6b_ft_l0_unit.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_09_gate6b_ft_l1_tail_balanced.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_10_gate6b_ft_l2_shape_balanced.yaml
```
