# Gate 5 N0/N1 e600 preflight

Smoke 与 e10 均为真实 P5 execution/calibration，不是正式性能结果；未启动 e600。

| run | epochs | best epoch/MSE | final MSE | joint rel | oracle-scale rel | oracle-shape rel | peak GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| n0_smoke | 1 | 1/0.379609 | 0.379609 | 0.680599 | 0.619137 | 0.48666 | 9169.921875 |
| n1_smoke | 1 | 1/0.338641 | 0.338641 | 0.66317 | 0.630123 | 0.343915 | 9169.921875 |
| n0_e10 | 10 | 9/0.196352 | 0.219491 | 0.569898 | 0.43097 | 0.355707 | 9169.921875 |
| n1_e10 | 10 | 9/0.210708 | 0.241309 | 0.576063 | 0.41664 | 0.359584 | 9169.921875 |

## Execution checks

- n0_smoke: passed=true；真实 train/valid=`672/128`、1024 nodes、B28；checkpoint/NPZ reload 通过；global-context standardizer 仅由 train 拟合；首 batch `26.80s`，稳态 batch 中位数 `2.68s`。
- n1_smoke: passed=true；真实 train/valid=`672/128`、1024 nodes、B28；checkpoint/NPZ reload 通过；global-context standardizer 仅由 train 拟合；首 batch `26.24s`，稳态 batch 中位数 `2.77s`。

N1 runtime audit 确认 pooled latent width=`96`；N0 pooled width=`0`。两者均无 OOM/NaN/Inf。
N0 最终接受的是 `N0_smoke_e1_r4`；此前三次尝试暴露并修复了 native metric 聚合、runtime audit 字段和 GPU graph-reduction replay 容差问题，失败产物未作为通过结果。

## e10 calibration

完整逐 epoch 四项 train/valid loss、native 指标、三组梯度和 timing 均在 machine-readable JSON。
- n0_e10: joint `0.6806->0.5699`，shape CV-RMSE `0.6191->0.4310`，scale error `0.8092->0.4243`，amplitude `0.5143->0.9337`；backbone/shape/scale gradient 中位数 `3.09/2.26/4.37`；稳态 epoch 中位数 `89.13s`。
- n1_e10: joint `0.6632->0.5761`，shape CV-RMSE `0.6301->0.4166`，scale error `0.5324->0.4199`，amplitude `0.7146->0.9876`；backbone/shape/scale gradient 中位数 `6.88/2.44/24.80`；稳态 epoch 中位数 `78.50s`。

这些 e10 数值仅用于校准和 loss/gradient audit，不用于 N0/N1 正式性能排序。

## Loss freeze

四项权重冻结为 `shape/log-scale/relative/raw = 1/1/1/1`，N0/N1 共用。最大 loss 中位数比为 `2.95x`，低于 `10x` 主导阈值；三组核心梯度均有限且非零。详细依据见 `configs/heat3d_v5/v5_gate5_loss_freeze.json`。

## Frozen e600 candidates

- N0: `configs/heat3d_v5/generated/V4P5_05_native_physics_only.yaml`
- N1: `configs/heat3d_v5/generated/V4P5_06_native_pooled_latent.yaml`
- Registry: `configs/heat3d_v5/v5_gate5_native_preflight_registry.csv`

解析后严格 diff 仅允许 identity/output 标识和 `model.scale_head_mode` 不同；dataset、split、model 其余字段、optimizer、LR、B28、seed、loss 和 checkpoint selection 均一致。正式 best 仍按最低 `valid_base_mse`。本轮未启动 e600。
