# V5 devbox / WSL2 最新训练结果（valid_iid-only）

本报告只读取两台机器已经完成的 e600 工件，并以同一 evaluator `1cf5bff739ecb51c5728ee6dd81a8cfc87c670fd` 重算四类 checkpoint。评估只访问 `train`（拟合 normalization/context）和 `valid_iid`；未访问 test、hard 或 sealed-IID，也未启动训练。

## 运行与完整性

| host | config | training commit | epochs | checkpoint reload | native audit | declared log |
|---|---|---:|---:|---|---|---|
| devbox | `V4P5_36_gate6m_v32_epoch_regroup_e600` | `414ab43` | 600 | `passed` | `True` | missing |
| wsl2 | `V4P5_38_gate6n_v36_r2r_mask_p005_e600` | `0c49084` | 600 | `passed` | `True` | missing |

train-only split counts are 672 train / 128 valid_iid, 1024 nodes/sample; parameter reload errors are exactly 0 and prediction replay errors are below the persisted 0.02 K tolerance for all four checkpoints.

## 统一 valid_iid 指标

relative 指标使用真实均方根分母；raw 指标为 control-volume 加权 RMSE。数值越低越好，amplitude 越接近 1、correlation 越高越好。

| run | checkpoint (epoch) | point-global % | sample-first % | raw K | shape CV-RMSE | scale log-RMSE | amp | corr | hotspot K | top5 K | strong-q K | low-ΔT bias K | low-ΔT RMSE K | over-ratio | base MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gate6m_b | point_global_best (e175) | 22.800185 | 20.895415 | 0.162217 | 0.152837 | 0.163202 | 1.010205 | 0.979827 | 0.297387 | 0.446200 | 0.350185 | 0.009444 | 0.022753 | 0.487528 | 0.033834 |
| gate6m_b | sample_first_best (e249) | 23.576065 | 19.961127 | 0.168304 | 0.148274 | 0.161025 | 0.981069 | 0.980975 | 0.300400 | 0.453653 | 0.352309 | 0.002400 | 0.017617 | 0.377934 | 0.036176 |
| gate6m_b | legacy_best (e175) | 22.800383 | 20.895839 | 0.162217 | 0.152838 | 0.163207 | 1.010212 | 0.979827 | 0.297390 | 0.446236 | 0.350188 | 0.009444 | 0.022753 | 0.487562 | 0.033835 |
| gate6m_b | final (e600) | 23.347487 | 20.369497 | 0.167425 | 0.146984 | 0.165625 | 0.999690 | 0.981075 | 0.307030 | 0.460516 | 0.372268 | 0.005253 | 0.018079 | 0.414474 | 0.035478 |
| gate6n | point_global_best (e231) | 21.944915 | 20.605917 | 0.157328 | 0.148421 | 0.178126 | 0.978197 | 0.980853 | 0.293789 | 0.448356 | 0.354706 | 0.010120 | 0.022048 | 0.496030 | 0.031344 |
| gate6n | sample_first_best (e543) | 22.728351 | 19.942082 | 0.163253 | 0.139849 | 0.175598 | 0.984124 | 0.982575 | 0.295443 | 0.444153 | 0.344556 | 0.003977 | 0.017159 | 0.372678 | 0.033621 |
| gate6n | legacy_best (e231) | 21.944841 | 20.606026 | 0.157327 | 0.148423 | 0.178126 | 0.978197 | 0.980853 | 0.293796 | 0.448389 | 0.354729 | 0.010120 | 0.022048 | 0.496110 | 0.031343 |
| gate6n | final (e600) | 22.643387 | 20.160080 | 0.162680 | 0.141205 | 0.173589 | 0.995176 | 0.982201 | 0.295825 | 0.442198 | 0.341918 | 0.004781 | 0.017843 | 0.389723 | 0.033371 |

## 诊断

- **Gate 6N 相对 Gate 6M-B 的 point-global best**：-0.855 percentage points，raw RMSE -0.00489 K，shape CV-RMSE -0.00442；但 scale log-RMSE 变化 +0.01492、amplitude 变化 -0.03201（出现轻微欠幅）。
- **sample-first best**：Gate 6N 为 19.942%，Gate 6M-B 为 19.961%；这是轻微改善，但 sample-first checkpoint 仅作诊断，不能替换当前保存的 `valid_base_mse`/point-global 选择。
- **Gate 6N final**：相对 Gate 6M-B final 的 point-global、sample-first、raw、shape、hotspot/top5/strong-q 与 low-ΔT RMSE 均下降；scale log-RMSE 仍上升。
- **attention**：两者均非 NaN/坍缩，N 的 normalized entropy 更高、maximum weight 更低，说明分布更弥散；与连续物理特征的相关性以 `log_inverse_kz_relative` 最强（约 -0.32），q/source fraction 相关较弱。
- **阈值**：两组 point-global valid RMSE 均大于 20%，因此按冻结 V5 `<20%` 可信门槛均不通过；N 的 sample-first best 低于 20% 仅作为诊断，不改变主选择。

## 可复现工件

- 完整 evaluator payload（未纳入 Git）：`/tmp/V4P5_36_gate6m_v32_epoch_regroup_e600_valid_only.json`、`/tmp/V4P5_38_gate6n_v36_r2r_mask_p005_e600_valid_only.json`。
- compact JSON：`configs/heat3d_v5/gate6_remote/latest_valid_only_summary.json`；CSV：`configs/heat3d_v5/gate6_remote/latest_valid_only_metrics.csv`。
- Gate6M registry 已登记 B；Gate6N registry 已补充 V5 result columns 并登记 N；两行均为 `completed_valid_only`，required metrics complete=true。
