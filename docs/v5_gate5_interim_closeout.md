# V5 Gate 5 interim closeout

统一 evaluator 使用 true-RMS 分母：`100 * sqrt(sum(error^2) / sum(true^2))`。
`test_iid` 与全部 hard roles 仅报告，不参与训练、标准化、超参数或 checkpoint 选择。

## MSE-best / final clean summary

| Run | Checkpoint (epoch) | Role | point-global rel RMSE % | sample-first CV rel RMSE % | raw CV RMSE K | amp | corr | hotspot K | top5 K | strong-q K | bg bias K | bg RMSE K | bg over | shape CV-RMSE | scale log-RMSE | legacy MSE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | best (e271) | valid_iid | 27.066 | 27.0652 | 0.192549 | 0.992689 | 0.953853 | 0.364016 | 0.537863 | 0.456507 | 0.0140089 | 0.0301197 | 0.840335 | 0.223835 | 0.181322 | 0.0476792 |
| B0 | best (e271) | test_iid | 27.3536 | 26.5617 | 0.236314 | 0.982563 | 0.958376 | 0.425176 | 0.573953 | 0.402734 | 0.0122668 | 0.029777 | 0.810858 | 0.21251 | 0.183594 | 0.0760224 |
| B0 | final (e600) | valid_iid | 28.164 | 24.3385 | 0.200508 | 0.978981 | 0.964352 | 0.365828 | 0.53821 | 0.461242 | 0.0070506 | 0.0230681 | 0.528796 | 0.201489 | 0.179675 | 0.0516261 |
| B0 | final (e600) | test_iid | 27.7207 | 23.7139 | 0.239834 | 0.981577 | 0.965729 | 0.4213 | 0.568098 | 0.4034 | 0.00539881 | 0.0240882 | 0.493581 | 0.194393 | 0.16705 | 0.0780766 |
| N0 | best (e120) | valid_iid | 31.488 | 28.3108 | 0.224819 | 1.01011 | 0.981697 | 0.421082 | 0.63495 | 0.489644 | 0.00584953 | 0.0229602 | 0.460353 | 0.147248 | 0.308533 | 0.0645314 |
| N0 | best (e120) | test_iid | 34.0829 | 29.6713 | 0.302439 | 0.942275 | 0.981519 | 0.533808 | 0.791049 | 0.622164 | 0.00410564 | 0.0257297 | 0.410914 | 0.143655 | 0.341419 | 0.118028 |
| N0 | final (e600) | valid_iid | 33.5973 | 27.7343 | 0.240301 | 1.04733 | 0.983269 | 0.418678 | 0.637081 | 0.494749 | 0.00666075 | 0.023162 | 0.417721 | 0.137644 | 0.302359 | 0.0734664 |
| N0 | final (e600) | test_iid | 35.979 | 28.1778 | 0.319043 | 0.979154 | 0.984949 | 0.539081 | 0.787225 | 0.626561 | 0.00546171 | 0.0261789 | 0.362995 | 0.128086 | 0.325003 | 0.131525 |
| N1 | best (e261) | valid_iid | 29.9645 | 25.864 | 0.217149 | 0.993944 | 0.97274 | 0.384901 | 0.588478 | 0.468166 | 0.0107879 | 0.025583 | 0.467736 | 0.180016 | 0.236892 | 0.058438 |
| N1 | best (e261) | test_iid | 28.7793 | 24.1708 | 0.254391 | 0.976165 | 0.971642 | 0.422389 | 0.608385 | 0.441117 | 0.00960021 | 0.0269299 | 0.442164 | 0.179971 | 0.22934 | 0.0841532 |
| N1 | final (e600) | valid_iid | 30.8192 | 25.5259 | 0.224649 | 1.01358 | 0.97323 | 0.388788 | 0.575585 | 0.451489 | 0.00782293 | 0.0223353 | 0.408283 | 0.175298 | 0.232076 | 0.0618191 |
| N1 | final (e600) | test_iid | 28.3734 | 23.8576 | 0.252285 | 0.998304 | 0.97418 | 0.428996 | 0.608654 | 0.443529 | 0.00493137 | 0.0223603 | 0.385972 | 0.170855 | 0.229602 | 0.0817965 |

可信模型门槛为 valid/test point-global true-RMS relative RMSE 均 `<20%`；B0=fail、N0=fail、N1=fail。

## MSE-best hard report-only summary

| Run | Role | point-global rel RMSE % | sample-first CV rel RMSE % | raw CV RMSE K |
|---|---|---:|---:|---:|
| B0 | hard_train_holdout | 74.4726 | 55.1186 | 6.19169 |
| B0 | hard_challenge_valid | 85.3492 | 56.1628 | 8.96962 |
| B0 | hard_challenge_test | 68.3269 | 60.9737 | 4.83216 |
| N0 | hard_train_holdout | 46.4669 | 37.1612 | 3.83756 |
| N0 | hard_challenge_valid | 46.4394 | 34.3564 | 4.7873 |
| N0 | hard_challenge_test | 42.9273 | 42.3929 | 3.01091 |
| N1 | hard_train_holdout | 44.7717 | 36.9017 | 3.69092 |
| N1 | hard_challenge_valid | 52.3054 | 34.6828 | 5.48544 |
| N1 | hard_challenge_test | 44.3725 | 42.5465 | 3.07821 |

## Native oracle bottleneck

| Run | joint % | oracle-scale % | oracle-shape % | physics-scale % | scale replacement gain pp | shape replacement gain pp | bottleneck |
|---|---:|---:|---:|---:|---:|---:|---|
| N0 | 28.3108 | 14.7248 | 22.2705 | 51.4232 | 13.5861 | 6.04034 | scale_dominant |
| N1 | 25.864 | 18.0016 | 16.5329 | 52.0352 | 7.86237 | 9.33108 | shape_dominant |

判断：`mixed_model_dependent`。

## N3 execution smoke

状态：`passed`；commit `83a1113`；output `output/heat3d_v5_gate5_preflight/N3_smoke_e1`。

## N3 e600 launch

状态：`running_e600`；host `wsl2`；training PID `151910`；commit `83a11132c49aa8c0beb740b7a5fca2f0b80344e4`；tmux `v5g5_n3_e600`。

output `output/heat3d_v5_runs/V4P5_07_native_pooled_latent_global_film`；log `output/heat3d_v5_logs/V4P5_07_native_pooled_latent_global_film.log`。

完整公式、split hash、checkpoint SHA/epoch 和每个 role 的全指标保存在各远程 run 的 `v5_metrics.json`，并镜像到 V5 registry result payload。
