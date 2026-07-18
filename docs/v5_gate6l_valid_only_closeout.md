# Gate 6L valid-only frozen closeout

状态：`completed_valid_iid_four_checkpoint`。本轮只重放既有 checkpoint，没有训练、改参或重新选择 checkpoint；评估角色仅 `valid_iid`，`test/hard/sealed` 均未访问，也没有自动晋级。

- evaluator commit: `ebbeed01b34b2e790bc3d7b87a6d64a8c6c70d8b`
- training commit: `461d810`
- 统一公式：`heat3d_v5_clean_metrics_v2_true_rms`
- 样本：128，节点/样本：1024；normalization/global context 仅由 train=672 拟合

## 四 checkpoint 统一结果

| 模型 | checkpoint | epoch | point-global % | sample-first % | raw CV K | shape CV | scale log | amplitude | correlation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O075 | point_global_best | 280 | 23.4299 | 19.8446 | 0.167186 | 0.150128 | 0.157300 | 0.999360 | 0.980136 |
| O075 | sample_first_best | 305 | 24.2256 | 19.4684 | 0.172405 | 0.149628 | 0.162594 | 0.975698 | 0.980241 |
| O075 | legacy_best | 280 | 23.4304 | 19.8449 | 0.167189 | 0.150128 | 0.157307 | 0.999363 | 0.980136 |
| O075 | final | 600 | 24.4679 | 19.7798 | 0.174185 | 0.149371 | 0.160538 | 0.995656 | 0.980089 |
| Dual | point_global_best | 298 | 22.9329 | 21.3879 | 0.162800 | 0.152925 | 0.184347 | 1.004228 | 0.979296 |
| Dual | sample_first_best | 316 | 23.5926 | 20.7025 | 0.168099 | 0.148514 | 0.186268 | 0.989299 | 0.980606 |
| Dual | legacy_best | 298 | 22.9333 | 21.3880 | 0.162802 | 0.152925 | 0.184351 | 1.004226 | 0.979296 |
| Dual | final | 600 | 23.3623 | 20.9600 | 0.166815 | 0.150030 | 0.183437 | 0.994220 | 0.979921 |

完整的 hotspot/top-5/strong-q、low-ΔT、legacy MSE、SHA256、参数量与 reload 结果见 JSON/CSV 工件。

## Point-global-best 的冻结分层

| 模型 | 分层 | n | point-global % | sample-first % | raw CV K | shape CV | scale log |
|---|---|---:|---:|---:|---:|---:|---:|
| O075 | Q1 | 32 | 18.0703 | 17.8950 | 0.018105 | 0.148404 | 0.112823 |
| O075 | Q2 | 32 | 27.2788 | 22.2780 | 0.068232 | 0.160231 | 0.196545 |
| O075 | Q3 | 32 | 20.7009 | 18.8278 | 0.112357 | 0.140247 | 0.128140 |
| O075 | Q4 | 32 | 23.7573 | 20.3776 | 0.306915 | 0.151631 | 0.176619 |
| O075 | nominal_to_hard | 81 | 21.2304 | 21.4961 | 0.157454 | 0.156863 | 0.172927 |
| O075 | Q2_intersection_nominal_to_hard | 21 | 31.4686 | 25.9555 | 0.078472 | 0.168437 | 0.240312 |
| O075 | scale_abs_error_top10pct | 13 | 37.6979 | 38.8301 | 0.370203 | 0.192238 | 0.382767 |
| O075 | scale_signed_low_p10 | 13 | 37.2922 | 33.4854 | 0.358116 | 0.191623 | 0.356768 |
| O075 | scale_signed_central_p10_p90 | 102 | 16.9494 | 16.3091 | 0.114205 | 0.142816 | 0.082006 |
| O075 | scale_signed_high_p90 | 13 | 29.1178 | 33.9436 | 0.211254 | 0.166008 | 0.252146 |
| Dual | Q1 | 32 | 21.7495 | 20.9437 | 0.021806 | 0.149376 | 0.185206 |
| Dual | Q2 | 32 | 30.2186 | 25.4549 | 0.076238 | 0.172245 | 0.224832 |
| Dual | Q3 | 32 | 22.2064 | 19.2064 | 0.119831 | 0.139500 | 0.148905 |
| Dual | Q4 | 32 | 22.7542 | 19.9465 | 0.292179 | 0.150580 | 0.170036 |
| Dual | nominal_to_hard | 81 | 21.9586 | 22.9993 | 0.162006 | 0.162784 | 0.199726 |
| Dual | Q2_intersection_nominal_to_hard | 21 | 34.0057 | 29.3009 | 0.085752 | 0.186860 | 0.266680 |
| Dual | scale_abs_error_top10pct | 13 | 37.6479 | 45.6791 | 0.289800 | 0.200142 | 0.444128 |
| Dual | scale_signed_low_p10 | 13 | 34.8587 | 35.8262 | 0.332496 | 0.195900 | 0.405746 |
| Dual | scale_signed_central_p10_p90 | 102 | 18.2938 | 16.9493 | 0.126886 | 0.141466 | 0.100045 |
| Dual | scale_signed_high_p90 | 13 | 35.1253 | 41.7756 | 0.155193 | 0.199859 | 0.302407 |

## 逐样本配对结论

差值方向统一为右模型减左模型；误差指标中负值表示右模型改善。CI 为固定 seed、20,000 次 paired bootstrap 的 95% 区间。

| 对比 | 指标 | 差值 | 95% CI | right 改善概率 | win rate | 中位差 |
|---|---|---:|---:|---:|---:|---:|
| O075_minus_V32 | point_global_relative_rmse_pct | 1.021550 | [-0.949222, 2.697632] | 0.1540 | 0.4922 | 0.100236 |
| O075_minus_V32 | sample_first_cv_relative_rmse_pct | -1.190203 | [-2.813643, 0.300050] | 0.9364 | 0.5000 | -0.013169 |
| O075_minus_V32 | raw_cv_weighted_rmse_K | 0.007119 | [-0.005648, 0.019495] | 0.1495 | 0.5000 | 0.000043 |
| O075_minus_V32 | shape_cv_rmse | 0.004431 | [-0.000150, 0.009102] | 0.0290 | 0.4297 | 0.004576 |
| O075_minus_V32 | scale_log_rmse | -0.040182 | [-0.080631, -0.005822] | 0.9919 | 0.5703 | -0.013788 |
| Dual_minus_V32 | point_global_relative_rmse_pct | 0.524550 | [-0.447423, 1.491637] | 0.1427 | 0.3906 | 0.687873 |
| Dual_minus_V32 | sample_first_cv_relative_rmse_pct | 0.353072 | [-0.661393, 1.358698] | 0.2392 | 0.4141 | 0.540771 |
| Dual_minus_V32 | raw_cv_weighted_rmse_K | 0.002733 | [-0.004492, 0.009778] | 0.2271 | 0.4141 | 0.001747 |
| Dual_minus_V32 | shape_cv_rmse | 0.007228 | [0.002547, 0.012224] | 0.0006 | 0.4062 | 0.005050 |
| Dual_minus_V32 | scale_log_rmse | -0.013135 | [-0.045039, 0.014435] | 0.7807 | 0.5625 | -0.004538 |
| Dual_minus_O075 | point_global_relative_rmse_pct | -0.496999 | [-2.029876, 1.311448] | 0.7066 | 0.3984 | 0.740163 |
| Dual_minus_O075 | sample_first_cv_relative_rmse_pct | 1.543275 | [0.204637, 2.903002] | 0.0118 | 0.4141 | 0.764137 |
| Dual_minus_O075 | raw_cv_weighted_rmse_K | -0.004385 | [-0.015956, 0.007731] | 0.7538 | 0.4141 | 0.001657 |
| Dual_minus_O075 | shape_cv_rmse | 0.002797 | [-0.001947, 0.007738] | 0.1259 | 0.4922 | 0.001047 |
| Dual_minus_O075 | scale_log_rmse | 0.027047 | [0.003985, 0.049489] | 0.0117 | 0.4375 | 0.006079 |

Tail contribution 的逐样本、Q4 与 scale-tail 贡献保存在主 JSON 和 `gate6l_paired_samples.csv`。

## 冻结判断

- O075 相对 V32 改善 sample-first 与 scale，但 point-global、raw CV 和 shape 退化。
- Dual 相对 V32 的 point-global/raw 更接近，但 shape 与 sample-first 退化；相对 O075 虽恢复部分 Q4 point-global SSE，却显著损失 sample-first 与 scale。
- 按冻结的 point-global 唯一晋级准则，O075 与 Dual 均不自动晋级；V32 保持当前候选地位。本结论不触发新实验。
