# Gate 6D Global Context coverage audit

距离空间仅由 train=672 拟合的 24 维 Global Context 标准化特征构成；valid_iid=128 只作查询。没有 target-derived distance feature，未访问 test/hard。

| error target | Pearson(distance,error) | Spearman(distance,error) |
|---|---:|---:|
| n3_sample_relative_rmse_pct | 0.390827 | 0.346926 |
| l2_sample_relative_rmse_pct | 0.432651 | 0.297381 |
| l2_minus_n3_sample_relative_rmse_pct | 0.125385 | -0.142963 |
| n3_point_sse_K2 | -0.021396 | 0.020744 |
| l2_point_sse_K2 | -0.023214 | -0.007462 |
| l2_minus_n3_point_sse_K2 | -0.005110 | -0.117961 |

distance min/median/mean/max = 0.659023 / 1.449255 / 1.522563 / 4.873137。

coverage distance 四分位、逐样本 nearest train ID，以及覆盖最差且误差最高的样本保存在 JSON。
