# Heat3D v2 Optimizer Ablation Summary

## 结论

P3 的 A1/A2/A3 说明 Optax Adam/AdamW 基础设施已经可用，但 optimizer alone 没有解决 field-shape collapse。A1/A2 相比 A0 改善了 overall RMSE、centered correlation 和 top-k overlap，但 valid error、bin0 bias、field variance ratio 和 amplitude ratio 不如 A0。A0 仍作为 historical/reference baseline。

因此下一步不应继续只调 optimizer，而应进入 P4 model capacity ablation，检查更大的 latent width、processor steps 和 MLP depth 是否能缓解 over-smoothing / variance collapse。

## Best 指标表

| Run | Optimizer | Overall RMSE / MAE | Valid RMSE / MAE | bin0 bias / over_ratio | variance_ratio | corr | amplitude_ratio | top_k_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | manual GD lr=1e-2 | 3.94142286e-02 / 2.46786287e-02 | 2.73558855e-02 / 2.30635281e-02 | 1.89760931e-02 / 1.0 | 4.10044037e-02 | 4.56299540e-02 | 9.16572387e-02 | 6.25e-03 |
| A1 | Adam lr=1e-3 | 3.90378230e-02 / 2.51350654e-02 | 2.84380960e-02 / 2.42013317e-02 | 2.14052191e-02 / 1.0 | 3.57320844e-02 | 1.36138719e-01 | 8.15467583e-02 | 1.15234375e-02 |
| A2 | AdamW lr=1e-3 wd=1e-4 | 3.90378244e-02 / 2.51350905e-02 | 2.84380826e-02 / 2.42013181e-02 | 2.14052576e-02 / 1.0 | 3.57338164e-02 | 1.36145671e-01 | 8.15616168e-02 | 1.19140625e-02 |
| A3 | AdamW lr=3e-4 wd=1e-4 | 3.93442445e-02 / 2.56944472e-02 | 3.00267802e-02 / 2.58436535e-02 | 2.25032821e-02 / 1.0 | 3.84694186e-02 | 1.37145300e-01 | 9.18662357e-02 | 2.32421875e-02 |

## 诊断判断

Adam/AdamW 改善了部分 global error 和 top-k signal，但没有恢复足够的 field variance / amplitude。A3 的 top-k overlap 更高，但 valid error 和 bin0 bias 更差，属于 tradeoff，而不是明确优胜。

P4 应保留 A0 manual 和 A2 AdamW small model 作为对照，先跑一组更大容量 M1，而不是做 optimizer sweep。
