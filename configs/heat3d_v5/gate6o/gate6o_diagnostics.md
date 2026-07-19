# Gate 6O train/valid-only diagnostics

仅访问 `train` 与 `valid_iid`；`test/hard/sealed` 均未访问。

| field | point-global % | sample-first % | raw CV K | shape CV | scale log |
|---|---:|---:|---:|---:|---:|
| e231 | 21.944593 | 20.605934 | 0.157326 | 0.148426 | 0.178120 |
| e543 | 22.727811 | 19.941592 | 0.163248 | 0.139847 | 0.175594 |
| shape e231 + scale e543 | 22.839331 | 20.598101 | 0.163307 | 0.148426 | 0.175594 |
| shape e543 + scale e231 | 21.841259 | 19.981103 | 0.157351 | 0.139847 | 0.178120 |
| ensemble α=0.5 | 22.017989 | 19.897675 | 0.158013 | 0.140173 | 0.175561 |
| e231 train-affine scale | 22.080700 | 20.448275 | 0.158024 | 0.148426 | 0.173680 |
| e543 train-affine scale | 22.575918 | 20.009046 | 0.162236 | 0.139847 | 0.174279 |

e543 相对 e231 的 shape CV-RMSE 差为 `-0.008579`，paired bootstrap
95% CI 为 `[-0.012399, -0.005050]`；sample-first 差为 `-0.664342`
个百分点，95% CI 为 `[-1.283552, -0.065540]`。point-global 差为
`+0.783218` 个百分点，其区间跨零。

Q1–Q3 的 point-SSE 净差均有利于 e543；Q4 净增加 `216.027 K²`，
超过前三个 quartile 的改善，说明 point-global 退化集中在高温升尾部的
scale 路径。`shape_e543+scale_e231` 同时达到 point-global
`21.841259%` 和 sample-first `19.981103%`，支持冻结 e543 shape 后
校准 scale。

按 Gate 6O 预注册 frozen-shape 规则选择 `e543` 作为 Stage 2
初始化。branch swap、ensemble 与 affine calibration 不参与该选择。
