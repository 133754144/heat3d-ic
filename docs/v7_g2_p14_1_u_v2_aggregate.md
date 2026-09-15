# V7 G2-P14.1：e200 高分辨率 valid-only 聚合

本 receipt 汇总冻结的 Heat3D-on-DeepOHeat-v1 e200 三个 checkpoint，在同一
`valid_iid` 128 个 physical cases 上的 native、IDW dense 与 U-v2 dense 结果。
未读取 P1i `test_iid`、sealed 或 DeepOHeat official100；没有训练或覆盖 checkpoint。

## 结果

|表示|输出域|sample-first relative RMSE %（mean ± sample SD）|point-global relative RMSE %|RMSE [K]|MAE [K]|peak RMSE [K]|推理时间 s（mean ± SD）|
|---|---:|---:|---:|---:|---:|---:|---:|
|native|1024|0.77994 ± 0.01395|0.89307 ± 0.02666|0.21159 ± 0.00443|0.17846 ± 0.00304|0.39121 ± 0.02730|20.884 ± 0.142|
|IDW dense|571,256|1.43150 ± 0.01659|1.43864 ± 0.01549|0.37427 ± 0.00415|0.27905 ± 0.00327|0.41690 ± 0.03523|—|
|U-v2 dense|571,256|0.81414 ± 0.00605|0.84818 ± 0.00913|0.21915 ± 0.00219|0.16360 ± 0.00013|0.38787 ± 0.03066|1013.04 ± 58.56|
|IDW oracle (GT support)|571,256|1.22215 ± 0.00000|1.22040 ± 0.00000|0.31821 ± 0.00000|0.22604 ± 0.00000|0.14185 ± 0.00000|—|

精确 per-seed 与所有字段见 [`v7_g2_p14_1_u_v2_aggregate.json`](v7_g2_p14_1_u_v2_aggregate.json)。

U-v2 latency 包括完整 query graph 构造与 direct-query model forward；IDW latency 延续
P14 evaluator 口径。U-v2 是神经 direct-query path，不能定义仅由 GT 1024 温度值驱动的
value-only oracle，因此 oracle 行仅提供 IDW lower-bound diagnostic，不做 model/reconstruction
误差相减分解。

## Provenance

- protocol：`configs/heat3d_v7/g2_p14_1_heat3d_u_v2_dense_evaluation_protocol.json`
- script commit：`7dd27e7da75412f44414356c369c374a66779a56`
- dataset：frozen 768 train / 128 valid，subset/label/normalization SHA 与 protocol 一致
- raw receipt SHA：见聚合 JSON `raw_receipts`
- learned parameter count added：0
- U-v2 query domain：DeepOHeat-v1 full grid `101×101×56=571256`
