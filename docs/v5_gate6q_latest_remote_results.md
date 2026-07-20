# Gate 6Q 最新远程训练结果与诊断

本次检查严格避免 GPU 调用。devbox 有 V43 正在训练，因此只读取已完成 V42 的落盘 summary/diagnostics；没有在 devbox 运行 evaluator。WSL2 无训练进程，使用 `JAX_PLATFORMS=cpu` 对 V44 做 valid_iid 四 checkpoint true-RMS 评估。test/hard/sealed 均未访问。

## 远程状态

| host | status | run | commit | evaluator |
|---|---|---|---|---|
| devbox | V43 running (PID 167164/167165) | V42 partial read-only | `580cf50` | not run |
| wsl2 | idle | V44 completed e600 | `580cf50` | CPU `580cf504` |

## WSL2 V44 true-RMS valid_iid

| checkpoint | epoch | point-global % | sample-first % | raw CV K | shape | scale log | amp | corr | base MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| point_global_best | 329 | 22.0602 | 18.9078 | 0.158943 | 0.140377 | 0.153252 | 1.002418 | 0.982901 | 0.031674 |
| sample_first_best | 477 | 22.7110 | 18.4656 | 0.163984 | 0.136548 | 0.157749 | 0.980819 | 0.983653 | 0.033570 |
| legacy_best | 329 | 22.0595 | 18.9073 | 0.158937 | 0.140375 | 0.153247 | 1.002412 | 0.982902 | 0.031672 |
| final | 600 | 22.5433 | 18.6558 | 0.162789 | 0.137982 | 0.155456 | 0.991470 | 0.983206 | 0.033076 |

V44 point-global best 未通过 `<20%`；sample-first best 为 18.4656%，但不能替代 point-global/base-MSE checkpoint。final 相比 point-global best 的 point-global 和 raw CV RMSE 回退。

## devbox V42（未重放）

V42 已完成 e600，base-MSE/point-global best 为 e257，sample-first best 为 e591。由于 devbox V43 正在运行，本轮没有在 devbox 执行 JAX/evaluator；下表的 `valid_iid_error_pct`、native joint、shape/scale 为训练时 persisted diagnostics，不能与 WSL2 当前 true-RMS payload 混写成同一正式比较。

| epoch | base MSE | persisted rel % | native joint % | raw CV K | shape | scale abs log error |
|---:|---:|---:|---:|---:|---:|---:|
| 257 | 0.031321 | 21.9369 | 19.2508 | 0.156348 | 0.143009 | 0.107763 |
| 591 | 0.032917 | 22.4892 | 17.7063 | 0.161517 | 0.132688 | 0.102631 |
| 600 | 0.033652 | 22.7393 | 17.8536 | 0.163087 | 0.133326 | 0.102842 |

## 诊断结论

- V44 相比历史 V38：sample-first、shape 和 scale log-RMSE 改善，但 point-global 约退化 0.115 个百分点、raw CV RMSE 轻微变差，属于目标权衡。
- V44 attention 有效且未坍缩：point-global best entropy 约 0.829，平均最大权重约 0.050；与 `log_inverse_kz_relative` 的相关性约 −0.353，source/q 相关性较弱。
- V44 final 的 oracle-scale 13.798%、oracle-shape 10.932%，joint 18.656%，physics-scale 51.252%；不能把收益归因于单一 shape 或 scale 分支。
- V43 当前没有落盘结果；待其自然结束后再做统一 CPU evaluator，不应根据运行中状态选择 checkpoint。

## 工件与 registry

- JSON：`configs/heat3d_v5/gate6q/gate6q_latest_remote_results.json`；CSV：`configs/heat3d_v5/gate6q/gate6q_latest_remote_results.csv`。
- `v5_gate6q_training_registry.csv` 已登记 V44 完整 valid-only 结果，V42 标记为 partial metrics；V43 保持未完成。
