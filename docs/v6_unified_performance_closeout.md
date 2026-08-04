# V6 unified performance benchmark

本报告不重训、不调参，仅访问 valid_iid；test/sealed 保持关闭。P1i 代表连续物理参数的 V6 layered family，random-block 为跨数据集运行时 OOD 诊断。正式质量仍来自 128 valid × 3 seed，32 样本只用于计时队列与分辨率诊断。

生产/FVM执行 commit：`04c63db7134308768b04914339a5d5fae67e56de`；direct-N 诊断执行 commit：`df2ef336062dfb09db57d07f7d5b592d071485bf`。

## Frozen quality

- support point-global: 2.027348 ± 0.094738%
- support sample-first: 1.629402 ± 0.013132%
- full-field point-global: 3.442626 ± 0.058435%

上述正式精度来自 128 个 valid_iid × 3 seeds；以下 32 样本只用于统一计时队列和分辨率诊断，不能替代正式精度。

## Four-state timing

| route | process cold (s) | new topology/JIT-cached (s) | known topology/new physics (s) | fully cached (s) |
|---|---:|---:|---:|---:|
| model_support | 9.7630 | 1.6488 | N/A | 0.0023 |
| production_reconstruction | 9.9705 | 1.7638 | N/A | 0.0100 |
| fvm | 2.3379 | N/A | 1.6864 | 1.6207 |

表内是连续 wall-clock 的逐样本中位数；N/A 表示冻结数值合同下该状态不可定义，并非零耗时。process cold 包含独立进程启动至预测序列化完成；fully cached 是持久进程内图/JIT/重建映射均缓存的重复推理。

## P1i resolution, accuracy, and runtime

| N | model cold (s) | FVM cold (s) | cold speedup | model cached (s) | FVM cached (s) | cached speedup | model PG (%) | FVM PG (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 9.9125 | 0.6521 | 0.07× | 0.003720 | 0.006258 | 1.68× | 3.120 | 1.190 |
| 8192 | 9.9853 | 0.6610 | 0.07× | 0.003882 | 0.012109 | 3.12× | 3.307 | 1.191 |
| 16384 | 10.0505 | 0.6747 | 0.07× | 0.004588 | 0.025245 | 5.50× | 3.382 | 1.191 |
| 32768 | 9.9768 | 0.7138 | 0.07× | 0.005673 | 0.062941 | 11.09× | 3.385 | 1.192 |
| 65536 | 10.0761 | 0.8191 | 0.08× | 0.006358 | 0.157089 | 24.71× | 3.336 | 1.192 |
| 240825 | 10.1534 | 2.3056 | 0.23× | 0.012970 | 1.669673 | 128.73× | 3.072 | 0.070 |

P1i 在 fully-cached system-level 口径下从 4096 节点首次超过 1×；process-cold 在已测全部分辨率均未超过 1×。这是相同节点数但非同 DOF 布置、且精度不同的系统级比较；不能称为 matched-accuracy speedup。

## Direct-N structured-support OOD diagnostic

| family | N | cold status/median (s) | cached status/median (s) | point-global (%) |
|---|---:|---:|---:|---:|
| p1i | 4096 | passed / 10.7289 | passed / 0.0270 | 403.601 |
| p1i | 8192 | passed / 11.3123 | passed / 0.0260 | 171.741 |
| p1i | 16384 | passed / 11.9396 | passed / 0.0314 | 42.997 |
| p1i | 32768 | passed / 12.1711 | passed / 0.0437 | 14.730 |
| p1i | 65536 | passed / 13.1804 | passed / 0.0688 | 4.715 |
| p1i | 240825 | passed / 17.6450 | passed / 0.2057 | 215.326 |
| randomblock | 4096 | not_run_resolution_infeasible / N/A | not_run_resolution_infeasible / N/A | N/A |
| randomblock | 8192 | not_run_resolution_infeasible / N/A | not_run_resolution_infeasible / N/A | N/A |
| randomblock | 16384 | not_run_resolution_infeasible / N/A | not_run_resolution_infeasible / N/A | N/A |
| randomblock | 32768 | not_run_resolution_infeasible / N/A | not_run_resolution_infeasible / N/A | N/A |
| randomblock | 65536 | passed / 14.6551 | passed / 0.0572 | 55.287 |
| randomblock | 240825 | passed / 19.2022 | passed / 0.1973 | 63.201 |

该表只验证 checkpoint 在直接 N 节点结构化支撑上的运行兼容性。它不属于冻结生产路线，不参与 speedup、模型或分辨率选择；失败/OOM 也是正式诊断结果。

P1i direct-N 低于 20% 的诊断分辨率为 [32768, 65536]；该路线随 N 明显非单调，240825 再次失效，因此不能据局部低误差点宣称可泛化的高分辨率直接推理。

## Governance

Cold 在预测序列化完成时截止；SHA、指标、oracle、JSON 与 checker 均位于生产计时之外。连续 wall-clock 是主口径，阶段计时只用于归因，禁止相加替代。

random-block 使用 V6_03 layer checkpoint，仅为 runtime-only structured-support OOD diagnostic；其约 108% point-global 失败结果禁止形成生产加速结论。所有 direct-N 模型结果同样只作 structured-support OOD compatibility diagnostic，不进入生产 speedup 或模型/分辨率选择。

random-block 的 4096/8192/16384/32768 structured-FVM 对完整 32 样本队列均因至少一个几何块解析不足而 fail-closed；只有 65536 与 240825 可运行。其首次 >1× 仅是运行时诊断，不是生产结论。

历史 layer 数字均绑定其原数据集、样本、硬件、batch、预热、重复次数、计时边界与求解器定义；没有一项可按名称直接并入本轮统一表。旧 4.86× 仅称 cached steady-state speedup。

模型路线使用冻结 1024 source-aware support 推理，再重建/输出到 N 节点；FVM 使用准确 N 节点的合法结构化网格。节点数匹配但 DOF 放置不同。

首次 fully-cached system speedup >1×：P1i=4096，random-block=65536。

完整 median/mean/std/P95、阶段耗时、RAM/GPU 显存、CG iterations 与精度见 `configs/heat3d_v6_p1i/v6_unified_performance_timing.csv`、`v6_unified_performance_resolution.csv` 和 `v6_unified_performance_accuracy.csv`。原始执行 JSON 位于 `configs/heat3d_v6_p1i/v6_unified_raw/`。
