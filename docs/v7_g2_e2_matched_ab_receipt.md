# G2-E2 matched A/B runtime receipt

状态：`PASS_MATCHED_RUNTIME_FINITE`。本收据只证明运行时闭环，不提供 accuracy evidence。

## 固定条件

同一 devbox WSL2 RTX 5070、JAX 0.9.1/jaxlib 0.9.1、Python 3.14.3、`--xla_gpu_deterministic_ops=true`、B24、valid batch 32、legacy 单步路径。A 为 frozen G1 P1i Heat3D workload，B 为 frozen DeepOHeat-v1 Heat3D adaptation workload。两个 workload 均使用 768 train / 128 valid_iid；没有读取 P1i test/sealed 或 DeepOHeat official test。

## 运行时分解

| workload | preparation (s) | graph prep (s) | warm train step (s) | one valid forward (s) | compile count | peak JAX bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: G1 P1i | 338.444 | 327.886 | 83.927 | 160.670 | 1 | 3,483,968,256 |
| B: DeepOHeat-v1 | 316.173 | 261.903 | 76.382 | 157.217 | 1 | 3,825,095,168 |

## 图与 padding

| workload | p2r/r2p real | p2r/r2p padded | p2r/r2p padding | r2r real | r2r padded | r2r padding |
|---|---:|---:|---:|---:|---:|---:|
| A | 1,981,841 | 348,727 | 17.596% | 3,164,326 | 64,106 | 2.026% |
| B | 1,825,305 | 264,327 | 14.481% | 3,174,352 | 59,600 | 1.878% |

## 归因与边界

A/B 的 warm step、单个 valid forward 处于同一量级，B 并未因 DeepOHeat-v1 图而变慢；A 的 preparation/graph preparation 较长，属于 case-specific preparation 差异。当前归因冻结为 `STACK_DOMINANT_HOT_PATH_WITH_CASE_SPECIFIC_PREPARATION_DIFFERENCE`，不据此修改任何 science config，也不作 accuracy 或收敛结论。原始收据：`/tmp/g2_e2_matched_ab.json`，SHA256 `d8a360ce81f61c0ae70ea5826f9f1b822bbb9dca2613b11d71c5fb7b1f9b152a`。

Formal training 仍保持 blocked；native Linux 比较尚未完成，且本 probe 不能替代 profiler gate 或 exact-resume gate。
