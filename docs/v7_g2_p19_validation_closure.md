# V7 G2-P19：validation closure and inference profiling

## Scope and formal-view policy

本轮只读取冻结 `valid_iid` 128 个 physical cases 的输入和已冻结 checkpoint，
不训练、不读取 target/truth、不访问 `test_iid`、sealed 或 DeepOHeat official100。
DeepOHeat benchmark 的正式 Heat3D 视图只保留 **fresh e600 × 3 + U-v2
direct-query dense inference**。P14.1 e200、IDW 和相关 oracle 仅保留为
historical diagnostic，不进入 P18 正式比较表或主结论。

P18 checkpoint policy 已冻结：Heat3D 使用固定 e600 endpoint；DeepOHeat
full-minus-valid128 与 matched-768 均同时报告 validation-selected best 和
fixed final-100000 endpoint；不使用模糊的 best-to-best 主比较。

## IDW provenance

历史 IDW 来自 Heat3D V6/P1h full-field utility，而非 DeepOHeat-v1：
`rigno/heat3d_v6_full_field.py` at Heat3D commit
`f562a4d4f8b4181d234fa8c32906d341565acbb7`，`build_reconstruction_map` 使用
layer/interface-aware `cKDTree` KNN 和 inverse-distance-squared weights；P14.1
调用位置为 `scripts/evaluate_v7_g2_p14_1_heat3d_u_v2.py:386-395`。完整文件、
blob、实现和上游对照证据见
[`v7_g2_p19_idw_provenance.json`](v7_g2_p19_idw_provenance.json)。

上游 DeepOHeat-v1 在 pinned commit `3ef3d9c41666a56b5940b39a61166ccaa5aaedb2`
中由 `heat_volumetric.py:24-29` 直接调用模型、`models.py:292-309` 通过
trunk/branch `einsum` 产生 dense field，并在 `heat_volumetric.py:103-109`
定义 101×101×56 网格；该路径没有 IDW。

## Valid-only P19 profile

Raw receipt（仅保留在 devbox `/tmp`）SHA256 为
`baea3fa13f51529bcce36f6594e7632d18c44e62b47654c82704e5cacbcb6594`；精简
aggregation 为
[`v7_g2_p19_inference_profile_aggregation.json`](v7_g2_p19_inference_profile_aggregation.json)
（SHA256 `da2987b54a919bafde294442644dd4ee4c85304603296f5a7631f5778af825c4`）。
执行代码 provenance：repo commit `f9cd8b59512b42b609bd28fb88b8cd4fb0111e73`，
`scripts/profile_v7_g2_p19_inference.py` SHA256
`eee7d8502c999b54bdbb74992ef0721ccac03a3646334cf8a9040de4cf640d6b`，P19 protocol
SHA256 `4bc666c2e42e35cf54422986772fe4845113cc7fb37a3ad67ff251047cc87851`。
后续提交只新增 aggregation/documentation，不改变该执行 receipt。

计时在同一 devbox RTX 5070（WSL2，driver 591.86，JAX/jaxlib 0.9.1，Python
3.14.3，`XLA_PYTHON_CLIENT_PREALLOCATE=false`，deterministic=false）进行；
每个 device phase 显式 `jax.block_until_ready`，首个 valid pass 包含冷启动/编译，
steady pass 排除首个 case，cached fixture 重复 3 次。
整个单进程 profile（含 provenance/preparation 与两模型 valid-only phases）墙钟
`1173.911 s`；该值不应被解释为单次模型 latency。

| checkpoint | cold E2E (s) | steady E2E median / p95 (s) | graph median / p95 (s) | model median / p95 (s) | cached E2E median (s) | peak live GiB |
|---|---:|---:|---:|---:|---:|---:|
| Heat3D e600 fixed endpoint | 18.386050 | 8.873593 / 10.010835 | 3.922524 / 4.658105 | 4.999001 / 5.583132 | 0.473031 | 0.461 |
| DeepOHeat full-minus-valid128 best | 3.096993 | — | — | 0.001014 / 0.002004 (model-only) | 0.004822 | 0.035 |
| DeepOHeat full-minus-valid128 final | 1.467455 | — | — | 0.001835 / 0.002034 (model-only) | 0.004656 | 0.035 |

表中显存为各模型 timed rows 的 live peak；JAX 同一进程的累计 allocator peak
`5.194 GiB` 另保留在 JSON，不能归因给 DeepOHeat 单模型。DeepOHeat 的 steady
E2E median 分别为 best `0.004915 s`、final `0.005517 s`
（batch=4，128 samples）；Heat3D 为 batch=1、每 case 571,256 queries。该
计时是 inference-only engineering evidence，不替代 P18 accuracy，也不在
不同 batch/query boundary 下作未经重新定义的 Pareto 排名。Heat3D 的 steady
E2E 约等于 graph construction + model forward，postprocess 中位数约
`0.002030 s`；cold pass 的额外成本由首轮 JAX executable 初始化体现。

所有 profile 输出 finite，三类 checkpoint 均成功加载，`weights_modified=false`；
source/manifest/normalization SHA 与 P19 protocol 一致，`target_truth_loaded=false`。

## Conservative convergence wording

P15 原先的 `CONVERGED_WITHIN_600` 标签已被保守解释取代：
`NO_CLEAR_BOUNDARY_RIGHT_CENSORING_STABLE_PLATEAU_UNPROVEN`。预注册的 native
和 direct-query dense 观察没有显示明确的 boundary right-censoring，但不证明
稳定 plateau；不授权延长 e600。P18 正式表删除 e200/IDW formal rows。

## Open gates / P20 prerequisites

已关闭：P14.1 valid-only evaluator、P15 e600×3、P17
full-minus-valid128×3、P18 common-valid table、P19 checkpoint/provenance
profile。仍开放：任何 test/sealed unlock、同硬件统一 latency protocol 的审计
意见、P20 evaluation-only unlock protocol 和最终 publication wording review。
在 P20 前不得打开官方 100 test、P1i test_iid 或 sealed，也不得启动
Therm-FM、sparse-1024 DeepOHeat、learned decoder 或 multi-HTC。
