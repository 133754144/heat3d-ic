# V6 inference benchmark qualification closeout

本轮冻结三 seed valid 结果：point-global 为 primary、sample-first 为 secondary；可信部署路线是 1024 source-aware support + layer-aware reconstruction。test/sealed 未访问，未训练或调参。

Fixed host: `DESKTOP-2GE35DV`; CPU `AMD Ryzen 7 9700X 8-Core Processor` (16 logical CPUs, 24168084 kB); model device `NVIDIA GeForce RTX 5070`; Python 3.14.3, JAX/JAXlib 0.9.1/0.9.1, NumPy/SciPy 2.4.2/1.17.1.

## Corrected timing

所有结果来自同一 WSL2 主机、32 个固定 valid 样本、B1/单线程。cold 为每样本新进程；model new-case 为 JIT 已建立但案例/图/映射未缓存，FVM 对应为新案例且未缓存系统；cached 为 model 图/JIT/重建映射或 FVM 组装系统已缓存。每项均为连续 wall-clock，阶段和不作为总时间相加。production 区间不含 oracle 或指标计算。

| family | route | state | median / mean / std / p95 s | peak RAM GiB | peak device GiB | GPU→CPU FVM/route speedup |
|---|---|---|---:|---:|---:|---:|
| p1i | fvm | cold | 3.0853 / 3.1791 / 0.3604 / 3.4409 | 0.386 | 0.000 | — |
| p1i | fvm | jit_cached_new_case | 1.6442 / 1.6720 / 0.1262 / 1.9169 | 0.580 | 0.000 | — |
| p1i | fvm | fully_cached_repeat | 1.6065 / 1.6119 / 0.1281 / 1.8049 | 1.255 | 0.000 | — |
| p1i | model_support | cold | 10.8051 / 10.8108 / 0.2919 / 11.0704 | 1.730 | 0.178 | — |
| p1i | model_support | jit_cached_new_case | 1.5626 / 1.4576 / 0.2968 / 1.7559 | 3.445 | 0.130 | — |
| p1i | model_support | fully_cached_repeat | 0.0023 / 0.0026 / 0.0005 / 0.0034 | 3.417 | 0.130 | — |
| p1i | production_reconstruction | cold | 10.8540 / 10.8746 / 0.1795 / 11.1213 | 1.803 | 0.130 | 0.284× |
| p1i | production_reconstruction | jit_cached_new_case | 1.7771 / 1.6089 / 0.2863 / 1.9087 | 3.744 | 0.130 | 0.925× |
| p1i | production_reconstruction | fully_cached_repeat | 0.0102 / 0.0109 / 0.0022 / 0.0125 | 4.443 | 0.130 | 158.183× |
| randomblock | fvm | cold | 3.5611 / 3.5440 / 0.0610 / 3.6042 | 0.418 | 0.000 | — |
| randomblock | fvm | jit_cached_new_case | 1.3184 / 1.3088 / 0.0365 / 1.3447 | 0.598 | 0.000 | — |
| randomblock | fvm | fully_cached_repeat | 1.2775 / 1.2670 / 0.0368 / 1.3025 | 1.293 | 0.000 | — |
| randomblock | model_support | cold | 11.2340 / 11.2136 / 0.2422 / 11.5924 | 1.734 | 0.130 | — |
| randomblock | model_support | jit_cached_new_case | 0.0843 / 0.0798 / 0.0174 / 0.0903 | 3.320 | 0.130 | — |
| randomblock | model_support | fully_cached_repeat | 0.0030 / 0.0033 / 0.0011 / 0.0054 | 3.316 | 0.130 | — |
| randomblock | production_reconstruction | cold | 11.3778 / 11.3821 / 0.1649 / 11.6257 | 1.802 | 0.178 | 0.313× |
| randomblock | production_reconstruction | jit_cached_new_case | 0.2231 / 0.2182 / 0.0141 / 0.2320 | 3.616 | 0.130 | 5.909× |
| randomblock | production_reconstruction | fully_cached_repeat | 0.0108 / 0.0111 / 0.0018 / 0.0119 | 3.951 | 0.130 | 118.617× |

## Stage decomposition (median seconds)

| family | route/state | data | graph | JIT/forward | map build/apply | output | FVM assembly/solve |
|---|---|---:|---:|---:|---:|---:|---:|
| p1i | fvm/cold | 0.0251 | 0.0000 | 0.0000 | 0.0000/0.0000 | 0.0003 | 0.0289/1.6062 |
| p1i | fvm/jit_cached_new_case | 0.0142 | 0.0000 | 0.0000 | 0.0000/0.0000 | 0.0003 | 0.0284/1.6019 |
| p1i | fvm/fully_cached_repeat | 0.0000 | 0.0000 | 0.0000 | 0.0000/0.0000 | 0.0003 | 0.0000/1.6063 |
| p1i | model_support/cold | 0.0006 | 4.7494 | 3.8409 | 0.0000/0.0000 | 0.0001 | 0.0000/0.0000 |
| p1i | model_support/jit_cached_new_case | 0.0005 | 1.5589 | 0.0029 | 0.0000/0.0000 | 0.0001 | 0.0000/0.0000 |
| p1i | model_support/fully_cached_repeat | 0.0004 | 0.0000 | 0.0018 | 0.0000/0.0000 | 0.0000 | 0.0000/0.0000 |
| p1i | production_reconstruction/cold | 0.0006 | 4.6758 | 3.8394 | 0.1403/0.0179 | 0.0003 | 0.0000/0.0000 |
| p1i | production_reconstruction/jit_cached_new_case | 0.0008 | 1.6263 | 0.0029 | 0.1292/0.0127 | 0.0003 | 0.0000/0.0000 |
| p1i | production_reconstruction/fully_cached_repeat | 0.0008 | 0.0000 | 0.0031 | 0.0000/0.0060 | 0.0002 | 0.0000/0.0000 |
| randomblock | fvm/cold | 0.0064 | 0.0000 | 0.0000 | 0.0000/0.0000 | 0.0003 | 0.0422/1.2679 |
| randomblock | fvm/jit_cached_new_case | 0.0031 | 0.0000 | 0.0000 | 0.0000/0.0000 | 0.0003 | 0.0309/1.2839 |
| randomblock | fvm/fully_cached_repeat | 0.0000 | 0.0000 | 0.0000 | 0.0000/0.0000 | 0.0003 | 0.0000/1.2771 |
| randomblock | model_support/cold | 0.0005 | 4.4509 | 3.8816 | 0.0000/0.0000 | 0.0001 | 0.0000/0.0000 |
| randomblock | model_support/jit_cached_new_case | 0.0004 | 0.0810 | 0.0024 | 0.0000/0.0000 | 0.0001 | 0.0000/0.0000 |
| randomblock | model_support/fully_cached_repeat | 0.0004 | 0.0000 | 0.0025 | 0.0000/0.0000 | 0.0000 | 0.0000/0.0000 |
| randomblock | production_reconstruction/cold | 0.0006 | 4.4595 | 3.8337 | 0.1449/0.0167 | 0.0003 | 0.0000/0.0000 |
| randomblock | production_reconstruction/jit_cached_new_case | 0.0007 | 0.0862 | 0.0028 | 0.1257/0.0066 | 0.0002 | 0.0000/0.0000 |
| randomblock | production_reconstruction/fully_cached_repeat | 0.0007 | 0.0000 | 0.0034 | 0.0000/0.0065 | 0.0002 | 0.0000/0.0000 |

## Accuracy

| family | route/domain | point-global % | sample-first % | raw CV K | peak/source/background K |
|---|---|---:|---:|---:|---:|
| p1i | model_support/support_1024 | 1.7545 | 1.4104 | 1.2949 | 2.637/2.691/1.240 |
| p1i | production_reconstruction/support_1024 | 1.7545 | 1.4105 | 1.2950 | 2.637/2.691/1.240 |
| p1i | production_reconstruction/full_240825 | 3.0720 | 3.7932 | 2.9302 | 2.676/2.963/2.930 |
| p1i | oracle_reconstruction/full_240825 | 2.4899 | 3.4308 | 2.6290 | 0.387/1.160/2.633 |
| p1i | dataset_consistent_fvm/full_240825 | 0.0696 | 0.0458 | 0.0366 | 0.510/0.296/0.032 |
| randomblock | model_support/support_1024 | 112.6276 | 125.1684 | 41.5474 | 51.179/40.087/41.610 |
| randomblock | production_reconstruction/support_1024 | 112.6278 | 125.1686 | 41.5474 | 51.180/40.088/41.610 |
| randomblock | production_reconstruction/full_240825 | 108.3401 | 118.2579 | 42.1189 | 51.132/39.528/42.124 |
| randomblock | oracle_reconstruction/full_240825 | 4.2152 | 7.2413 | 2.5503 | 0.059/0.255/2.553 |
| randomblock | dataset_consistent_fvm/full_240825 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000/0.000 |

P1i 1024+reconstruction full-field point-global=3.0720%，通过 <20% 资格门；random-block OOD diagnostic=108.3401%，不具生产兼容性。

## Historical layer audit

历史数字均可在各自归档协议内复现，但没有一组可直接与本轮资格计时合并：旧 P1i 只测 1 个样本且 cold 为分段派生；layer 基准使用不同数据集、batch、主机边界、重复数或 nonmatched-DOF FVM。原 4.86× 只称 cached steady-state speedup；Route A 只称 structured-support OOD compatibility diagnostic。

## Sample complexity and solver iterations

- p1i: 32 valid samples, unique support hashes=32, CG iterations median/P95=1528.5/1764.0; source-region median=7.0, conductivity-region median=6.0.
- randomblock: 32 valid samples, unique support hashes=16, CG iterations median/P95=1190.0/1210.0; source-region median=5.0, conductivity-region median=4.5.

| family | support/FVM nodes | support hashes | source/k regions median | CG iterations median/P95 | new-case graph s | cached FVM solve s |
|---|---:|---:|---:|---:|---:|---:|
| p1i | 1024/240825 | 32 | 7.0/6.0 | 1528.5/1764.0 | 1.5589 | 1.6063 |
| randomblock | 1024/240825 | 16 | 5.0/4.5 | 1190.0/1210.0 | 0.0810 | 1.2771 |

## JIT shape qualification

- P1i uses fixed dummy-edge padding. Frozen equivalence: max=0.007294 K, RMSE=0.001838 K (limits 0.01/0.002 K).
- random-block fixed padding was rejected (max=0.022064 K, RMSE=0.006103 K). Formal timing keeps raw graphs and warms the 16 preregistered support-shape families with a third valid variant before measuring two new variants per group.

## Rejected attempts

P1i 无完整 forward JIT 的初次计时、random-block 全局 edge padding、以及误用 P1i mesh builder 的 random-block FVM 均 fail closed；它们的日志/审计 SHA 保留在 machine-readable closeout，且没有数字进入正式表。

## Qualification

本报告只判定计时和复现资格，不扩大模型适用域。P1i native support 的质量与 1024+layer-aware reconstruction 分开报告；random-block 是冻结 layer checkpoint 的跨结构 OOD 诊断，不用于调参或宣称生产泛化。
