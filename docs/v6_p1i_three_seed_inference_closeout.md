# V6 P1i three-seed formal closeout and inference benchmark

本报告仅使用 train 输入拟合冻结标准化，并评价 `valid_iid`；test 与 sealed IID 均未打开。训练协议、checkpoint 和模型参数未修改，也未重训或调参。

## Three-seed primary checkpoint

| seed | epoch | support point-global % | support sample-first % | support raw CV K | full-field point-global % | full-field raw CV K | peak/source/background K | layer/interface/top/bottom K |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 559 | 2.109943 | 1.640709 | 1.316123 | 3.459884 | 2.985701 | 4.150/2.910/2.986 | 1.464/1.483/0.788/1.140 |
| 1 | 455 | 2.048171 | 1.632499 | 1.259137 | 3.490488 | 2.970243 | 3.781/2.915/2.970 | 1.452/1.487/0.798/0.998 |
| 2 | 587 | 1.923931 | 1.614998 | 1.223539 | 3.377505 | 2.947569 | 3.393/2.686/2.948 | 1.368/1.403/0.729/0.958 |

三 seed mean±std：support point-global 2.027348±0.094738%，sample-first 1.629402±0.013132%，full-field point-global 3.442626±0.058435%。

## Checkpoint reliability and late-epoch behavior

- 三个独立 Python 进程均成功加载 best/sample-first/base/final/latest；参数归档 schema 为 optimizer-aware v2。
- 跨进程重放 RMSE 均低于 0.01 K；GPU scatter 的极少数单点差异完整保留在 machine-readable audit 中。
- support point-SSE best→final 分别变化 seed0 4.117%, seed1 3.345%, seed2 0.838%；seed1 的 sample-first 与 full-field final 略优，但不能替代预注册 point-global primary。
- primary sample-relative tail（p95/max）分别为 seed0 3.625%/6.567%, seed1 3.632%/6.511%, seed2 3.436%/5.172%；top-10 point-SSE 占比分别为 seed0 43.63%, seed1 42.77%, seed2 37.35%。

## Resolution–accuracy–latency

固定 `devbox` SSH alias（系统报告 hostname `XYH-Desktop`、WSL2）、RTX 5070、B1、同一首个 valid 样本；每个稳态阶段预热一次后重复 20 次。FVM 与直接模型使用同节点数结构化网格；source 以 control-volume overlap 守恒投影。

| nodes | direct PG % | direct graph/JIT/steady-e2e s | 1024+recon PG % (oracle floor) | 1024+recon core/e2e s | FVM PG % | FVM assembly/solve/e2e s | FVM/B core / e2e |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 6477.7187 | 1.684/2.828/0.3415 | 1.3286 (0.6824) | 0.3299/0.3540 | 0.7141 | 0.0002/0.0030/0.0032 | 0.010×/0.009× |
| 4096 | 450.6820 | 3.834/6.921/0.3392 | 1.3387 (0.6943) | 0.3270/0.3516 | 0.6861 | 0.0003/0.0064/0.0068 | 0.021×/0.019× |
| 16384 | 53.0814 | 4.483/8.321/0.3315 | 1.3439 (0.7047) | 0.3275/0.3526 | 0.6829 | 0.0013/0.0408/0.0420 | 0.128×/0.119× |
| 65536 | 3.3059 | 5.347/8.738/0.3281 | 1.3460 (0.7087) | 0.3267/0.3576 | 0.6827 | 0.0050/0.2340/0.2390 | 0.731×/0.668× |
| 240825 | 255.7418 | 9.357/9.081/0.4461 | 1.8568 (1.3236) | 0.3342/0.3856 | 0.0442 | 0.0230/1.8517/1.8756 | 5.613×/4.864× |

## Applicability

- 当前 checkpoint 的可信默认路径仍是 P1i 原生 source-aware 1024 support 后进行 layer-aware reconstruction；它在所有目标分辨率上最稳定。
- 直接结构化高分辨率不具稳定兼容性：误差随分辨率非单调，65536 的偶然恢复没有在 240825 延续，因此所有 A 路线结果都只作 compatibility diagnostic，不构成生产适用区间。
- 基准精度与耗时使用一个预注册 valid 样本，适合工程兼容性与阶段计时，不替代 128-sample 三 seed 质量统计。
- FVM 和模型同节点数，但计算图/数值方法不同；speedup 仅为同硬件单样本墙钟比较。
