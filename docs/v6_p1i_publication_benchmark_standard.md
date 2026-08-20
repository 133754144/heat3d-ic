# V6 publication benchmark standard v1

本文件冻结正式计时方法，不发布新的 latency 或 speedup。正式测量只能在
`benchmark_standard_freeze=GO` 后执行；当前 publication timing 状态必须保持
`NO_GO_pending_full_measurement`。

## 统一 workload

五条路线为 E16384+reconstruction、U-v2 16384+reconstruction、U-v2
direct240825、E240825 direct control 与 FVM240825。它们均以 RAM 中的
`k/q/BC` 为起点，以同步完成的 240825-node result 为终点。I/O、accuracy、
qualification、hash、equivalence 与 serialization 均在 service span 外。

正式表必须分别报告：cold-service first case、fresh distinct-case median/p95、
repeat-case cache-hot、resident core、真实 B16→B32 marginal、Q1、真实 Q2 的
submit→result/inter-completion/samples/s，以及 CPU workers/processes、VRAM 和
RAM。resident neural core 与 FVM prepared-system solve-only 都不得称为 E2E。

## 生命周期与公平性

每个 `route × randomized_order` 使用全新独立 Python process，固定 seeds 为
`20260814/20260815/20260816`。JIT 只能使用 train split 中不属于 timed
population 的专用 input；不读该 case 的 target，不预热任何 timed case、
graph shape 或 packing shape。E/U 统一使用单 CPU/KDTree/graph/reconstruction
worker，Q2 为两个 service workers；FVM 为两个 persistent processes、每进程
一线程。

Q2 不得先串行遍历同一批 case。其 arrival rule 是先提交两个不同工况，之后每
完成一个再补一个。原 residual gate 保持
`0 <= residual <= max(0.025 s, 0.05×E2E)`，不得放宽。cold、fresh 与 cache-hot
必须分别建表并记录 cache provenance，禁止混池。

## Exactness 与缓存边界

所有 runtime 实现必须在 1024/16384/240825 覆盖 graph/metadata/edge hash、
prepared payload 与 prediction equivalence。CPU deterministic comparison 要求
bitwise exact；GPU 要报告冻结 tolerance 下的 max/RMSE。checkpoint 不得改变。

允许进入 service startup 的只有 import/runtime、CUDA context、checkpoint、
不可变 mesh/layer partition、normalization，以及专用非 population case 的模型
kernel compile/cache load。support/CV、sample-varying graph、dynamic context、
`TypedGraph` group packing、H2D、forward 与 reconstruction apply 属于 fresh
case-specific 成本。static graph、static structural packing 和 reconstruction map
只能计入明确标为 known-support cache-hot 的 workload。

静态定位：`_prepare_group` 位于 high-N executor；`build_graphs` 位于 Heat3D
graph builder/RIGNO；`TypedGraph` 由 RIGNO graph construction 创建。首次 shape
或 cache miss 必须留在对应 cold/fresh pool，不能用 route-specific prewarm 隐藏。

## Low-cost smoke

smoke 仅使用四个 frozen-valid32 input token：cold、fresh、cache-hot repeat，以及
两个未串行预访问的 Q2 cases；warmup 使用独立 train input。五路线×三顺序共
15 个独立进程，只验证生命周期、资源、公平性、分类、真实并发、residual gate
和已有 exactness evidence。smoke 内的微型 timing 不具 publication 资格，不生成
speedup，也不代替后续完整 measurement。

机器可读标准：
`configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_standard_v1.json`。
smoke 原始记录：
`configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_standard_smoke.json`。

角色合同：不训练、不访问 test/sealed、不调 accuracy、不运行完整 valid32/valid96
矩阵，不改 checkpoint/dataset/graph policy。

## Smoke closeout

冻结协议提交后在 WSL2/RTX 5070 上执行了 15 个独立 Python processes（五路线
×三个固定 seeds），每个进程只使用四个 valid32 input tokens。15 个 PID 全部
唯一；专用 warmup case `v6p1if1_0389` 来自 train split、未读取 target，且其
shape 与 timed shapes 不同。E/U CPU policy 完全一致；所有 Q2 都由两个真实
workers 并发执行、没有 Q2 population 的 serial prepass，最小重叠时间大于零。
原 residual hard gate 在 cold/fresh/cache-hot 三类均可执行并通过。

1024/16384/240825 的 graph/payload/prediction exactness evidence 均绑定到已有冻结
artifact；smoke 不重跑 accuracy。原始 smoke SHA256 为
`83b91811378d47b13d73e863ed3126bb46970a1a532e2f4125951bd377599ae3`。

结论：`benchmark_standard_freeze = GO`；
`publication_timing_freeze = NO_GO_pending_full_measurement`。smoke 内部微型耗时
不可引用为正式 latency，也未计算 speedup。
