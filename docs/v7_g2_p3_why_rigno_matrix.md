# G2-P3 “Why RIGNO?” backbone 证据矩阵

一手来源：[DeepONet](https://www.nature.com/articles/s42256-021-00302-5)、[FNO/ICLR 2021](https://openreview.net/forum?id=c8P9NQVtmnO)、[GINO/NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)、[Transolver/ICML 2024](https://proceedings.mlr.press/v235/wu24r.html)、[RIGNO](https://arxiv.org/abs/2501.19205)。参数量是本轮冻结 recipe，不是架构的普遍常数。

| backbone | irregular point-native | regular-grid dependency | geometry inductive bias | input/output resolution coupling | sparse-conditioning suitability | computation path | frozen parameter scale | expected query-resolution scaling |
|---|---|---|---|---|---|---|---|---|
| DeepONet | 部分：trunk 可查任意点；branch 通常固定 sensors | 不要求规则 latent grid，但要求固定 branch observation layout | 无内建 geometry graph；需由输入/扩展提供 | input sensors 固定，output query 解耦 | 固定 sparse sensors 可用；variable point set 非原生 | branch coefficients × trunk bases | recipe-dependent | trunk/inner product 近似随 `N_query` 线性 |
| FNO | 否，标准实现先规则栅格化 | **是** | Fourier global convolution；geometry 通常靠 mask/coords | 标准 grid-to-grid；可 zero-shot super-resolution 但仍在规则 grid | 稀疏点需插值/rasterization | FFT spectral layers | recipe-dependent | 每层约 `O(N_grid log N_grid)` |
| GINO | **是：external input/output** | 内部固定 regular latent grid | SDF/point cloud + input/output GNO | input/output point sets 解耦 | 是，但 radius coverage 必须预冻结 | input radius graph → latent FNO → output radius graph | **13,673,988** | query cost 随 output-GNO edges；latent FNO cost固定于32³ |
| Transolver | **是** | 否 | learned Physics-Attention slices | tokens 与输出点同集的官方 PDE recipes；可变点数 | **是，点 tokens 直接接收 fields** | points→learned slices→slice attention→points | **716,737** | paper 给出对点数近线性路径，约随 `N·slices` 增长 |
| RIGNO | **是** | 否 | multi-scale regional mesh + physical↔regional edges | input/output physical meshes 可分离 | **是，区域聚合 sparse observations** | physical→regional graph→multi-scale message passing→query decode | vanilla **826,277**；Heat3D **892,776** | processor 主要随 regional graph；decoder 随 query-edge count |

选择 RIGNO 的方法论理由是它把 irregular observations、multi-scale geometry 和独立 query decode 放在同一 graph operator 中，正好承载 Heat3D 的研究接口；这不是性能结论。GINO 也能通过 irregular↔latent-grid GNO 完成 resolution-decoupled query，Transolver 也能直接处理 point tokens，因此二者是必要 external baselines，而不是陪衬。

## G1 / G2 逻辑边界

- **G1**：在固定 RIGNO family 内回答 Heat3D 新机制相对 Vanilla RIGNO 是否有效。它不能回答 RIGNO 是否优于其他 backbone。
- **G2**：在同一 P1i physical cases、同一 `coords + 11 physical features` 信息预算和预注册 checkpoint/metrics 下比较 Vanilla RIGNO、GINO、Transolver、Heat3D，回答 RIGNO 是否是合理 backbone，以及 Heat3D 实际处于什么性能/成本位置。
- G2 formal 结果出来前不得写“RIGNO 优于 GINO/Transolver”。若 GINO 或 Transolver 更优，论文结论必须如实转为 backbone trade-off 与 Heat3D mechanism 的边界。
