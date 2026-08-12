# V6 阶段总结与性能冻结

## 冻结结论

V6 的论文主线冻结为：`P1i formal1024_v1 + V6_06/07/08 三 seed + E16384 reconstruction`。其中 seed0 的 point-global-best e559 checkpoint 是性能与架构研究的 reference，seed1/2 只作重复性验证。本报告完成后停止使用 frozen valid32 继续选择或优化架构；后续只能按新的独立预注册协议开展确认。

U-direct240825 被保留为 **architecture-freeze candidate**：它证明固定 1024 输入编码与区域处理可以直接解码到 240825 个 solver nodes，且在同一输出域严格优于 E240825-direct；但它没有取代 E16384 主线，因为 E16384 的 valid32 PG 更低、三 seed 生产证据更完整。test 与 sealed IID 始终关闭。

## 阶段演进与关键决策

1. P1h 建立共享 1024 support 与 full-field sidecar；P1i 将输入物理量扩展到连续宽覆盖，同时保持 perfect interface contact，`R_contact=0`。
2. V6_03/V6_04 建立层状数据上的 canonical 与 DualAttention 对照；P1i V6_06/07/08 将冻结的 V5-best 架构迁移到 sample-varying 1024 support。
3. R0 确认三 seed checkpoint 在冻结 1024 输入上的严格 CPU replay；GPU 历史 replay 在冻结数值容差内。
4. B 路线随查询点数增加 regional nodes；E 路线固定 `Nr=256`，形成压缩且更稳定的高分辨率区域表示。valid32 resolution sweep 在冻结 0.1 percentage-point non-inferiority + latency 规则下选择 E16384。
5. U 路线使用不对称查询：`x_in=1024`、encoder/processor 仍在原训练尺度上运行，仅将 decoder/R2P 输出扩展至 N。U5 进一步用 `lean_output_query_v2` 直接构造 split-decoder 必需张量，不再先构造完整 supervised output group 再删除字段。
6. P9 以完整 payload hash、persistent worker、untimed warmup 和三组随机样本顺序冻结 E16384 吞吐；U5 在同一 Python/JAX session 重测 E16384、E240825-direct 与 U-direct240825。

## 数据、checkpoint 与 validation 合同

- Dataset：`heat3d_v6_p1i_continuous_physics1024_v1`，正式 manifest SHA256 `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514`；240825-node full-field sidecar SHA256 `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb`。
- Primary checkpoint：V6_06 seed0 point-global best e559，SHA256 `51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e`。V6_07/V6_08 分别是 seed1/2 replication。
- 正式模型质量：128 个 `valid_iid` × 3 seeds。性能和架构开发：预注册、与误差无关的 frozen valid32；它不能替代正式 valid128 质量结论。
- 仅 train 输入拟合 normalization/context。test、sealed IID 未访问；没有训练、checkpoint 修改或数据修改。
- 适用范围：P1i layered、perfect-contact、dual-Robin family。对接触热阻变化、未见结构族或未覆盖的材料/边界分布，不主张保证。

三 seed primary checkpoint 的 support PG 为 `2.027348 ± 0.094738%`，support sample-first 为 `1.629402 ± 0.013132%`；使用冻结 1024 source-aware support 再重建至 240825 节点时，full-field PG 为 `3.442626 ± 0.058435%`。这些是 valid128 正式精度，不与 valid32 architecture rows 混合统计。

## 指标与物理 reference

- **PG true-RMS relative RMSE**：所有点/样本的总平方误差开根后，以对应真值总均方根归一化。
- **sample-first CV relative RMSE**：先按样本和 control volume 计算相对场误差，再对样本汇总；它不会被高能量样本按总 SSE 自动加权。
- **raw CV-weighted RMSE K**：使用物理 control volume 加权的绝对温差误差，单位 K。
- **source/background RMSE**：分别在非零热源区和背景区计算 CV-weighted 误差。
- **peak RMSE**：逐样本峰值温升误差的均方根。
- **interface RMSE**：层间温降/界面相关温度量的误差；数据集本身仍是 perfect contact，并不等价于学会非零接触热阻。

FVM 是 240825-node reference solution。下表中的 surrogate error 均是相对同一 FVM full-field 标签的误差；FVM 自身表中的约 `0.06965% / 0.03655 K` 来自合法结构化网格相对冻结 reference 的离散化误差，不应与“神经网络真值”混淆。FVM 在物理一致性、守恒可解释性和域外可靠性方面优于 surrogate。

## 历史方案精度全景（valid32）

| route | 输出方式 | N / Nr | PG % | raw K | source K | peak K | interface K | 证据口径 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| native1024 + reconstruction | 1024 推理后重建 | 1024 / 256 | 3.072016 | 2.930207 | 2.962902 | 2.675661 | 1.416680 | P5-R 同次 accuracy/runtime |
| B4096 + reconstruction | adaptive B | 4096 / 512 | 2.795165 | 2.612941 | 3.519986 | 3.688481 | 0.645706 | P5-R |
| B8192 + reconstruction | adaptive B | 8192 / 1024 | 2.739829 | 2.458792 | 3.587505 | 3.479191 | 0.510097 | P5-R |
| E16384 + reconstruction | fixed regional mesh | 16384 / 256 | 2.702279 | 2.284809 | 3.872085 | 4.018083 | 0.385738 | U5 当前代码同 session |
| E32768 + reconstruction | fixed regional mesh | 32768 / 256 | 2.672444 | 2.208698 | 3.903574 | 4.051111 | 0.384492 | P5-R |
| E240825 direct | 全场同构查询 | 240825 / 256 | 3.067116 | 2.488403 | 4.734638 | 7.206492 | 0.474357 | U5 当前代码同 session |
| U-direct240825 | 1024 编码/处理，全场解码 | 240825 / 256 | 2.818822 | 2.318094 | 3.855983 | 3.621800 | 0.436341 | U5 lean 当前代码同 session |
| FVM240825 | 物理 reference | 240825 | 0.069648 | 0.036553 | 0.295557 | 0.509663 | 0.000226 | legal structured-FVM mesh sensitivity |

这些行记录真实运行值，但历史 P5-R 与当前 U5 的 latency protocol 不同；因此上表只比较精度和架构含义，不把旧 latency 与新 latency 拼接成 Pareto。

## 冻结性能协议与结果

所有正式性能结果来自 WSL2 同机：RTX 5070 GPU、Ryzen 7 9700X CPU。GPU span 均在 `block_until_ready` 后截止。hash/equivalence、metrics、oracle 和 serialization 不进入 production timing。

术语严格冻结为：

1. **Fresh single-case latency**：一个新工况从 CPU support/CV、graph、map、group/H2D 到 forward 和 240825 重建的连续单样本耗时；静态程序/模型加载和 JIT warmup 在协议规定的位置独立处理。
2. **Warm/resident latency**：prepared group、device tensors、model 和 compiled executable 已常驻时的推理/重建耗时。
3. **Marginal added-case latency**：warm/persistent pipeline 从 B16 扩展至 B32 时每增加一个 case 的时间增量；不是 B1 延迟。
4. **Batch throughput**：完整 batch wall-time 的 `samples/s`，并报告平均/边际 per-case。
5. **Full fresh multi-case throughput**：32 个不同真实 `k/q/BC` 的 CPU preprocessing + GPU batch inference，不能用重复同一输入冒充。

### P9 E16384 与 persistent FVM

| workload | median | p95 | throughput | 相对 saturated FVM |
|---|---:|---:|---:|---:|
| E16384 fresh B1 | 0.851959 s | 1.806525 s | 1.174 samples/s | 1.359× per-case lower-bound |
| E16384 warm resident B1 | 0.013642 s | 0.013841 s | 73.300 samples/s | 仅 inference，不称 E2E speedup |
| E16384 2×B16，完整 valid32 | 12.192828 s | 14.992299 s | 2.624 samples/s | 3.041× |
| E16384 B32，完整 valid32 | 10.907152 s | 11.716352 s | 2.934 samples/s | 3.399× |
| E16384 marginal added case | 0.323254 s | 0.353504 s | — | — |
| persistent FVM P2，完整 valid32 | 37.073368 s | 37.415207 s | 0.863 samples/s | 1.000× |

P9 的三组 randomized order 都完整覆盖 32 个不同工况；Neural 与 FVM 都在 persistent worker 初始化后进行 untimed warmup。完整 anchor/query groups、inputs、graph、native physics、global/scale context、QK features、selected CV 和 reconstruction map 的 hash 均与 serial reference 一致。process8 虽有最高孤立 preprocessing throughput，但与 GPU runtime 并存时出现 RAM/swap 压力，因此生产 backend 冻结为 exact 且 RAM-resident 的 process4。

### U5 当前代码同 session Pareto

| route | fresh median / p95 | PG % | raw K | VRAM GiB | 结论 |
|---|---:|---:|---:|---:|---|
| E16384-reconstruction | 3.040339 / 3.650992 s | 2.702279 | 2.284809 | 0.280 | 主线 accuracy/production route |
| E240825-direct | 1.087986 / 1.125314 s | 3.067116 | 2.488403 | 3.810 | 被 U-direct 同输出域支配 |
| U-direct240825 | 0.658215 / 0.692390 s | 2.818822 | 2.318094 | 5.679 | direct architecture candidate |

U-direct 相对 E240825-direct：PG 改善约 `0.248294 percentage points`，fresh median 加速约 `1.653×`，同时 raw/source/peak/interface 均改善。E16384 仍比 U-direct 低约 `0.116543 pp` PG、`0.033286 K` raw，但 U-direct 的延迟约为 E16384 的 `0.216×`，且无需 240825-node reconstruction map。两者构成 accuracy–latency–VRAM Pareto，不能用单一轴宣称全面胜出。

## 可发表优势与边界

在冻结适用域内，V6/P1i 展示的是 **controlled-error surrogate acceleration**：

- 对大量新物理工况，GPU batch 让完整 valid32 吞吐达到 persistent CPU FVM 的约 `3.40×`；
- warm/resident B1 约 `13.6 ms`，新增 case 的边际成本约 `323 ms`，适合 large-scale parametric sweep、design-space exploration 和批量优化；
- 同一 checkpoint 支持 1024、B/E 多种查询分辨率，并能以 U 路线执行 `1024 training-scale encoding → 240825 full-resolution decoding`；
- U-direct 避免为每个新工况重新求解完整物理线性系统，也避免 reconstruction apply，但仍保留冻结的 1024 anchor-derived context/scale；
- GPU 并行批量求解的优势主要体现在 throughput 和低边际成本，而不是所有单样本 fresh latency。

这些结论不表示 surrogate 比 FVM 更精确。FVM 是 reference solution，守恒和域外物理一致性更强；surrogate 有约 2.7%–3.1% valid32 full-field PG，source/peak 误差高于 overall PG，并受训练分布、perfect-contact、support/graph 语义和 checkpoint 固定容量约束。不同 workload 的 speedup 只有在 timing boundary 匹配时才可比较；warm neural core、known-case FVM lower bound 与 fresh E2E 不互换。

## Architecture freeze recommendation

冻结双层结论：

- **Production/reference architecture：E16384-reconstruction。** 它在预注册 valid32 规则下被选中，保持最低的可接受 full-field error、低 VRAM，并已有 P9 persistent/batch 公平基准。
- **Architecture-freeze candidate：U-direct240825。** 它在同 240825 输出域严格支配 E240825-direct，证明 fixed-resolution encoding → full-resolution decoding 的可行性；在新的独立 population 确认前，不替代 E16384。
- **停止 valid32 架构调优。** 不再根据 valid32 搜索 route、factor、radius、regional count、packing 或 batch；若继续，应先冻结独立确认集/门槛，且 test/sealed 仍不能用于调参。

机器可读证据：`v6_p1i_p9_performance_freeze_closeout.json`、`v6_p1i_u5_direct_timing_freeze_closeout.json` 及其 runtime raw artifacts。历史 route 精度来自 `v6_p1i_p5r_resolution_sweep_closeout.json`，三 seed 正式质量来自 `v6_p1i_three_seed_inference_closeout.json`。
