# V6 阶段总结与最终架构/性能冻结

本次 post-freeze 收口继续执行“停止 valid32 架构调优”的治理约束。统一时间层使用固定术语：Fresh single-case latency、Warm/resident latency、Marginal added-case latency、Batch throughput；所有主路线最终输出域均为 240825 solver nodes。

## 最终冻结结论

V6 的论文主线最终冻结为 `P1i formal1024_v1 + V6_06/07/08 三 seed + E16384-reconstruction`。seed0 point-global-best e559 checkpoint 是 reference，seed1/2 是独立重复。`frozen valid32` 的 route/graph/packing/model 调优自本报告起永久关闭。

B、E、U 被统一定义为三种并列的 **inference strategies**：B 使用随 N 增长的区域节点，E 固定 `Nr=256` 后重建至完整场，U 在 native 1024 编码/处理后直接解码到完整场。它们是策略定义，不等于都已取得部署资格。独立 valid96 确认显示 E 可稳定执行并保持三 seed 优势；历史 U-v1 在正式 valid population 中出现 `query outside native domain` 并按预注册硬门 fail-closed。U-v2 只修复该查询域适用性：保持 native1024 图、checkpoint 与图策略完全不变，以无 clamp 的 native-bbox 仿射坐标和冻结 R2P nearest-repair 支持有限 query-domain extrapolation。production/reference 仍冻结 E16384；U-v2 仅为并列 direct inference strategy 的 valid96 diagnostic/characterization，E240825-direct 仅作 architecture control。

全程没有训练、checkpoint/model/dataset/manifest 修改，也没有访问 test 或 sealed IID。FVM 是物理 reference；本文不宣称 surrogate 精度优于 FVM。

## 阶段演进与物理含义

1. P1h 建立共享 1024 support 与 full-field sidecar；P1i 扩展为连续物理输入，同时保持 perfect interface contact，`R_contact=0`。
2. V6_06/07/08 将冻结的 V5-best 架构迁移到 sample-varying P1i support；三 seed 正式 valid128 的 support PG 为 `2.027348 ± 0.094738%`，1024 source-aware support 重建到 240825 节点后的 full-field PG 为 `3.442626 ± 0.058435%`。
3. R0 在 CPU 上严格复现三 seed 的原始 1024 checkpoint 路径；GPU 历史 replay 在冻结数值容差内。
4. B 是 adaptive regional-density 路线；E 固定 `Nr=256`，以压缩区域表示换取更稳定的 high-N 查询；冻结 valid32 规则选择 E16384。
5. U 是不对称查询路线：`x_in=1024`，encoder/processor 保持训练尺度，仅 decoder/R2P 输出扩展到 N。它证明了 `1024 training-scale encoding → 240825 decoding` 的局部可行性，但 formal-valid 几何覆盖失败阻止其成为当前生产路线。
6. Post-freeze 统一了 P9/U5 runner、GPU 同步与 timing boundary，并在 frozen valid96 做了不调参的独立确认。

## 冻结数据、checkpoint 与角色合同

- Dataset：`heat3d_v6_p1i_continuous_physics1024_v1`；formal manifest SHA256 `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514`。
- Full-field sidecar：240825 nodes，SHA256 `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb`。
- Seed0 primary checkpoint：e559，SHA256 `51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e`；seed1/2 使用各自冻结 point-global-best checkpoint。
- `valid32` 已证明是 formal valid128 的冻结子集；`valid96` 在读取结果前唯一确定为 `formal valid128 − valid32`，顺序为冻结的 `SHA256(sample_id)` 顺序。
- normalization/context 仅由 train 输入拟合。test/sealed 未访问；没有模型选择或调参。
- 适用范围是 P1i layered、perfect-contact、dual-Robin family；非零接触热阻、未见结构族或 native-domain 之外的 query geometry 不在保证范围内。

## 指标与 reference

- **PG true-RMS relative RMSE**：所有样本/节点 SSE 开根后，以真值全局 RMS 能量归一化。
- **sample-first CV relative RMSE**：先逐样本计算 CV 加权相对误差，再汇总样本。
- **raw CV-weighted RMSE K**：以 control volume 加权的绝对温升误差。
- **source/background RMSE**：分别在非零热源区和背景区统计。
- **peak RMSE**：逐样本峰值温升误差的 RMSE。
- **interface RMSE**：层间温降相关误差；不代表学习了非零接触热阻。

FVM240825 生成冻结 full-field 标签，是 reference solution。surrogate 的所有误差均相对该 reference；FVM 在守恒、物理一致性和域外可靠性上仍有根本优势。

## 历史方案精度全景（valid32，仅精度）

| strategy | 输出方式 | N / Nr | PG % | raw K | source K | peak K | interface K | provenance |
|---|---|---:|---:|---:|---:|---:|---:|---|
| native1024 + reconstruction | 1024 推理后重建 | 1024 / 256 | 3.072016 | 2.930207 | 2.962902 | 2.675661 | 1.416680 | P5-R |
| B8192 + reconstruction | adaptive B | 8192 / 1024 | 2.739829 | 2.458792 | 3.587505 | 3.479191 | 0.510097 | P5-R |
| E16384 + reconstruction | fixed Nr | 16384 / 256 | 2.702277 | 2.284802 | 3.872139 | 4.018064 | 0.385722 | post-freeze unified |
| E32768 + reconstruction | fixed Nr | 32768 / 256 | 2.672444 | 2.208698 | 3.903574 | 4.051111 | 0.384492 | P5-R |
| E240825 direct | architecture control | 240825 / 256 | 3.067098 | 2.488382 | 4.734604 | 7.206920 | 0.474357 | post-freeze unified |
| U-direct240825 | 1024 编码/处理，全场解码 | 240825 / 256 | 2.818831 | 2.318101 | 3.856026 | 3.622174 | 0.436350 | post-freeze unified |

历史 B/E resolution rows 用于架构沿革；最终 timing 不再复用 P5-R 或 P9/U5 的旧 latency。下表为最终的 valid96 同一 harness 数据，替代此前 valid32 post-freeze timing 表。

## 统一 240825 输出域性能

同一 WSL2 主机使用 RTX 5070 GPU 与 Ryzen 7 9700X CPU。三组 randomized valid96 order 为 `20260814/20260815/20260816`。GPU 在计时终点显式同步；accuracy/hash/equivalence/oracle/serialization 均不进入 service span。randomized order 的 population、support、真实 graph、native anchor k/q/CV、功率与体积守恒语义一致。

四层时间语义：

1. **fresh_single_case**：新 k/q/BC 从连续 CPU preprocessing 到同步的 240825 结果。
2. **resident_core**：neural prepared/resident inference；FVM 是 prepared-system solve-only，明确不是 E2E。
3. **batch_scale_marginal_fresh_case_estimate**：真实不同工况 `B16→B32` 的 `(T32−T16)/16`；不是重复同一输入。
4. **closed_loop_added_case_latency / saturated_streaming**：Q=1 必须完成后再提交；Q=2 则先提交两个请求、每完成一个补一个，另报 submit latency 与吞吐。

| strategy | fresh median / p95 s | resident median s | B16→B32 marginal s | Q1 median s | Q2 status / throughput | VRAM GiB |
|---|---:|---:|---:|---:|---|---:|
| E16384-reconstruction | 2.383024 / 3.625292 | 0.008487 | 2.643485 | 2.424122 | deprecated serial trace / N/A | 0.302 |
| U-v2-direct240825 | 1.520111 / 1.727624 | 0.059666 | 1.539729 | 1.520111 | not qualified: 1 pass + 1 residual-gate failure; pass-only 1.015156/s | 3.721 |
| E240825-direct control | 2.042407 / 2.636335 | 0.094279 | 2.046555 | 2.087861 | deprecated serial trace / N/A | 3.822 |
| FVM240825 | 1.713886 / 1.906241 | 1.498608 solve-only | 1.111950 | 1.713886 | qualified actual P2 / 0.904681/s | N/A |

FVM P2 是该 23 GiB 主机上能常驻且实测的饱和配置；P4/P8 因 prepared-system worker residency 失败而不作为结果。队列中的 FVM submit→result 包含排队等待，因此不能与 inter-completion 混为一谈。

- fresh single-case：E16384/U-v2/E240825 相对 FVM speedup 分别为 `0.719×/1.127×/0.839×`。修正后的 U-v2 fresh B1 快于 FVM；E16384 与 E240825 仍慢于 FVM。
- resident-core 比率：E16384/U-v2/E240825 相对 FVM solve-only 分别约 `176.6×/25.1×/15.9×`；这只是 prepared core ratio，不能称作 E2E speedup。
- 旧 E16384/E240825/U-v2 neural Q2 均由 serial trace 离线重放，现统一废弃。U-v2 的真实 Q2 首个顺序为 `1.015156 sample/s`，但第二顺序触发 residual hard gate，未取得 publication qualification；不得用该 pass-only 数字计算正式 speedup。FVM `0.904681 sample/s` 来自真实 persistent P2，继续有效。

因此可发表优势必须限定为：controlled-error、U-v2 修正后可复现的 fresh B1 优势、neural resident core 的低成本、GPU 并行潜力以及同 checkpoint 跨输出分辨率；不能继续沿用旧 P9 的 `3.40×` fresh-throughput 或任何 serial-trace Q2 结论，也不能把 resident ratio 说成 production E2E speedup。

## 独立 valid96 确认

| route | PG % | raw K | source K | peak K | interface K |
|---|---:|---:|---:|---:|---:|
| E16384-reconstruction | 3.356615 ± 0.061438 | 2.456785 ± 0.047238 | 4.095412 ± 0.079811 | 5.312322 ± 0.162981 | 0.419341 ± 0.036215 |
| E240825-direct control | 4.267199 ± 0.028495 | 2.882178 ± 0.075719 | 6.136646 ± 0.127421 | 10.323850 ± 0.397988 | 0.573623 ± 0.051114 |
| U-v2 direct (seed0 diagnostic) | 3.460815 | 2.435950 | 4.228456 | 5.725792 | 0.387372 |
| U-v2 direct (3-seed mean ± std) | 3.433555 ± 0.021381 | 2.395536 ± 0.040521 | 4.297351 ± 0.069376 | 5.254775 ± 0.335430 | 0.416257 ± 0.031543 |

E16384 − E240825 的 paired bootstrap 95% CI 在三 seed、全部五项指标上均严格小于零。例如 PG 差值为 seed0 `−0.8702 pp`（CI `[-1.1076,-0.6565]`）、seed1 `−0.9789 pp`（`[-1.2451,-0.7459]`）、seed2 `−0.8826 pp`（`[-1.1254,-0.6665]`）。这支持 E16384 的独立确认。

历史 U-v1 在 seed0 valid96 遇到 formal-valid query 超出 native domain，执行在任何 accuracy 统计前 fail-closed。随后预注册的 geometry-only audit 证明越界只发生于 3/96 样本、共 22,230/23,119,200 查询节点，最大 normalized overshoot 为 0.015873。U-v2 未回到 valid32 调优，也未改 graph policy/radius/checkpoint；其 valid96 结果仅作 diagnostic/characterization。U-v2−E16384 的 PG 为 `+0.09336 pp`（95% CI `[+0.04449,+0.15736]`），raw CV RMSE 为 `−0.04365 K`（`[-0.07617,-0.00237]`）；source/peak/interface CI 跨零。U-v2 显著优于 E240825 control，但不足以取代 E16384 production/reference。

## 可发表优势与边界

冻结证据支持以下谨慎表述：

- E16384 提供约 `3.36%` valid96 full-field PG 的 controlled-error surrogate，并跨三 seed 稳定优于同架构的 E240825-direct control。
- GPU resident core 具有毫秒级延迟和低边际计算潜力，适合在 support/graph 已准备的 parametric sweep 与 design-space exploration。
- 同一 checkpoint 可进行 fixed-resolution encoding 与不同输出分辨率查询；U-v2 已在 valid96 直接生成 240825-node 输出，但该结果只作 characterization，不作为新模型选择。
- surrogate 避免每个工况都执行完整物理线性求解，适合批量 GPU 推理研究；然而本轮真实 fresh streaming 吞吐没有超过 persistent FVM。

限制同样明确：FVM 精度和物理一致性占优；surrogate 存在约 2.7%–4.3% 的 population/route-dependent PG，source/peak 误差更大；模型仅覆盖冻结 P1i layered perfect-contact 分布。不同 timing 层不可互换，prepared resident core 不能代表 fresh production E2E。

## Final architecture/performance freeze

- **GO — E16384-reconstruction**：唯一 production/reference architecture；valid32 选择已关闭，valid96 三 seed 独立确认通过。
- **CONTROL — E240825-direct**：仅保留 architecture control；精度被 E16384 明确支配。
- **DIAGNOSTIC — U-v2-direct240825**：valid96 query-domain repair 与 accuracy/timing 均冻结；作为 parallel direct inference strategy 保留，不取代 E16384 production/reference。
- **B strategy**：作为历史 adaptive inference strategy 保留，不再搜索或优化。
- **Performance freeze**：采用 `v6_p1i_u_v2_timing_regression_closeout.json` 的 corrected U-v2 serial timing；旧 U-v2 `3.229151 s` fresh latency 与全部 neural serial-trace Q2 均废弃。FVM accuracy 只写作 reference/—，不再写成零误差模型行。

机器可读 accuracy 证据见 `configs/heat3d_v6_p1i/v6_p1i_u_v2_valid96_closeout.json`；最终 timing regression 与 corrected performance 证据见 `configs/heat3d_v6_p1i/v6_p1i_u_v2_timing_regression_closeout.json`、`docs/v6_p1i_u_v2_corrected_performance.csv`。从此不得依据 valid32 继续优化 architecture/route/graph/packing/model；test/sealed 仍保持关闭。
