# V6/P1i high-N graph-scale 因果消融

范围：冻结 seed0 checkpoint 与 frozen valid32；无训练、无 test/sealed 访问。A 与 FVM accuracy 证据只读复用；B/C 新增 8192/16384 accuracy/graph，A 仅补同 executor 的 8192/16384 timing-only。

| 策略 | N | full PG % | sample-first % | raw K | source K | peak K | anchor drift K | Nr | under-covered | warm/new-case ms | VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 8192 | 2.742758 | 3.086004 | 2.491700 | 3.606035 | 3.369424 | 2.194680 | 2048 | 0.000000 | 3.760/477.496 | 177.0 |
| A | 16384 | 2.817371 | 3.158629 | 2.534275 | 3.736571 | 3.496278 | 2.349743 | 4096 | 0.000000 | 6.771/576.410 | 295.6 |
| B | 8192 | 2.739835 | 3.031378 | 2.458789 | 3.587522 | 3.478606 | 2.185888 | 1024 | 0.000000 | 2.664/441.805 | 181.7 |
| B | 16384 | 2.755815 | 3.003881 | 2.435761 | 3.683810 | 3.440714 | 2.256876 | 2048 | 0.000000 | 4.963/545.607 | 237.8 |
| C | 8192 | 2.840876 | 3.183912 | 2.556106 | 3.675714 | 3.390984 | 2.288627 | 2048 | 0.003460 | 19.107/610.663 | 840.6 |
| C | 16384 | 2.816635 | 3.112672 | 2.513144 | 3.761769 | 3.536209 | 2.370471 | 4096 | 0.000950 | 47.354/909.193 | 2223.2 |

## 冻结判定

- B 明确改善 gate：**False**；平均 PG/raw 改善为 0.032239 pp / 0.065712 K，但 source/peak 非退化=False。
- C 明确改善 gate：**False**；它在 8192 退化并产生可测 under-coverage。
- 未触发 D；未冻结 winner，因此没有运行 4096/32768 扩展。生产策略继续使用 A。
- native-1024 physical-coverage 假设**不成立**。仅保持半径不能解释 8192+ 趋势；support 分布、P2R 稀疏度、context/scale 表示和图响应仍是复合因素。
- 历史 P1h 对照：8192/16384 的 P2R regional degree 为 28.225/28.726，Nr=1024/2048；P1i A 对应为 11.084/9.451，Nr=2048/4096。B 虽把 Nr 降至 P1h 密度，却未复现 P1h 的 physical/source coverage。
- 候选 timing 使用同步连续 span，但独立 neural/apply 子段包含一次首调用 JIT；它们不进入正式性能表。warm-cache/new-case 连续 span 与 accuracy 仍有效。
- A/B/C 的 8192/16384 new-case 现均来自同一 executor、同一 WSL2 RTX 5070、同一 valid32 的连续 span：fresh graph + group prepare + cached-map load/H2D + forward + GPU reconstruction。A 补测不读标签、不算指标、不保存预测。
- 实际新增计算：B8192 因初始误把 report-only k/q/CV 当 hard gate，在保留结果前有两次工程重试；最终 B8192/B16384/C8192/C16384 及 A8192/A16384 timing-only 均由 SHA 绑定。没有重跑 A/FVM accuracy，也没有运行 D 或 winner 扩展。

## 工程优先级

冻结策略仍为 A；新拓扑图构建主导端到端延迟，而 warm neural 仅为毫秒级。下一步优先 GPU 图构建优化。只有 support hash 重复时才优先固定图复用；对当前 B1 瓶颈，batch inference 优先级更低。
