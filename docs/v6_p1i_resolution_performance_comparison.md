# V6 P1i：1024+ΔT、Anchor-derived 与物理求解器性能对比

本报告保存本轮审查用的实测对比结果，不训练、不修改 checkpoint，也未访问
test/sealed。指标与原始实验文件保持一致；CSV 是数值表，本文档记录测量边界和
不可直接比较的部分。

## 主结果

| N | 1024+ΔT PG / raw K | 1024+ΔT 总耗时 cold / new / cached (s) | 1024+ΔT graph-free forward (s) | Anchor full-field PG / raw K | Anchor graph / E2E / steady / reconstruction (s) | FVM PG / raw K | FVM 总耗时 cold / cached (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 3.1201% / 2.9699 | 9.9125 / 1.7874 / 0.003720 | 0.002682 | 2.8238% / 2.6382 | 68.1768 / 88.4267 / 0.003265 / 1.9825 | 1.1905% / 1.5185 | 0.6521 / 0.006258 |
| 8192 | 3.3072% / 3.1138 | 9.9853 / 1.7507 / 0.003882 | 0.002694 | 2.7428% / 2.4917 | 73.9926 / 98.5114 / 0.004315 / 6.2996 | 1.1911% / 1.5188 | 0.6610 / 0.012109 |
| 16384 | 3.3819% / 3.1706 | 10.0505 / 1.7439 / 0.004588 | 0.002937 | 2.8174% / 2.5343 | 82.8008 / 109.1279 / 0.007674 / 6.9979 | 1.1912% / 1.5190 | 0.6747 / 0.025245 |
| 32768 | 3.3845% / 3.1687 | 9.9768 / 1.7753 / 0.005673 | 0.003011 | 2.9394% / 2.6469 | 94.7360 / 121.9154 / 0.017681 / 7.5372 | 1.1916% / 1.5192 | 0.7138 / 0.062941 |
| 65536 | 3.3360% / 3.1364 | 10.0761 / 1.7911 / 0.006358 | 0.002900 | 3.1442% / 2.7995 | 122.2256 / 152.6066 / 0.034644 / 8.5858 | 1.1919% / 1.5194 | 0.8191 / 0.157089 |
| 240825 | 3.0720% / 2.9302 | 10.1534 / 1.8478 / 0.012970 | 0.002929 | — | — | 0.06965% / 0.03655 | 2.3056 / 1.669673 |

1024 基线本身：1024+ΔT 的 support PG/raw 为 `1.7545% / 1.2950 K`，重建到
240825 后为 `3.0720% / 2.9302 K`。最新 Anchor-derived N=1024 为
support `1.7544% / 1.2949 K`、full-field `3.0720% / 2.9302 K`；其 graph、E2E、
steady forward、reconstruction 分别为 `58.7863 / 80.6248 / 0.002138 / 2.0501 s`。

## 口径和来源

- **1024+ΔT**：历史统一基准中的 `production_reconstruction`。分辨率行是
  `timing_queue_resolution_diagnostic_only`，时间为 32 个固定 valid 样本、B1
  下的逐样本 median；`cold/new/cached` 分别表示进程冷启动、JIT 已缓存但图/映射
  未缓存、图/JIT/重建映射均缓存。
- **Anchor-derived**：本轮 GPU-only ladder；冻结 1024 anchor context/scale，
  在 N 个 query nodes 上 forward，再重建到 240825 full field。表中 E2E、graph、
  reconstruction 是 executor 原样记录的 valid32 worker wall-clock，steady 是
  graph-free model forward median；没有把阶段时间相加代替 E2E。
- **FVM**：历史统一基准中的合法 structured-FVM resolution diagnostic，时间仍为
  同一 32-sample 队列下的逐样本 median；列出 process-cold 与 fully-cached。
- 三类精度评价域并不完全相同：1024+ΔT 的 N 行是 `PG@N`，Anchor 行是
  `full PG@240825`，FVM 行是对应结构化网格的 `PG@N`。因此不能把每一行的三种
  PG 当作严格同步的模型排名；它们用于显示各路线的实测误差与成本边界。

## 结论

1. Anchor-derived 在 4096–65536 的 full-field PG 为 `2.7428–3.1442%`，8192
   最低；没有出现高分辨率精度崩溃。
2. 1024+ΔT 的 graph-free forward 约 `0.0027–0.0030 s`，但完整 cached
   reconstruction 随输出分辨率增大；其 240825 full-field PG 为 `3.0720%`。
3. Anchor-derived 的主要成本是图构建：4096 到 65536 从 `68.18 s` 增至
   `122.23 s`；steady forward 从 `0.003265 s` 增至 `0.034644 s`。
4. FVM 的 240825 cold/cached 总耗时为 `2.3056/1.6697 s`，对应 full PG
   `0.06965%`；这是物理求解精度基线，不应与模型误差混为同一类。

## 可追溯来源

- [本轮 Anchor-derived CSV](</private/tmp/v6_p1i_gpu_only_high_n_valid32_b935713_extract/gpu_only_high_n_accuracy_resolution_latency.csv>)，GPU-only 执行提交：`b93571317a03f25aa69b9a1f4515c4b06aeba246`。
- [历史统一 timing CSV](</private/tmp/v6-p1i-governance-clean-20260807/configs/heat3d_v6_p1i/v6_unified_performance_timing.csv>)。
- [历史分辨率 timing CSV](</private/tmp/v6-p1i-governance-clean-20260807/configs/heat3d_v6_p1i/v6_unified_performance_resolution.csv>)。
- [历史准确率 CSV](</private/tmp/v6-p1i-governance-clean-20260807/configs/heat3d_v6_p1i/v6_unified_performance_accuracy.csv>)。
- 历史统一 benchmark 执行提交：`04c63db7134308768b04914339a5d5fae67e56de`。

原始实验仍保留在各自归档位置；本次只新增审查报告和数值 CSV，不复制或覆盖
data/output/checkpoint/log 工件。
