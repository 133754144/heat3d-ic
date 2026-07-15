# Gate 6D N3-L2 valid-only paired attribution

只使用 `valid_iid`：N3 best e402 对 L2 best e353。test/hard 未用于本分析。

| metric | N3 | L2 | L2-N3 |
|---|---:|---:|---:|
| point-global relative RMSE | 24.076221% | 23.729025% | -0.347196 pp |
| sample-first relative RMSE | 20.658884% | 20.835687% | 0.176804 pp |

改善样本 65/128，退化样本 63/128。
top-10 改善样本占全部正向改善 40.46%；true CV-RMS Q4 占 25.50%。

结论：L2 improvement is not concentrated in only a small top-improvement subset.

六个变量的四分位统计、逐样本 SSE/shape/scale/amplitude/oracle 指标和 top improvement/regression 均保存在 JSON。
