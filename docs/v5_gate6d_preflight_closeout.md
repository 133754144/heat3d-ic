# Gate 6D preflight closeout

冻结 evaluator `639872abcb0f7afd3b6c2d319a7d395bde75c9a4` 与 collector 不等价，故冻结 evaluator 结果为权威。

N3-L2 成对归因只使用 valid_iid；24D coverage 只用 train 拟合、valid 查询。sample-relative 改善不集中于少数 top-10；但 point-global SSE 的 true-DeltaT Q1-Q3 总体退化，Q4 提供全部净改善。

sealed IID seed 固定为 `2026071501`，当前仅冻结可执行合同，未生成标签、未推理。首次开启条件是候选与完整训练方案完全冻结。

本轮 training_started=false，未启动 multi-seed，未新增 loss 配置。
