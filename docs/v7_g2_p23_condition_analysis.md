# P23 condition-stratified analysis

分层变量和 quantile bin 规则已写入冻结协议：只允许使用 input-side/case-definition
变量，先在 valid case manifest 上计算 Q0/Q33/Q67/Q100，再连接模型误差。不会按
误差挑选 bins，也不会把 seed×sample 当独立实验单位。

本轮状态为 `PREREGISTERED_NOT_EXECUTED_FAIL_CLOSED`。由于统一 full-field per-sample
artifact 尚未齐备，不能报告任何 bin 结果；若未来执行后没有稳定模式，应明确写
`NO_CLEAR_CONDITION_SPECIFIC_ADVANTAGE`。
