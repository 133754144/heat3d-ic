# P24 sealed preregistration requirements (no unlock in P23)

P23 只列要求，不打开 sealed IID。未来单独的 evaluation-only unlock 至少需要：

1. reviewer-approved 的 test/sealed 隔离与事故 remediation receipt；
2. immutable dataset-generation/config、valid/test/sealed role manifest 和 RNG seed；
3. Heat3D、GINO、Transolver、Therm-FM 的最终 checkpoint SHA（以及对应 runner、repo、config、normalization SHA）；
4. frozen common evaluator SHA、指标定义、hotspot 定义和两个评价域（native sparse 与 full-field）；
5. valid-only checkpoint selection 已冻结，test/sealed 只做一次 evaluation-only 读取；
6. 明确不允许根据 test 结果重训、改模型或改统计规则。

P23 当前没有满足这些条件，也没有执行任何 unlock。
