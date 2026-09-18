# P24 sealed preregistration requirements (no unlock in P23)

P23 只列要求，不打开 sealed IID。未来单独的 evaluation-only unlock 至少需要：

1. reviewer-approved 的 test/sealed 隔离与事故 remediation receipt；
2. immutable dataset-generation/config、valid/test/sealed role manifest 和 RNG seed；
3. Heat3D、GINO、Transolver、Therm-FM 的最终 checkpoint SHA（以及对应 runner、repo、config、normalization SHA）；
4. frozen common evaluator SHA、指标定义、hotspot 定义和两个评价域（native sparse 与 full-field）；
5. valid-only checkpoint selection 已冻结，test/sealed 只做一次 evaluation-only 读取；
6. 明确不允许根据 test 结果重训、改模型或改统计规则。

P1i valid-only evaluator/domain/metric closure 已完成，且已生成
`docs/v7_g2_p1i_precloseout_status.md`；因此 P24 可进入 reviewer review。
但是本阶段仍未满足“正式 unlock 已获批准”的治理条件，尤其是 G1 direct sidecar 的
manifest enumeration caveat 与历史 incident remediation 必须在 unlock 前被明确接受。
本轮没有执行任何 unlock。
