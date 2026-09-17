# V7 G2 P23 status

状态：**`P23_NEEDS_AMENDMENT`**。

身份清单已冻结；Table A 保留 Heat3D V6、GINO、Transolver 的既有 valid-only
native-1024 参考结果。Table B 的共同 full-field evaluator、逐样本 artifact、paired
bootstrap 和 condition analysis 均已定义，但没有在 P1i 上运行，因为 V6 seed1/2
逐样本预测和 Therm-FM 逐样本预测尚未齐备。

另外，上一轮检查混合角色旧 CSV 时意外读到了 `test_iid` 行。事件、隔离和影响已在
[`v7_g2_p23_test_access_incident.json`](v7_g2_p23_test_access_incident.json) 中记录；
没有使用 test 数值、没有解锁 test/sealed，也没有访问 DeepOHeat official100，但因此
不能声称本阶段 test 全程 untouched。

下一步只能是 reviewer-approved 的 valid-only 数据/预测 manifest remediation，随后
运行已经冻结的 evaluator 和 10,000 次 paired case bootstrap。P24 仅保留
evaluation-only preregistration requirements，不在 P23 unlock。
