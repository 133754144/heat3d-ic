# V7 G2 P23 test-access incident (fail-closed)

在 P23 开始阶段，为检查旧文件结构执行了只读命令：

```text
sed -n '1,80p' configs/heat3d_v6_p1i/v6_p1i_error_tail_samples.csv
```

该 CSV 是混合角色的历史文件；输出在合法 `valid_iid` 行之后包含了
`test_iid` 行。因此本次必须记为 **accidental test_iid read**。没有把任何 test
数值复制到结果、没有用于模型选择/指标/分层分析，也没有解锁 sealed 或
DeepOHeat official100，但不能再宣称 P23 全程“test untouched”。

后续处置：该混合 CSV 被列入 denylist；本轮不运行依赖未单独验证的 valid-only
case manifest 的 paired bootstrap、condition analysis 或正式 publication ranking。
保留所有历史 FAIL-CLOSED 证据，不覆盖、不删除、不把该事件静默改写为“未访问”。
P23 readiness 固定为 `P23_NEEDS_AMENDMENT`，等待人工批准的隔离数据路径和重新审计。
