# V7 G2 P20 governance amendment

状态：`FROZEN_GOVERNANCE_CORRECTION`（2026-09-16）。

本修正区分历史事实与当前阶段边界：V6 的 `test_iid=128` 已在路线和 checkpoint 冻结后，于 E16384/model_seed0 路径打开过一次，且没有用于选择、调参或阈值修改。当前真正的最终未触碰 holdout 是另行预注册的 `sealed IID`，仍未生成、未打开。P20 以及后续 G2 receipt 只能声明“本轮未访问 test_iid/sealed”，不能再把 `test_iid` 写成全局未打开。

旧 receipt 保持不可变；其中的 no-access 语句解释为各自任务范围内的事实，不追溯改写为全局历史断言。G1 仍关闭且未修改。

证据来源与 SHA 见同名 JSON。P20 本轮没有训练、没有新评估、没有访问任何 DeepOHeat official100 或 sealed 数据。
