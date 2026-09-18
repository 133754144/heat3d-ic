# V7 G2 P23 status

状态：**`P23_VALID_ONLY_CLOSED_P1I_PRE_CLOSEOUT`**。

身份清单已冻结；Table A 保留 Heat3D V6、GINO、Transolver 的既有 valid-only
native-1024 参考结果。P1i valid-only 同域 sidecar comparison 已完成：Heat3D G1
Full e200 direct240825 ×3 与 Therm-FM ×3 使用同一 240,825-node evaluator，且
reproduction gate 通过。corrected paired bootstrap 已改为与主表相同的 pooled
sufficient-statistics estimator；旧不一致 receipt 仅保留为审计历史。

另外，上一轮检查混合角色旧 CSV 时意外读到了 `test_iid` 行。事件、隔离和影响已在
[`v7_g2_p23_test_access_incident.json`](v7_g2_p23_test_access_incident.json) 中记录；
没有使用 test 数值、没有解锁 test/sealed，也没有访问 DeepOHeat official100，但因此
不能声称本阶段 test 全程 untouched。

`d9d961a` checkpoint→P1i inference `FAIL_CLOSED` 仍然保留；`2960ab5` 只支持
historical sidecar identity/common-evaluator reproduction，不宣称 checkpoint inference
reproduction。字段形状和机制诊断见 `docs/v7_g2_p1i_precloseout_status.md`。

P24 仅进入 evaluation-only preregistration review；本阶段不 unlock sealed。
