# V6/P1i P4 fixed-support closeout

P4-A 使用与 P2/P3 相同的固定 support/graph，重新计算 valid32 标签指标；没有复用 adaptive accuracy 冒充 fixed accuracy。

| route | adaptive PG % | fixed PG % | ΔPG pp | adaptive raw K | fixed raw K | Δsource K | Δpeak K | Δinterface K | max sample Δraw K | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B8192_recon | 2.7398 | 3.0455 | +0.3056 | 2.4588 | 2.5896 | +1.9807 | +1.6056 | -0.0713 | +1.0806 | NO-GO (peak;point_global;source) |
| E32768_recon | 2.6725 | 3.0378 | +0.3654 | 2.2087 | 2.4505 | +1.2550 | +1.8159 | +0.0844 | +0.8677 | NO-GO (peak;point_global;raw;source) |

## Decision

- B8192-recon fixed support：NO-GO；PG、source、peak 超出预注册 margin。
- E32768-recon fixed support：NO-GO；PG、raw、source、peak 超出预注册 margin。
- fixed-support timing 不能与 adaptive-support accuracy 拼接为同一路线；P3 的 113x/44.8x 已在上一阶段废弃，本轮不产生新的 production speedup。
- 因两条路线均失败，按 fail-fast 合同不执行 P4-B/C/D/E，也不提出开启 test/sealed。

逐样本结果见 `docs/v6_p1i_p4a_fixed_support_paired.csv`。
