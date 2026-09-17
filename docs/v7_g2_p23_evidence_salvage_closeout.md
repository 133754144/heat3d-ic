# V7-G2 P23 evidence salvage closeout

本轮没有训练、没有重选 checkpoint、没有读取新的 `test_iid`、sealed 或 DeepOHeat official100。

## 结论

1. V6 seed1/seed2 的目标二进制 SHA 在完整取证搜索后正式关闭为 `PERMANENT_BINARY_ARTIFACT_LOSS_AFTER_ORIGINAL_WSL2_RETIREMENT`。历史 V6 三 seed 结果仍然有效；丢失的是未来 replay capability，不是历史结果。
2. 旧 V6 evaluator 与 P23 common evaluator 对 point-global relative RMSE、sample-first relative RMSE 和 peak-temperature RMSE 的 seed0 数值桥接均 PASS。历史 `raw_cv_weighted_rmse_K` 与新 unweighted `rmse_K` 不可混用。
3. 历史 aggregate Table B 只保留上述三个有语义桥接的指标；不对 V6 三 seed aggregate 做 paired bootstrap，也不新增缺少历史同定义证据的 hotspot/MAE 主表比较。
4. V7 e200 seed0/1/2 的 checkpoint、SHA、200 epoch receipt、valid-only 选择规则和 exact reload 均通过。它是独立的 dynamic-LR 200-epoch reference cohort，不能写成 e600 的前 200 个 epoch。
5. e200 valid-only replay 已在其自身 DeepOHeat-v1 `101×101×56=571256` 域完成，并复用已冻结的 P14 full-field receipts。P23 current evaluator/Therm-FM 使用的是 V6 P1i `65×65×57=240825` 域和不同 case IDs，因此 e200×Therm-FM paired/bootstrap/condition comparison 被明确标为 `NOT_APPLICABLE_DOMAIN_MISMATCH`，没有强行混合。

详见：

- `v7_g2_p23_v6_artifact_loss_closeout.json`
- `v7_g2_p23_v6_metric_semantic_bridge.json`
- `v7_g2_p23_historical_table_b.json`
- `v7_g2_p23_v7_e200_identity_audit.json`
- `v7_g2_p23_v7_e200_replay_receipt.json`
- `v7_g2_p23_evidence_hierarchy.md`

P24 只保留两个候选：Option A（固定 seed0 sealed confirmation）或 Option B（先解决 e200/跨模型 domain 对称性后再做三 seed sealed reference）。本轮不选择、不执行任何一个。
