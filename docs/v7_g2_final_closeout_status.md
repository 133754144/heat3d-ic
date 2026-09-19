# V7 G2 final closeout status

状态：

- V7_G2_FULL_CLOSEOUT_COMPLETE
- G2_VALID_ONLY_EVIDENCE_CLOSED
- G2_DEVELOPMENT_COMPLETE
- P24_READY_FOR_REVIEW_NOT_UNLOCKED

已完成：

- Table A primary contract fixed to native1024 point-global relative RMSE；
- Table B primary contract fixed to full-field240825 sample-first relative RMSE；
- P8 hierarchical seed+case bootstrap for Heat3D vs GINO/Transolver；
- valid-only archive index and SHA receipts；
- P1i performance report and plotting audit；
- valid figures regenerated with exact dense policy and native diagnostic policy；
- input-only valid/test figure selection receipt frozen before test inference；
- valid and one-time test_iid visualization-only exact-dense figures；
- Therm-FM z_x_y -> x_y_z layout and slice-metric consistency audit；
- JSON parse, Python compile, evaluator synthetic and negative-gate tests pass。

保留的 fail-closed / governance items：

- d9d961a checkpoint-to-P1i inference FAIL_CLOSED；
- 2960ab5 PASS 仅表示 frozen historical sidecar identity/common-evaluator
  reproduction，不表示 checkpoint inference reproduction；
- test_iid 仅按冻结 selection receipt 做了一次 visualization-only inference；其结果不进入
  model selection、checkpoint selection、claim、Table A/B 或 bootstrap；
- sealed 和 DeepOHeat official100 未访问；
- independent off-host publication archive receipt is tracked separately；
- 不在本地 merge 到 research/v7，以避免修改 G1 主线；需人工审查后再决定。

本轮 visualization、archive 与 allowlist-only curated integration 均已通过；
`research/v7` ref 保持不变，后续主线集成提交为独立本地
`codex/g2-curated-integration@23098098657694e1e71ee9a45cfb27c7bb85a0d7`，未 push。
