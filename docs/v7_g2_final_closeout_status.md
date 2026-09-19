# V7 G2 final closeout status

状态：

- V7_G2_FULL_CLOSEOUT_COMPLETE
- V7_G2_ARCHIVED_AND_INTEGRATED
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
- test_iid 已有 documented historical access，并按冻结 selection receipt 做过一次
  visualization-only inference；其结果不进入 model selection、checkpoint selection、
  claim、Table A/B 或 bootstrap；本轮没有新的 test 访问；
- sealed IID 未访问，是唯一的最终 confirmatory holdout；DeepOHeat official100 未访问；
- cross-host publication archive receipt is tracked separately; the Mac mirror is
  verified but currently under `/private/tmp` and therefore needs durable archival before
  long-term release；
- `direct_parent_gate` 已标记 `SUPERSEDED_INVALID_GATE`；ancestry-based
  `merge-base --is-ancestor` gate 对现有 curated history 通过。

本轮 visualization、跨主机 archive、allowlist 与 ancestry 审计已通过；`research/v7`
已从 `365576d...` fast-forward 到 `4204674...`，没有 merge commit。receipt closure
将在主线同步后记录。
