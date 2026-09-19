# V7 G2 final closeout status

状态：

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
- JSON parse, Python compile, evaluator synthetic and negative-gate tests pass。

保留的 fail-closed / governance items：

- d9d961a checkpoint-to-P1i inference FAIL_CLOSED；
- 2960ab5 PASS 仅表示 frozen historical sidecar identity/common-evaluator
  reproduction，不表示 checkpoint inference reproduction；
- 没有新的 test_iid 数值证据，sealed 和 DeepOHeat official100 未访问；
- test visualization 因固定 prediction archive 缺失而 blocked；
- archive 尚无独立 off-host replica；
- 不在本地 merge 到 research/v7，以避免修改 G1 主线；需人工审查后再决定。
