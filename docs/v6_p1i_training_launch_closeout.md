# V6-P1i seed0 launch closeout and correction

The original `V6_05_V5best_P1i_seed0` launch was stopped before its first epoch
after an execution-contract audit found micro-B8 and validation/prediction-B16,
which did not match the V6best contract (micro-B24 and validation/prediction-B32).
Its output directory is retained as a historical, non-evaluable launch record;
it must not be overwritten. The replacement is
`V6_05_V5best_P1i_seed0_B24`, random initialized, with one real B24 update per
optimizer step. Only train is optimized and only `valid_iid` selects
checkpoints; the audited test and new sealed IID roles remain closed.

- The original HF tag and all 9216 frozen source files passed SHA256 replay.
- The 1024-case, 240825-node full-field sidecar passed exact sample-ID/shape/hash
  checks and is frozen at HF revision
  `p1i-formal1024-v1-fullfield-49023ac1-v2`.
- The historical B8 and B16 varying-support forward/backward/AdamW/reload
  smokes passed. Their peak device bytes were 2,048,815,872 and 4,074,500,096,
  respectively; these are pre-correction diagnostics, not the replacement
  training contract.
- Frozen V6_03 e111 was explicitly compatible at the 11-channel schema and
  checkpoint-normalization level with random-block, but its one-time random-block
  test transfer failed badly (232.114% support and 218.229% full-field
  point-global CV-relative RMSE). This documents a model applicability boundary
  and was not used to alter P1i training.
- The stopped launch used host `DESKTOP-2GE35DV`, training PID `71882` (launcher
  PID `71870`), tmux `V6_05_V5best_P1i_seed0`, and training commit
  `93ea04a52b5cfcc1a9e9af027bcd6747151737ae`; its log ends in
  `KeyboardInterrupt` during startup and contains no completed epoch.
- The replacement YAML is prepared but not started. Its batch contract is
  `batch_size=24`, `micro_batch_size=24`, `validation_batch_size=32`,
  `prediction_batch_size=32`, `drop_last=false`, 32 updates per epoch.
