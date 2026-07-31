# V6-P1i seed0 launch closeout

`V6_05_V5best_P1i_seed0` started on WSL2 from random initialization at training
commit `93ea04a52b5cfcc1a9e9af027bcd6747151737ae`. Only train is optimized and
`valid_iid` selects checkpoints; the audited test and new sealed IID roles remain
closed.

- The original HF tag and all 9216 frozen source files passed SHA256 replay.
- The 1024-case, 240825-node full-field sidecar passed exact sample-ID/shape/hash
  checks and is frozen at HF revision
  `p1i-formal1024-v1-fullfield-49023ac1-v2`.
- B8 and B16 varying-support forward/backward/AdamW/reload smokes passed. Their
  peak device bytes were 2,048,815,872 and 4,074,500,096, respectively; B8 is the
  registered micro-batch size.
- Frozen V6_03 e111 was explicitly compatible at the 11-channel schema and
  checkpoint-normalization level with random-block, but its one-time random-block
  test transfer failed badly (232.114% support and 218.229% full-field
  point-global CV-relative RMSE). This documents a model applicability boundary
  and was not used to alter P1i training.
- Launch host `DESKTOP-2GE35DV`, PID `71870`, tmux
  `V6_05_V5best_P1i_seed0`; the run is not monitored to completion here.
