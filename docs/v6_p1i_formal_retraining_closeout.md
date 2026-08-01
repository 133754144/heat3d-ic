# V6 P1i formal retraining closeout

## Frozen contract

`formal1024_v1`, its train/valid/test split, V6_03 architecture, four-term
loss, B24, e600 schedule and `valid_rel_rmse_v4_pct` selection remain frozen.
All formal runs are random initialized. Only train and `valid_iid` are opened;
the audited test and sealed confirmatory IID roles remain closed.

The reliable runner writes resolved config, complete command and provenance
before training. Epoch 0 participates in all best selections. On improvement it
atomically writes optimizer-aware best checkpoints; it writes latest every ten
epochs and final immediately after the epoch loop. Params, optimizer state,
epoch and all best records are reload-checked. Later metadata, diagnostics or
export failures cannot remove these checkpoints.

## Legacy seed0 archive limitation

The legacy e600 run at training commit
`dfe3cf62b2e5795630f5504480b2facfd5f6c65c` preserved both valid prediction
archives, and identical copies are verified on devbox and WSL2. Its registered
log was never created and no complete terminal scrollback survived a search of
both repositories, log/output trees, shell histories, tmux state, temporary
directories and Codex sqlite logs. Therefore the user-supplied final excerpt is
archived explicitly as incomplete; no complete log is fabricated.

The Hugging Face upload path is preregistered in the archive manifest. The
2026-08-02 upload attempt was blocked by Hub proxy 503/TLS timeout despite a
present local token; revision/commit remain pending external network recovery.

## Legacy valid full-field result

The frozen 240825-node sidecar was read for 128 `valid_iid` samples only.
Oracle support reconstruction isolates the sampling/reconstruction floor;
e542/e600 rows combine model and reconstruction error.

| row | point-global true-RMS % | point-global CV % | sample-first CV % | raw CV RMSE K | peak RMSE K |
|---|---:|---:|---:|---:|---:|
| oracle 1024 reconstruction | 3.066851 | 3.614129 | 3.584837 | 2.730219 | 0.324126 |
| legacy e542 | 3.439267 | 3.959241 | 3.963344 | 2.990927 | 3.905090 |
| legacy e600 | 3.451445 | 3.962897 | 3.972401 | 2.993689 | 3.960272 |

Thus the 240825-node result is dominated by the varying-support reconstruction
floor. e542 remains slightly better than e600, consistent with the 1024-point
valid evaluation. Full source/background/layer/interface/surface metrics and
worst samples are in the JSON/CSV closeout.

## Gate and formal queue

The real B24 e3 fault-injection smoke passed: 32 real updates per epoch, finite
loss/gradients/predictions, no OOM/NaN/Inf, and exact params/optimizer reload for
best/latest/final after the intended metadata exception. The formal seed0 run
is assigned to devbox; WSL2 runs seed1 and then seed2 serially. The queue runs a
checkpoint reload check and valid support/full-field evaluation after seed1;
any failure prevents seed2 from starting.
