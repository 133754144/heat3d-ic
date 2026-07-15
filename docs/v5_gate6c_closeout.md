# Gate 6C closeout

Scratch-L1/L2 均已完成 e600，并在训练与 checkpoint 选择完成后打开 test/hard。
`test_iid` 记为 `legacy_observed_test`；hard roles 记为 `observed_report_only`，后续不得用于候选选择或调参。

| candidate | host | best epoch | valid point-global | threshold |
|---|---|---:|---:|---|
| V4P5_11_gate6c_scratch_l1_tail_balanced | devbox | 346 | 24.433257% | failed |
| V4P5_12_gate6c_scratch_l2_shape_balanced | wsl2 | 353 | 23.729025% | failed |
