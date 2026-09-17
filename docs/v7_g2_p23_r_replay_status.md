# V7-G2 P23-R replay status

Status: **`P23_R_NEEDS_AMENDMENT`**.

R0 remediation is accepted while the earlier incident receipt remains
unchanged.  P23-R used only the independently staged 128-case `valid_iid`
manifest (`f42cca…`) and frozen checkpoints.  It performed no training, no
checkpoint reselection, and no access to sealed IID or DeepOHeat official100.

## Completed

- The corrected evaluator passed its synthetic sufficient-statistics tests.
- Therm-FM seeds 0/1/2 were replayed on all 128 valid cases.  The P22
  unweighted metric sanity check is `REPLAY_REPRODUCTION_PASS`; the largest
  difference from the frozen P22 per-seed metrics is `7.49e-05`.
- V6 Heat3D seed0 was mapped from its frozen 1024-point prediction through the
  original `rigno.heat3d_v6_full_field.build_reconstruction_map`; its partial
  metrics reproduce the historical V6 full-field values.

## Fail-closed boundary

The exact V6 frozen parameter/checkpoint artifacts for seeds 1 and 2 are not
present on devbox or in the checked local candidates.  They are not replaced by
retraining.  Consequently a strict six-archive evaluator, paired bootstrap,
condition-error analysis, and Table B remain blocked.  The available partial
JSON/CSV is diagnostic only and is not a publication comparison.

The checkpoint and replay SHA inventory is in
`docs/v7_g2_p23_r_checkpoint_verification.json` and the complete machine
receipt is `docs/v7_g2_p23_r_replay_status.json`.
