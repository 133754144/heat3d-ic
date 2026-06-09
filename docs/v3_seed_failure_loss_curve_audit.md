# Heat3D v3 Seed Failure Loss-Curve Audit

Purpose: offline audit of completed B88 sample_shuffle e400 `loss_summary.json`
files to locate when successful and failed model seeds separate. No training was
run.

Generated ignored outputs on devbox:

- `output/heat3d_v3_seed_loss_curve_audit/seed_loss_curve_audit.json`
- `output/heat3d_v3_seed_loss_curve_audit/seed_loss_curve_selected_epochs.csv`
- `output/heat3d_v3_seed_loss_curve_audit/seed_loss_curve_audit.md`

| run | label | best epoch | best iid | final iid | e20 iid | e100 iid | e400 iid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nearest_seed0 | mid_plateau | 361 | 0.02303 | 0.02308 | 0.2427 | 0.1100 | 0.02308 |
| nearest_seed1 | early_bad | 400 | 0.6234 | 0.6234 | 0.9521 | 0.7751 | 0.6234 |
| nearest_seed3_A1 | early_bad | 400 | 0.7048 | 0.7048 | 1.0140 | 0.8909 | 0.7048 |
| nearest_seed4_A2 | early_bad | 400 | 0.8301 | 0.8301 | 1.1200 | 1.0240 | 0.8301 |
| nearest_seed5_A3 | early_bad | 400 | 0.6812 | 0.6812 | 0.9209 | 0.7742 | 0.6812 |
| nearest_seed6_A4 | early_bad | 400 | 0.6023 | 0.6023 | 0.9583 | 0.7724 | 0.6023 |
| nearest_seed7_A5 | early_bad | 400 | 0.7426 | 0.7426 | 1.0780 | 0.9538 | 0.7426 |
| discrete_seed0 | mid_plateau | 374 | 0.02301 | 0.02306 | 0.2629 | 0.07831 | 0.02306 |
| discrete_seed1 | early_bad | 400 | 0.6232 | 0.6232 | 0.9695 | 0.7778 | 0.6232 |
| discrete_seed6_B4 | early_bad | 400 | 0.5024 | 0.5024 | 0.9779 | 0.7435 | 0.5024 |
| C1_seed1_warmup50 | early_bad | 400 | 0.6145 | 0.6145 | 0.9887 | 0.7984 | 0.6145 |
| C2_seed1_warmup100 | early_bad | 400 | 0.6099 | 0.6099 | 0.9979 | 0.8487 | 0.6099 |
| C3_seed1_minlr1e-5 | early_bad | 399 | 0.6175 | 0.6175 | 0.9502 | 0.7792 | 0.6175 |
| C4_seed1_minlr3e-5 | early_bad | 400 | 0.6046 | 0.6046 | 0.9498 | 0.7790 | 0.6046 |
| D1_seed1_wd0 | early_bad | 400 | 0.6236 | 0.6236 | 0.9481 | 0.7739 | 0.6236 |
| D2_seed1_wd1e-5 | early_bad | 400 | 0.6231 | 0.6231 | 0.9632 | 0.7781 | 0.6231 |
| D3_seed1_adam | early_bad | 400 | 0.6225 | 0.6225 | 0.9497 | 0.7748 | 0.6225 |
| G1_seed1_graphseed1 | early_bad | 399 | 0.6236 | 0.6236 | 0.9644 | 0.7755 | 0.6236 |
| G3_seed0_graphseed1 | mid_plateau | 365 | 0.02432 | 0.02463 | 0.2615 | 0.06900 | 0.02463 |

## Early Split

The split is visible during warmup. At epoch 1, nearest seed0 valid_iid is
`0.9723`, while nearest seed1/seed6 are already worse at `1.1200`/`1.1305`.
By epoch 10, seed0 reaches `0.3607`, while seed1/seed6 remain near
`0.9950`/`0.9922`. By epoch 20, seed0 is `0.2427`, but failed seeds remain
near `0.95`.

## Conclusions

- Failure is primarily `early_bad`, not late undertraining. Most failed runs
  improve slowly but remain on a bad trajectory from the first 10-20 epochs.
- Higher min_lr, longer warmup, lower/no weight decay, Adam vs AdamW, and
  graph_seed=1 do not rescue model_seed1.
- discrete_radius reproduces the same seed sensitivity: seed0 succeeds, seed1
  and seed6 stay poor.
- LR sweep is still worth running because the split occurs inside the early
  optimization path, but schedule-only changes are unlikely to be sufficient if
  activation/gradient path diagnostics show collapse.
- The next higher-value step is initialization / latent path auditing before
  launching many more e400 sweeps.
