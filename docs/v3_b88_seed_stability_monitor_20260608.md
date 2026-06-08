# Heat3D v3 B88 Seed-Stability Monitor

Date: 2026-06-08. Scope: SSH/devbox read-only monitoring and diagnostics for
B88 `sample_shuffle` seed-stability / optimizer-path runs. No new training was
started during this review. Generated diagnostics remain under ignored
`output/`.

## SSH State

- Branch: `research/v3-startup-supervision`
- Remote HEAD checked: `3d4467f970648689675ae3324c9d7087d2201d96`
- Running Heat3D training process: none found.
- Diagnostic logs: `output/heat3d_v2_runs/diagnostic_logs/b88_seed_stability_completed_20260608/`
- Completed diagnostics: 60 / 60 commands passed for G1, G3, C1, C2, C3, C4.

## Task Status

| group | variants | status |
|---|---|---|
| G | G1, G3 | completed, diagnostics generated |
| C | C1, C2, C3, C4 | completed, diagnostics generated |
| C | C5 | interrupted or stopped after epoch 355/400; output dir empty, no `loss_summary.json` |
| C | C6 | not run |
| D | D1, D2, D3 | not run |
| A | A1-A5 | not run |
| B | B1-B5 | not run |

C5 last logged line:
`epoch 355/400 lr=4.22e-06 train=0.69 iid=0.82 iid_err=140.09% stress=0.83 stress_err=126.86% best=e355/0.82`.

## Results

Best-prediction metrics are shown. The original B88 baseline rows are included
for context.

| run | best_epoch | valid_iid_loss | valid_stress_loss | overall DeltaT RMSE | valid DeltaT RMSE | test_id DeltaT RMSE | top-k overlap | field corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline nearest model_seed0 graph0 | 361 | 0.02303 | 0.03398 | 0.004623 | 0.002621 | 0.007559 | 0.9131 | 0.9864 |
| baseline nearest model_seed1 graph0 | 400 | 0.62342 | 0.61825 | 0.02559 | 0.008873 | 0.04561 | 0.3445 | 0.8146 |
| G3 nearest model_seed0 graph1 | 365 | 0.02432 | 0.03581 | 0.004653 | 0.002512 | 0.007354 | 0.9068 | 0.9840 |
| G1 nearest model_seed1 graph1 | 399 | 0.62357 | 0.62077 | 0.02564 | 0.008826 | 0.04560 | 0.3293 | 0.8116 |
| C1 model_seed1 warmup50 | 400 | 0.61451 | 0.60904 | 0.02536 | 0.008815 | 0.04537 | 0.3393 | 0.8168 |
| C2 model_seed1 warmup100 | 400 | 0.60990 | 0.60358 | 0.02523 | 0.008790 | 0.04523 | 0.3752 | 0.8195 |
| C3 model_seed1 minlr1e-5 | 399 | 0.61746 | 0.61213 | 0.02544 | 0.008813 | 0.04544 | 0.3363 | 0.8172 |
| C4 model_seed1 minlr3e-5 | 400 | 0.60456 | 0.59752 | 0.02510 | 0.008698 | 0.04508 | 0.3559 | 0.8238 |
| baseline discrete model_seed0 graph0 | 374 | 0.02301 | 0.03254 | 0.004391 | 0.002530 | 0.007478 | 0.9111 | 0.9872 |
| baseline legacy model_seed0 graph0 | 312 | 0.20945 | 0.28644 | 0.01766 | 0.02496 | 0.02660 | 0.7008 | 0.8939 |

## Assessment

- `graph_seed=1` preserves the good model_seed0 behavior and preserves the bad
  model_seed1 behavior. This points to model initialization / optimizer path as
  the dominant instability, not rmesh graph seed.
- C1-C4 improve model_seed1 only marginally. Even the best variant, C4, remains
  far from the seed0 repaired result (`0.60456` vs `0.02303` valid_iid loss).
- C5 was interrupted after epoch 355 and was trending poorly (`iid=0.82`), so it
  is not worth resuming unless the goal is only to complete the matrix.
- The current evidence does not justify changing graph policy defaults or moving
  to a stable B88 baseline claim.

## Continue Or Stop

Recommended next runs:

1. Run A1-A5 first. These are the most useful runs because they estimate the
   nearest-repair model-seed success rate beyond seeds 0-2.
2. Run D1-D3 next if A remains unstable. Adam / weight-decay variants directly
   test optimizer-path sensitivity for the known-bad model_seed1.
3. Run B1-B5 only if discrete_radius remains a serious candidate after A/D. It
   is useful for policy comparison, but prior seeds show discrete and nearest
   fail similarly on bad seeds.

Lower priority:

- Do not resume C5 unless matrix completeness matters; it has no completed
  output and its partial trajectory is weak.
- Skip C6 for now unless lower-lr schedule variants become important again.
