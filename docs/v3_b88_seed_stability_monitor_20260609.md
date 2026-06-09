# Heat3D v3 B88 Seed-Stability Monitor

Date: 2026-06-09. Scope: SSH/devbox read-only monitoring and diagnostics for
B88 `sample_shuffle` seed-stability / optimizer-path runs. No new training was
started during this review. Diagnostics were generated only for already
completed runs and remain under ignored `output/`.

## SSH State

- Remote branch: `research/v3-startup-supervision`
- Remote HEAD checked: `e2ab35efa683ee637903667df8196b02d08c3721`
- Running Heat3D training process: none found.
- New diagnostics generated: 130 / 130 commands passed for A1-A5, B1-B5, D1-D3.
- Diagnostic logs: `output/heat3d_v2_runs/diagnostic_logs/b88_seed_stability_ABD_completed_20260609/`

## Task Status

| group | variants | status |
|---|---|---|
| A | A1-A5 nearest_repair model_seed3-7 | completed, diagnostics generated |
| B | B1-B5 discrete_radius model_seed3-7 | completed, diagnostics generated |
| D | D1-D3 optimizer / weight_decay variants | completed, diagnostics generated |
| G | G1, G3 | completed earlier, diagnostics present |
| C | C1-C4 | completed earlier, diagnostics present |
| C | C5 | interrupted after epoch 355/400; output dir empty, no `loss_summary.json` |
| C | C6 | not run |

C5 last logged line remains:
`epoch 355/400 lr=4.22e-06 train=0.69 iid=0.82 iid_err=140.09% stress=0.83 stress_err=126.86% best=e355/0.82`.

## Core Results

Best-prediction metrics are shown. Lower `valid_iid_loss` and lower overall
DeltaT RMSE are better.

| run | best_epoch | valid_iid_loss | valid_stress_loss | overall DeltaT RMSE | top-k overlap | field corr |
|---|---:|---:|---:|---:|---:|---:|
| baseline nearest seed0 | 361 | 0.02303 | 0.03398 | 0.004623 | 0.9131 | 0.9864 |
| baseline discrete seed0 | 374 | 0.02301 | 0.03254 | 0.004391 | 0.9111 | 0.9872 |
| baseline legacy seed0 | 312 | 0.20945 | 0.28644 | 0.01766 | 0.7008 | 0.8939 |
| nearest A1 seed3 | 400 | 0.70477 | 0.70397 | 0.02871 | 0.2846 | 0.7852 |
| nearest A2 seed4 | 400 | 0.83014 | 0.83439 | 0.03197 | 0.2494 | 0.7421 |
| nearest A3 seed5 | 400 | 0.68118 | 0.65508 | 0.02989 | 0.3953 | 0.6204 |
| nearest A4 seed6 | 400 | 0.60228 | 0.58146 | 0.02691 | 0.5168 | 0.7868 |
| nearest A5 seed7 | 400 | 0.74264 | 0.74051 | 0.03039 | 0.2961 | 0.7733 |
| discrete B1 seed3 | 400 | 0.70535 | 0.70445 | 0.02868 | 0.3078 | 0.7888 |
| discrete B2 seed4 | 399 | 0.82991 | 0.83418 | 0.03198 | 0.2070 | 0.7424 |
| discrete B3 seed5 | 400 | 0.59061 | 0.57578 | 0.02766 | 0.4541 | 0.6791 |
| discrete B4 seed6 | 400 | 0.50237 | 0.50169 | 0.02402 | 0.5188 | 0.8513 |
| discrete B5 seed7 | 400 | 0.74228 | 0.74102 | 0.03043 | 0.2703 | 0.7724 |
| D1 seed1 wd0 | 400 | 0.62363 | 0.61839 | 0.02560 | 0.3846 | 0.8146 |
| D2 seed1 wd1e-5 | 400 | 0.62311 | 0.61761 | 0.02558 | 0.3764 | 0.8145 |
| D3 seed1 Adam | 400 | 0.62252 | 0.61723 | 0.02556 | 0.3465 | 0.8148 |

Previously completed C/G runs also failed to recover seed1. The best C variant
was C4 with `valid_iid_loss=0.60456`, still far from seed0.

## Assessment

- No currently running task is present on devbox.
- No new successful seed was found. A1-A5 all failed; nearest repair appears
  successful only for model_seed0 among tested seeds 0-7.
- Discrete radius does not solve seed instability. B1-B5 all failed, matching
  the earlier discrete seed1/seed2 failures.
- D1-D3 show that removing weight decay, reducing weight decay, or switching to
  Adam does not rescue the known-bad model_seed1 path.
- C5 is an interrupted run with a poor trajectory; it is not worth resuming.
- C6 is the only unrun config, but current evidence makes it low value.

## Recommendation

Do not continue the remaining C5/C6 queue items unless matrix completeness is
required. The useful conclusion has already been reached: B88 graph repair can
produce an excellent seed0 model, but the path is not seed-stable. Further work
should shift from more B88 e400 sweeps to diagnosing initialization/optimization
dynamics, e.g. early-epoch activation/gradient audits for seed0 versus failed
seeds, checkpoint/trajectory comparison, or model-path changes gated by that
evidence.
