# Heat3D v3 B88 Sample-Shuffle e400 Results

Scope: read-only evaluation of completed SSH runs. Diagnostics were generated
from existing `predictions.npz` and `best_predictions.npz`; no training was
started in this review. Output diagnostics remain under ignored `output/`.

## Diagnostics

- Runs: 7 B88 `latent96_s6_mlp2` sample-shuffle e400 configs.
- Diagnostics: final and best predictions for baseline comparison, error bins,
  run summary, condition diagnostics, and field-shape diagnostics.
- Status: 70 / 70 diagnostics commands passed.
- Logs: `output/heat3d_v2_runs/diagnostic_logs/b88_sample_shuffle_e400_7runs/`.

## B88 Results

Best-prediction metrics are shown. `valid_iid_loss` and `valid_stress_loss` are
from `loss_summary.json`; RMSE, top-k, correlation, and bin0 metrics are from
the generated diagnostics.

| policy | model_seed | best_epoch | valid_iid_loss | valid_stress_loss | overall DeltaT RMSE | valid DeltaT RMSE | test_id DeltaT RMSE | top-k overlap | field corr | bin0 bias | bin0 over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | 0 | 312 | 0.2095 | 0.2864 | 0.01766 | 0.02496 | 0.02660 | 0.7008 | 0.8939 | 0.005624 | 0.8738 |
| nearest_repair | 0 | 361 | 0.02303 | 0.03398 | 0.004623 | 0.002621 | 0.007559 | 0.9131 | 0.9864 | 0.0008795 | 0.7433 |
| nearest_repair | 1 | 400 | 0.6234 | 0.6183 | 0.02559 | 0.008873 | 0.04561 | 0.3445 | 0.8146 | 0.0001556 | 0.5197 |
| nearest_repair | 2 | 400 | 0.6858 | 0.6576 | 0.02793 | 0.009369 | 0.04662 | 0.3771 | 0.7260 | 0.001554 | 0.5512 |
| discrete_radius | 0 | 374 | 0.02301 | 0.03254 | 0.004391 | 0.002530 | 0.007478 | 0.9111 | 0.9872 | 0.0008195 | 0.7312 |
| discrete_radius | 1 | 400 | 0.6232 | 0.6163 | 0.02556 | 0.008838 | 0.04563 | 0.3285 | 0.8134 | 0.0001832 | 0.5180 |
| discrete_radius | 2 | 399 | 0.6809 | 0.6786 | 0.02832 | 0.01813 | 0.04775 | 0.2717 | 0.6988 | 0.001686 | 0.5481 |

## Comparison To Existing Runs

Only completed runs with available best diagnostics are included.

| run | best_epoch | valid_iid_loss | valid_stress_loss | overall DeltaT RMSE | valid DeltaT RMSE | field corr |
|---|---:|---:|---:|---:|---:|---:|
| B96 latent96 legacy seed0 | 364 | 0.2279 | 0.3467 | 0.01933 | 0.02594 | 0.9004 |
| B96 latent96 nearest seed0 | 398 | 0.03604 | 0.05465 | 0.006715 | 0.004085 | 0.9782 |
| B96 latent96 nearest seed1 | 400 | 0.6581 | 0.6586 | 0.02656 | 0.009523 | 0.7963 |
| B96 latent96 discrete seed0 | 398 | 0.03799 | 0.05090 | 0.006766 | 0.004077 | 0.9785 |
| B64 m2width legacy seed0 | 391 | 0.2295 | 0.3586 | 0.01925 | 0.02708 | 0.8880 |
| B64 m2width nearest seed0 | 400 | 0.02543 | 0.04410 | 0.005405 | 0.003108 | 0.9840 |

## Assessment

- B88 `sample_shuffle` is trainable and produced complete e400 outputs for all
  seven configs.
- For model_seed0, both graph-repair policies strongly beat legacy. The best
  overall DeltaT RMSE is `0.004391` for pure `discrete_radius`; nearest repair
  is close at `0.004623`.
- B88 model_seed0 improves over B96 latent96 seed0 and is competitive with or
  better than B64 m2width nearest repair on overall RMSE, though B64 has slightly
  lower valid split RMSE.
- The result is not seed-stable. model_seed1 and model_seed2 fail for both
  nearest repair and discrete radius, with best epochs at or near epoch 400,
  low top-k overlap, and low field correlation.
- The evidence supports graph repair as necessary for the strong seed0 result,
  but does not yet support a stable model/config claim. The dominant current
  risk is initialization or optimizer path sensitivity, not B88 trainability.

## Next Step

Do not promote B88 sample-shuffle e400 as a stable baseline yet. The next
diagnostic should isolate seed sensitivity before adding more long runs: compare
fixed batch/graph seeds with model_seed sweeps, inspect early loss trajectory
for failed seeds, and check whether a longer warmup or smaller effective LR can
recover seed1/seed2 without changing model, decoder, loss, or objective.
