# Heat3D v1 Diagnostic Closeout

## Scope

Heat3D v1 is frozen as a diagnostic / research-stage baseline. It is not a
formal benchmark, not a publication-ready model result, and not evidence that
OOD generalization or arbitrary 3D IC thermal prediction is solved.

The purpose of v1 is to provide a reproducible reference for dataset generation,
controlled training, loss diagnostics, and error analysis before starting the v2
training-system work.

## Completed Work

The v1 branch now contains the main pieces needed for a diagnostic baseline:

- medium256 physics-label dataset pipeline;
- medium1024 Gap-A planning, generation-ready manifest, and
  `medium1024_gapA_full1024_v2` generation path;
- diversity repair for true `q_field`, `k_field`, and `temperature`
  duplication;
- metadata, loader-stage compatibility, generated-subset checkers, and label
  diagnostics;
- controlled training/export runner with progress logging, compact stdout, loss
  schedules, best-valid selection, and optional best-prediction export;
- baseline comparison, run summary, error-bin diagnostics, condition-wise
  diagnostics, diversity diagnostics, and multi-seed summary tooling;
- report package generation for dataset statistics, loss curves, ablations, and
  representative prediction figures.

## Current Best Diagnostic Run

The current best v1 diagnostic configuration is:

```text
dataset: medium1024_gapA_full1024_v2
loss_mode: background_pseudo_negative
pseudo_negative_loss_type: relative_l1
pseudo_negative_weight: 0.10
background_relative_weight: 0.10
lr: 1e-2 constant
best_epoch: 33
best_overall_RMSE / MAE: 3.94142446e-02 / 2.46786651e-02
best_valid_RMSE / MAE: 2.73560372e-02 / 2.30636740e-02
bin_0_bias: 1.89761732e-02
bin_0_RMSE_rel / MAE_rel: +256.56% / +404.44%
bin_0_over_ratio: 1.0
```

These numbers are diagnostic references only. They should be used to compare v2
changes against the frozen v1 behavior, not as a formal performance claim.

## Main Findings

The v1 runner can learn nontrivial mid- and high-DeltaT temperature structure
better than the zero-delta baseline in the current diagnostic setting.
Condition-wise and error-bin analysis show that loss design materially changes
the tradeoff between low-background calibration and hotspot retention.

The most useful v1 loss variant is the relative-L1 pseudo-negative background
penalty. It reduces low-DeltaT background bias more effectively than the earlier
MSE pseudo-negative penalty and simple L1 variant.

Best-valid export is necessary. In medium1024 Gap-A probes, validation loss can
reach its lowest value before the final epoch while train loss continues to
decrease, so final-epoch predictions and best-valid predictions must be reported
separately.

## Limitations

The main unresolved issue is systematic low-DeltaT `bin_0` background
overprediction. Even the best current v1 diagnostic run keeps
`bin_0_over_ratio=1.0`, meaning the sign of the background error remains
consistently positive.

The current runner is intentionally smoke-scale and simplified:

- small model capacity;
- latent size and message-passing depth are too limited for the next stage;
- full-batch hand-written gradient descent instead of AdamW / Optax training;
- background-dominated loss behavior;
- insufficient hotspot, peak, and field-shape supervision;
- coarse `8x8x6` resolution;
- prediction fields are over-smoothed, with peak and hotspot amplitude still
  underestimated.

The v1 result should therefore be read as a controlled diagnostic baseline, not
as a final model architecture or training recipe.

## Freeze Decision

Freeze v1 at this point and use it as the baseline/reference for v2. Further
work should move to a new v2 development branch and focus on the training
system, model capacity, optimizer, and field-shape diagnostics rather than
continuing to tune v1 loss weights.
