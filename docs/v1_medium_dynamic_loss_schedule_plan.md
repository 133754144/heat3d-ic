# Heat3D v1 Medium Dynamic Loss Schedule Plan

## Scope

This note documents optimization/loss-stage diagnostic tooling for Heat3D v1
medium256. It is not a formal benchmark, not an OOD generalization claim, not a
final model-performance conclusion, and not high-fidelity solver validation.

No PDE, boundary-condition, energy-balance, or flux residual loss is added here.

## Current Diagnostic Findings

The current balanced reference run is:

```text
constant lr=1e-3
loss-mode background_l1_bias
hotspot_weight=0.02
```

It is the present background/hotspot tradeoff baseline for medium256
optimization diagnostics.

The `two_stage` LR schedule ran correctly, but did not clearly improve over the
constant `lr=1e-3` / `hotspot_weight=0.02` setting. This suggests that simply
lowering the learning rate late in training is not the main bottleneck.

The `background_l1_relative` / `background_relative_weight=0.05` run reduced
the low-DeltaT `bin_0` / `bin_1` positive background bias and improved overall
MAE. However, it also increased high-temperature `bin_3` / `bin_4`
underprediction and slightly weakened peak/hotspot diagnostics.

The next diagnostic question is whether a curriculum can keep the useful early
background correction while reducing late pressure on high-temperature regions.

## Proposed Curriculum

The staged loss keeps the supervised target and existing raw-background
diagnostics, but changes selected weights over training:

```text
loss = base_mse
     + background_l1_weight(t) * background_l1
     + background_bias_weight(t) * background_signed_bias
     + background_over_weight(t) * background_overprediction
     + background_relative_weight(t) * background_relative_abs
     + hotspot_weight(t) * hotspot_retention_loss
```

The first recommended server experiment uses `linear_anneal`:

```text
epoch 1 -> 200:
  background_relative_weight: 0.05 -> 0.01
  hotspot_weight:              0.00 -> 0.05

held constant:
  background_l1_weight:        1.0
  background_bias_weight:      1.0
  background_over_weight:      1.0
```

Rationale:

- Early `background_relative_weight=0.05` has already shown useful correction
  for low-temperature background bins.
- Late `background_relative_weight=0.01` should reduce continued pressure that
  can worsen high-DeltaT underprediction.
- Increasing `hotspot_weight` from `0.00` to `0.05` tests whether peak/hotspot
  and `bin_3` / `bin_4` behavior can recover after the early background phase.
- Linear annealing is preferred before abrupt two-phase changes because it
  avoids a discontinuous objective jump.

## Added Tooling

The training runner supports:

```text
--loss-weight-schedule constant
--loss-weight-schedule two_phase
--loss-weight-schedule linear_anneal
```

`constant` preserves previous behavior. `two_phase` switches from start weights
to end weights after `--loss-transition-epoch`. `linear_anneal` interpolates
from start weights to end weights through `--loss-transition-epoch`, then holds
the end values.

The runner records the actual epoch weights in `epoch_history` and
`loss_weight_history`, while `run_config.json` and `loss_summary.json` record
the schedule parameters.

The terminal log now defaults to `--log-mode compact` so report epochs show the
core optimization diagnostics without dropping any JSON fields. Use
`--log-mode full` for the longer debug line and `--log-mode quiet` to suppress
per-epoch report lines.

## Recommended Server Command Shape

```text
--loss-mode background_l1_relative
--loss-weight-schedule linear_anneal
--loss-transition-epoch 200
--background-relative-weight-start 0.05
--background-relative-weight-end 0.01
--hotspot-weight-start 0.00
--hotspot-weight-end 0.05
--background-l1-weight 1.0
--background-bias-weight 1.0
--background-over-weight 1.0
```

After training, inspect comparison, run-summary analysis, and error-binning
diagnostics. The diagnostic target is lower `bin_0` / `bin_1` overprediction
without worsening `bin_3` / `bin_4`, peak, or hotspot metrics.
