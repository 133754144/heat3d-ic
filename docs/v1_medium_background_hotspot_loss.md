# Heat3D v1 Medium Background-Hotspot Loss

## Purpose

This note documents the first loss-stage experiment for Heat3D v1 medium256.
Recent diagnostics show a specific pattern: hotspot and peak metrics can improve
against zero_delta, while low-DeltaT background regions are overpredicted. That
background bias can make mean recovered-temperature RMSE/MAE worse even when
high-DeltaT behavior improves.

The `background_hotspot` loss mode is a controlled training diagnostic for that
loss behavior. It is not a formal benchmark, not a model-performance
conclusion, not OOD generalization evidence, and not high-fidelity solver
validation.

## Loss Mode

The training export runner now supports:

```text
--loss-mode mse
--loss-mode background_hotspot
```

Default remains:

```text
--loss-mode mse
```

The default MSE mode keeps the existing supervised normalized DeltaT loss and
does not change `predictions.npz`.

## Background-Hotspot Objective

For `background_hotspot`, the objective is:

```text
loss = base_mse
     + background_weight * background_penalty
     + hotspot_weight * hotspot_retention_loss
```

Initial parameters:

```text
--background-quantile 0.50
--hotspot-quantile 0.90
--background-weight 0.5
--hotspot-weight 1.0
```

Definitions:

- `base_mse`: existing supervised MSE on normalized DeltaT
- `background_mask`: true DeltaT at or below the batch/group background quantile
- `background_penalty`: penalty for predicting nonzero DeltaT in background
  regions
- `hotspot_mask`: true DeltaT at or above the batch/group hotspot quantile
- `hotspot_retention_loss`: prediction error in high-DeltaT regions

The runner computes all loss terms in normalized DeltaT training space. The
background zero target is raw `DeltaT = 0` converted into normalized DeltaT
using train-only normalization stats. This avoids silently mixing raw DeltaT and
normalized training-space losses.

## Motivation From Error Bins

The error-binning diagnostics target the same pattern:

- low DeltaT bins, especially `bin_0` / `bin_1`, can show positive signed bias
  and worse trained RMSE/MAE than zero_delta
- high DeltaT bins can improve against zero_delta

`background_hotspot` tests whether a supervised loss can reduce low-DeltaT
overprediction while retaining high-DeltaT / hotspot behavior.

## Recorded Artifacts

The runner records loss settings in:

```text
run_config.json
loss_summary.json
```

`loss_summary.json` also records final loss components and report-epoch
component fields in `epoch_history`.

`predictions.npz` remains unchanged: recovered-temperature predictions keyed by
sample id.

## Scope Boundary

This is not a physical/PDE loss. It does not add:

- PDE residual loss
- boundary-condition residual loss
- energy-balance loss
- flux/interface residuals

Those may be considered later after the supervised loss behavior is understood.

## First Server Experiment

The first full medium256 server experiment should only test whether the loss
reduces background overprediction while preserving hotspot advantages. Treat it
as a diagnostic loss-stage experiment, not as a benchmark.
