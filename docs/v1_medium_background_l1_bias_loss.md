# Heat3D v1 Medium Background L1 Bias Loss

## Scope

This note documents a loss-stage diagnostic for Heat3D v1 medium256 controlled
training. It is not a formal benchmark, not an OOD generalization claim, not a
final model-performance conclusion, and not a high-fidelity solver validation.

No PDE, boundary-condition, energy-balance, or flux residual loss is added here.

## Current Heat3D Loss Before This Change

The v1 medium training export runner predicts normalized temperature rise:

```text
raw_deltaT = T_true - T_ref
normalized_deltaT = (raw_deltaT - train_target_delta_mean) / train_target_delta_std
pred_raw_deltaT = pred_norm_deltaT * train_target_delta_std + train_target_delta_mean
T_pred = T_ref + pred_raw_deltaT
```

Default `mse` mode:

```text
base_mse = mean((pred_norm_deltaT - true_norm_deltaT)^2)
loss = base_mse
```

Existing `background_hotspot` mode:

```text
background_mask = true_raw_deltaT <= quantile(true_raw_deltaT, background_quantile)
hotspot_mask = true_raw_deltaT >= quantile(true_raw_deltaT, hotspot_quantile)

background_penalty = mean(pred_raw_deltaT^2 over background_mask)
hotspot_retention_loss = mean((pred_norm_deltaT - true_norm_deltaT)^2 over hotspot_mask)

loss = base_mse
     + background_weight * background_penalty
     + hotspot_weight * hotspot_retention_loss
```

## Why Raw Background MSE Was Not Enough

The medium256 loss-stage runs showed:

- MSE baseline can improve peak, p95, and hotspot-location diagnostics while
  leaving mean RMSE/MAE worse than zero_delta.
- Error-binning shows systematic low-DeltaT overprediction in `bin_0` and
  `bin_1`.
- `background_hotspot` raw background MSE penalty improved this only weakly.
- The `bg5.0_hot0.0` diagnostic showed that disabling hotspot retention does
  not remove the background bias, so the hotspot term is not the main cause.

Raw background MSE penalizes magnitude, but the observed failure is more
specific: a low-DeltaT positive prediction floor. The next diagnostic loss
therefore adds terms that directly target background absolute magnitude,
signed bias, and one-sided overprediction.

## New `background_l1_bias` Loss

The runner now supports:

```text
--loss-mode background_l1_bias
```

Definition:

```text
base_mse = mean((pred_norm_deltaT - true_norm_deltaT)^2)

background_mask = true_raw_deltaT <= quantile(true_raw_deltaT, background_quantile)
hotspot_mask = true_raw_deltaT >= quantile(true_raw_deltaT, hotspot_quantile)

background_l1 =
  mean(abs(pred_raw_deltaT) over background_mask)

background_signed_bias =
  abs(mean(pred_raw_deltaT - true_raw_deltaT over background_mask))

background_overprediction =
  mean(relu(pred_raw_deltaT - true_raw_deltaT) over background_mask)

hotspot_retention_loss =
  mean((pred_norm_deltaT - true_norm_deltaT)^2 over hotspot_mask)

loss = base_mse
     + background_l1_weight * background_l1
     + background_bias_weight * background_signed_bias
     + background_over_weight * background_overprediction
     + hotspot_weight * hotspot_retention_loss
```

Recommended first test:

```text
--background-quantile 0.50
--hotspot-quantile 0.90
--background-l1-weight 1.0
--background-bias-weight 1.0
--background-over-weight 1.0
--hotspot-weight 0.0
```

The first diagnostic keeps `hotspot_weight=0.0` to isolate whether the
background floor can be reduced. If hotspot diagnostics regress sharply after
that, a follow-up can try `--hotspot-weight 0.1`.

## Relation To Upstream RIGNO

The upstream RIGNO repository uses a standard supervised operator-learning
training loss. In `rigno/train.py`, the default `loss_fn` is `mse_loss`. In
`rigno/metrics.py`, `mse_loss(gtr, prd)` is:

```text
mean((prd - gtr)^2)
```

The steppers in `rigno/stepping.py` produce normalized targets and predictions
for the selected training interpretation: output, residual, or time derivative.
Evaluation also reports MSE and relative Lp metrics.

No upstream background/hotspot weighting or PDE residual loss was used as a
template here. The upstream behavior is the supervised baseline reference. The
Heat3D `background_l1_bias` loss is a task-specific supervised correction for
low-temperature-rise background overprediction.

## Recorded Artifacts

`run_config.json` and `loss_summary.json` record:

```text
loss_mode
background_quantile
hotspot_quantile
background_weight
hotspot_weight
background_l1_weight
background_bias_weight
background_over_weight
```

`loss_summary.json` also records final loss components and report-epoch
`epoch_history` fields for:

```text
base_mse
background_penalty
background_l1
background_signed_bias_loss
background_overprediction_loss
hotspot_retention_loss
bg_pred_raw_mean
bg_signed_bias
bg_abs_mean
hotspot_raw_mae
```

`predictions.npz` remains unchanged: recovered-temperature predictions keyed by
sample id.

## Server Command

Run this from the SSH Git-only checkout after pulling the research branch:

```bash
python scripts/run_heat3d_v1_medium_controlled_training_export.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium256_v2 \
  --epochs 50 \
  --lr 1e-5 \
  --seed 0 \
  --report-every 5 \
  --loss-mode background_l1_bias \
  --background-quantile 0.50 \
  --hotspot-quantile 0.90 \
  --background-l1-weight 1.0 \
  --background-bias-weight 1.0 \
  --background-over-weight 1.0 \
  --hotspot-weight 0.0 \
  --output-dir output/heat3d_v1_medium_runs/medium256_e050_bg_l1_bias_hot0.0_seed0 \
  --save-predictions
```

Then run comparison, run-summary analysis, and error-binning diagnostics on the
resulting run directory.
