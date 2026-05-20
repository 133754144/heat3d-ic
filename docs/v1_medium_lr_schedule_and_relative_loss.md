# Heat3D v1 Medium LR Schedule and Safe Relative Background Loss

## Scope

This note documents optimization/loss-stage diagnostic tooling for Heat3D v1
medium256. It is not a formal benchmark, not an OOD generalization claim, not a
final model-performance conclusion, and not high-fidelity solver validation.

No PDE, boundary-condition, energy-balance, or flux residual loss is added here.

## Current SSH Observations

Recent controlled medium256 runs show:

- `background_l1_bias` with longer training and `lr=1e-3` can improve overall
  RMSE, MAE, p95, peak, and hotspot diagnostics against zero_delta.
- `hotspot_weight=0.0`, `0.02`, and `0.05` form a visible
  background-hotspot tradeoff.
- `bin_0` low-temperature-rise background still shows persistent
  overprediction.
- `bin_1` has improved clearly.
- `bin_3` / `bin_4` still show underprediction.
- Learning rate materially affects the result, so LR behavior should be made
  explicit and recorded in the run artifacts.

These are optimization/loss-stage diagnostics only.

## Upstream RIGNO Optimizer Audit

The upstream repository inspected locally was:

```text
/tmp/rigno-upstream-loss-audit
```

Relevant files:

```text
rigno/train.py
rigno/metrics.py
rigno/stepping.py
example.ipynb
README.md
```

The upstream training script uses `optax.inject_hyperparams(optax.adamw)` with
`weight_decay=1e-08`. The main training path in `rigno/train.py` does not use a
constant learning rate. It builds an Optax joined schedule:

```text
cosine_onecycle_schedule for the first 90% of transition steps
exponential_decay for the final 10% of transition steps
```

The default flags include:

```text
lr_init = 1e-05
lr_peak = 2e-04
lr_base = 1e-05
lr_lowr = 1e-06
```

The onecycle schedule has a small warmup fraction (`pct_start = 0.02`) and is
followed by final exponential decay. The example notebook uses AdamW with an
`optax.exponential_decay` schedule. The README and `train.py` also support
loading an old experiment through `--params`, restoring the checkpointed state
and model configuration as initialization for a new run.

Reference implication for Heat3D: the current medium training runner is a
controlled diagnostic runner using manual full-batch gradient descent. It does
not need to copy upstream optimizer code, but it should expose simple,
deterministic LR schedules so LR effects can be audited.

## Added LR Schedules

The runner now supports:

```text
--lr-schedule constant
--lr-schedule warmup_cosine
--lr-schedule two_stage
```

Default remains:

```text
--lr-schedule constant
```

Definitions:

```text
constant:
  lr_epoch = lr

two_stage:
  if epoch <= second_stage_epoch or second_stage_epoch <= 0:
      lr_epoch = lr
  else:
      lr_epoch = second_stage_lr

warmup_cosine:
  if warmup_epochs > 0 and epoch <= warmup_epochs:
      lr_epoch increases linearly from min_lr to lr
  else:
      lr_epoch follows cosine decay from lr to min_lr
```

`run_config.json`, `loss_summary.json`, and `epoch_history` record the schedule
and per-epoch LR.

## Safe Relative Background Loss

The runner now supports:

```text
--loss-mode background_l1_relative
```

It extends the existing `background_l1_bias` loss with a safe relative
background absolute-error term.

Definitions:

```text
raw_deltaT = T_true - T_ref
normalized_deltaT = (raw_deltaT - train_target_delta_mean) / train_target_delta_std
pred_raw_deltaT = pred_norm_deltaT * train_target_delta_std + train_target_delta_mean

background_mask =
  true_raw_deltaT <= quantile(true_raw_deltaT, background_quantile)

background_l1 =
  mean(abs(pred_raw_deltaT) over background_mask)

background_signed_bias =
  abs(mean(pred_raw_deltaT - true_raw_deltaT over background_mask))

background_overprediction =
  mean(relu(pred_raw_deltaT - true_raw_deltaT) over background_mask)
```

The safe relative denominator is:

```text
fixed:
  denom = max(abs(true_raw_deltaT), relative_floor)

p50:
  denom = max(abs(true_raw_deltaT),
              max(relative_floor, batch_abs_true_raw_deltaT_p50))

p75:
  denom = max(abs(true_raw_deltaT),
              max(relative_floor, batch_abs_true_raw_deltaT_p75))
```

The relative term is:

```text
background_relative_abs =
  mean(abs(pred_raw_deltaT - true_raw_deltaT) / denom over background_mask)
```

The combined loss is:

```text
loss = base_mse
     + background_l1_weight * background_l1
     + background_bias_weight * background_signed_bias
     + background_over_weight * background_overprediction
     + background_relative_weight * background_relative_abs
     + hotspot_weight * hotspot_retention_loss
```

First diagnostic setting:

```text
--background-l1-weight 1.0
--background-bias-weight 1.0
--background-over-weight 1.0
--background-relative-weight 0.05
--relative-floor 0.02
--relative-floor-mode fixed
--hotspot-weight 0.02
```

## Why Not Plain Relative Error

Plain relative error,

```text
abs(pred_raw_deltaT - true_raw_deltaT) / abs(true_raw_deltaT)
```

is unsafe for this task because `true_raw_deltaT` can be near zero in the
background. That would create very large loss values and unstable gradients
precisely in the low-temperature-rise bin. The safe denominator keeps the
relative term bounded by a floor and keeps it as a small auxiliary term rather
than replacing the absolute-error and bias losses.

## Recommended Server Ablations

First isolate LR behavior without relative loss:

```text
--lr-schedule two_stage
--second-stage-epoch 200
--second-stage-lr 1e-4
--loss-mode background_l1_bias
--hotspot-weight 0.02
```

Then isolate safe relative loss at constant LR:

```text
--lr-schedule constant
--loss-mode background_l1_relative
--background-relative-weight 0.05
--relative-floor 0.02
--relative-floor-mode fixed
--hotspot-weight 0.02
```

For both runs, inspect comparison, run-summary analysis, and error-binning
diagnostics. The primary diagnostic question is whether `bin_0`
overprediction decreases without reintroducing high-DeltaT underprediction or
hotspot degradation.
