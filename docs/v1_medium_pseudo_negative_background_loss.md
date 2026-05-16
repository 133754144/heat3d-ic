# Heat3D v1 Pseudo-Negative Background Loss

This note documents a loss-stage diagnostic extension for Heat3D v1 medium
training. It is not contrastive learning, not hard negative mining, not a
physics/PDE residual loss, and not a formal benchmark claim.

## Motivation

`medium1024_gapA_full1024_v2` e50/e100 diagnostics show that the current main
failure mode is persistent positive bias in low-DeltaT `bin_0` background
regions. Increasing `background_relative_weight` from `0.05` to `0.10` improved
overall metrics and reduced `bin_0` bias slightly, but `bin_0`
overprediction-ratio remained `1.0`. This suggests the background term helps
but does not directly remove systematic positive output bias in the most
near-zero regions.

## Loss Mode

The new optional mode is:

```bash
--loss-mode background_pseudo_negative
```

Default behavior remains `--loss-mode mse`. Existing `mse`,
`background_hotspot`, `background_l1_bias`, and `background_l1_relative` modes
are unchanged.

## High-Confidence Pseudo-Negative Mask

The mask is defined in raw DeltaT space:

```text
pseudo_negative_mask =
    true_raw_deltaT <= quantile(true_raw_deltaT, pseudo_negative_quantile)
```

If `--pseudo-negative-delta-threshold` is set, the mask also requires:

```text
true_raw_deltaT <= pseudo_negative_delta_threshold
```

This deliberately uses true raw DeltaT, not normalized DeltaT. Normalized zero
is the train-set mean raw DeltaT and is not equivalent to physical
`raw_deltaT = 0`.

The mask is intended to capture high-confidence near-zero background points. It
does not label all low-power samples as negatives and does not assume all weak
positive temperature rise should be pushed to zero.

## Overprediction-Only Penalty

The model output is first recovered into raw DeltaT:

```text
pred_raw_deltaT = pred_norm_deltaT * train_target_delta_std
                + train_target_delta_mean
```

The pseudo-negative penalty is a squared positive hinge:

```text
pseudo_negative_over_loss =
    mean(relu(pred_raw_deltaT - true_raw_deltaT - margin)^2
         over pseudo_negative_mask)
```

The full `background_pseudo_negative` loss keeps the existing
`background_l1_relative` components and adds the overprediction-only term:

```text
loss = base_mse
     + background_l1_weight * background_l1
     + background_bias_weight * background_signed_bias
     + background_over_weight * background_overprediction
     + background_relative_weight * background_relative_abs
     + pseudo_negative_weight * pseudo_negative_over_loss
     + hotspot_weight * hotspot_retention_loss
```

This only penalizes positive background error above the margin. It does not
penalize negative error in the pseudo-negative region.

## Parameters

- `--pseudo-negative-quantile`, default `0.25`
- `--pseudo-negative-delta-threshold`, optional
- `--pseudo-negative-weight`, default `0.1`
- `--pseudo-negative-over-margin`, default `0.0`
- `--pseudo-negative-min-count`, default `1`

If too few points satisfy the mask, the pseudo-negative term safely falls back
to zero for that group.

## Diagnostics

The runner records pseudo-negative diagnostics in `run_config.json`,
`loss_summary.json`, and `epoch_history`:

- `train_pseudo_negative_count`
- `valid_pseudo_negative_count`
- `train_pseudo_negative_over_loss`
- `valid_pseudo_negative_over_loss`
- `train_pseudo_negative_bias`
- `valid_pseudo_negative_bias`
- `train_pseudo_negative_over_ratio`
- `valid_pseudo_negative_over_ratio`

Compact report epochs include:

- `valid_pn_bias`
- `valid_pn_over`
- `valid_pn_over_ratio`

The intended next diagnostic question is whether this reduces `bin_0` signed
bias and overprediction ratio without worsening high-DeltaT bins. It should be
evaluated with existing comparison, run-summary, error-bin, and condition-wise
diagnostic tooling before considering longer e200/e300 runs.
