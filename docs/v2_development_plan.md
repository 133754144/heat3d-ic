# Heat3D v2 Development Plan

## Goal

Heat3D v2 should turn the v1 diagnostic baseline into a more capable and
controlled training system. The immediate objective is not to expand the dataset
or claim publication-ready results, but to address the training and model
limitations exposed by v1.

The starting dataset for v2 is `medium1024_gapA_full1024_v2`. Higher resolution
and larger datasets should wait until the model and optimizer stack are upgraded.

## Model Capacity Upgrade

The v1 smoke-scale configuration is too small for the observed field-shape and
hotspot errors. v2 should test larger neural-operator configurations:

- latent width from `16` to `64` or `128`;
- processor/message-passing steps from `2` to `4` or `6`;
- deeper MLP blocks where supported by the local model interface;
- explicit reporting of parameter count and training memory footprint.

These changes should be introduced through configuration, not hard-coded into
the training script.

## Optimizer And Training System

v2 should replace the hand-written full-batch gradient update path with a
standard optimizer stack:

- Optax Adam or AdamW;
- gradient clipping;
- optional weight decay;
- learning-rate schedule support;
- checkpoint and best-prediction selection by validation metric;
- clean resume / evaluation-only paths where practical.

The v1 best-valid export behavior should be retained: final predictions and
best-valid predictions must remain distinguishable.

## Configuration System

The v1 runner accumulated many CLI options for controlled diagnostics. v2 should
separate configuration into dataset, model, optimizer, loss, and run sections.

The target is a repeatable run protocol where the same config can drive local
smoke, SSH training, comparison, and post-run diagnostics without rewriting
large shell commands.

## Hotspot, Peak, And Field-Shape Supervision

v1 diagnostics show over-smoothed predictions and underestimated hotspot
amplitude. v2 should add explicit supervised diagnostics and, where useful,
loss terms for:

- peak temperature error;
- p95 / p99 absolute error;
- hotspot amplitude;
- top-k hotspot overlap;
- field variance ratio;
- spatial correlation;
- slice-level field-shape metrics.

These are supervised output-space diagnostics and losses. They are not PDE,
boundary-condition, or energy residual losses.

## Staged / Curriculum Loss

v1 identified a background-hotspot tradeoff. v2 should treat this as a training
schedule problem rather than one static loss-weight search:

- early background suppression for low-DeltaT calibration;
- mid-stage hotspot retention;
- high-bin fidelity terms for underpredicted hot regions;
- late-stage calibration and validation-based checkpoint selection.

Every staged loss should report final and best-valid metrics, error bins, and
condition-wise diagnostics.

## V2 Baseline Protocol

The frozen v1 best diagnostic run is the reference point:

```text
dataset: medium1024_gapA_full1024_v2
loss_mode: background_pseudo_negative
pseudo_negative_loss_type: relative_l1
pseudo_negative_weight: 0.10
background_relative_weight: 0.10
lr: 1e-2 constant
best_epoch: 33
```

v2 comparisons should report:

- zero-delta baseline;
- frozen v1 best diagnostic configuration;
- v2 final epoch;
- v2 best-valid epoch;
- split-wise and condition-wise metrics;
- error-bin diagnostics, especially `bin_0` background and high-bin
  underprediction;
- seed sensitivity before making any broader claim.

## Data Roadmap

Keep `medium1024_gapA_full1024_v2` as the v2 starting point. Do not expand to a
larger or higher-resolution dataset until the v2 training system shows stable
improvement over the frozen v1 baseline.

After the training system is upgraded, the next data steps can include:

- reviewing whether seed sensitivity persists;
- designing stronger held-out stack and BC splits;
- adding higher-resolution labels only when the optimizer/model stack is ready;
- keeping medium256 and medium1024 Gap-A as debug and ablation sets.

## Reporting Boundary

Use conservative language for v2 until a fixed protocol and multi-seed results
exist:

- diagnostic;
- controlled training;
- research-stage;
- v1 baseline comparison;
- benchmark-candidate preparation.

Do not claim formal benchmark completion, OOD generalization, production-ready
thermal simulation, or publication-ready model performance from v2 smoke runs.
