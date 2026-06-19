# V4 Metrics Contract

Read this file only for V4 metrics-profile, checkpoint-selection, registry, or
evaluation-audit questions.

The V4 metrics contract is `configs/heat3d_v4/metrics_v0.json`. It records the
metric names, allowed checkpoint-selection metric, and aggregation policy used
by V4 registry-generated configs.

## Checkpoint Selection

The default and currently allowed checkpoint-selection metric is
`valid_base_mse`.

`valid_base_mse` is measured in the normalized validation objective space. It is
kept separate from raw DeltaT reports so targeted losses, physical-scale
diagnostics, or final-probe summaries do not silently change checkpoint
selection.

## Performance Metrics

`mse`, `rmse`, and `mae` are overall model-performance metrics. They are useful
for comparing model quality across splits or groups when computed with the V4
aggregation rule, but they do not replace `valid_base_mse` for checkpoint
selection unless a future metrics contract explicitly allows that.

## Raw DeltaT Metrics

Raw DeltaT metrics report physical-scale error after recovery to DeltaT or
temperature-rise units. They are useful for reporting and interpretation, but
they are not the default checkpoint-selection metric.

Normalized validation and raw DeltaT answer different questions:

- normalized validation tracks the training/selection objective consistently;
- raw DeltaT reports physical-scale error for readers and diagnostics.

## Split And Background Diagnostics

V4 result audits keep split-aware scalar diagnostics separate from selection.
`valid_iid` and `valid_stress` summarize in-distribution and stress behavior,
while error percentages explain relative movement. Low-DeltaT background
columns such as `bin_0_over_ratio` and `le_0p05_over_ratio` track the V2/V3
background overprediction failure mode.

## Final-Probe, OOD, Region, And Diagnostic Metrics

Final-probe and OOD metrics report stress behavior and extrapolation behavior.
Region and hotspot metrics report localized error modes. Diagnostic metrics
such as field-shape, graph-coverage, and repair-count measures explain
mechanisms.

These metrics may support analysis, failure triage, or paper-facing tables, but
they cannot replace default checkpoint selection unless a future contract
explicitly promotes them.

## Aggregation

All V4 reported metrics should be computed per sample first. Split-level and
group-level reports then summarize those per-sample values with mean, median,
and standard deviation.

A single global flattened error over all nodes and samples is not a substitute
for cross-sample statistics.
