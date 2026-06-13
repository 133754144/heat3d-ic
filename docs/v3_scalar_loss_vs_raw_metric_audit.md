# Heat3D v3 Scalar Loss vs Raw Metric Audit

## Scope

This note explains why the v3 long-run ranking can differ between scalar
validation loss and raw DeltaT mechanism metrics. It is an audit note only:
no model, decoder, loss, objective, or training behavior is changed here.

## Metric Spaces

The controlled runner's `valid_iid` and `valid_stress` scalar losses are
computed in the configured training target space, currently
`normalized_deltaT`, with the same point/sample aggregation used by the
training loss.

The mechanism diagnostics recover raw DeltaT fields and then compute
field-level metrics such as RMSE, z-score RMSE, peak relative error,
top-k overlap, centered correlation, and amplitude ratio. These metrics
measure different behavior:

- scalar loss emphasizes normalized pointwise MSE under the runner's
  aggregation;
- raw RMSE/MAE emphasizes absolute DeltaT error after recovery;
- z-score RMSE and centered correlation emphasize shape independent of scale;
- top-k overlap and peak relative error emphasize hotspot localization and
  peak recovery.

## Current Interpretation

Current long-run diagnostics show a mismatch:

- current best scalar reference: B6 best;
- current best raw mechanism reference: S3 final;
- S2/S3/S1 can rank better on some recovered-field metrics while ranking worse
  on scalar validation loss.

This is not inherently contradictory. It can happen when normalized MSE is
won by a subset of high-weight or high-normalization samples, while another
run better preserves raw amplitude, spatial shape, or hotspots across the
field-level diagnostics.

## Next Decisions

The paired per-sample mismatch audit should decide which explanation is most
likely:

- if the gap is concentrated in a small number of hard samples, consider
  hard-sample or condition weighting later in P7;
- if the gap comes from normalization or aggregation, consider objective and
  evaluation alignment later in P7;
- if the gap comes from local structure or hotspot failures, continue P3/P5
  decoder and local-path investigation before changing the objective.

Until S4 and paired mismatch results are reviewed, do not treat either scalar
loss or raw mechanism metrics as the sole selection criterion.
