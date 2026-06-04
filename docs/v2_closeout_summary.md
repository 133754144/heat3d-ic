# Heat3D v2 Closeout Summary

Heat3D v2 closes as a research-stage reproducible baseline. It is not a formal
benchmark, and the remaining `valid_iid` error is still approximately 70%.

## Stable V2 Baseline

- Dataset protocol: `medium1024_gapA_full1024_v2`.
- Default split map: `configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json`.
- Primary validation: `valid_iid`.
- Diagnostic split: `valid_stress`.
- Controlled-training anchor: M2 B96, node/edge latent `128/128`, processor
  steps `6`, MLP hidden layers `2`, AdamW warmup-cosine, e400.

## Main V2 Outputs

- YAML-driven, reproducible controlled training and command planning.
- B96/B48 mini-batch training with deterministic audit records.
- Final-vs-best prediction export.
- Split-aware, field-shape, condition, error-bin, and background/bin0
  diagnostics.
- Memorization, input-target representation, boundary fallback, and graph-path
  audits.
- Canonical `raw_deltaT_rmse` and `raw_deltaT_true_mse` diagnostics, with
  `raw_deltaT_mse` retained only as a compatibility alias.

## Closeout Findings

- B48, M2.5, and larger-capacity runs are diagnostic history, not the default
  mainline.
- Increasing capacity alone did not reduce error near the 20% target.
- One-sample RIGNO memorization remains around 42% error.
- A pointwise MLP fits the 1/4-sample cases below 20%, reducing the likelihood
  that target recovery alone is the primary blocker.
- The leading bottleneck is the RIGNO graph coverage and graph-to-output path.

V2 stops before high-risk graph/model-path changes. Those changes start in v3
from the merged `main` baseline.
