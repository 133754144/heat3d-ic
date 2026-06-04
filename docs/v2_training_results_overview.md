# Heat3D v2 Training Results Overview

Scope: research-stage diagnostic summary for `medium1024_gapA_full1024_v2`
using `medium1024_gapA_stratified_split_seed0.json`. This is not a formal
benchmark.

## Closeout Anchor

The stable v2 controlled-training anchor is M2 B96:

- model: node/edge latent `128/128`, processor steps `6`, MLP hidden layers `2`;
- batch: `96`;
- optimizer/schedule: AdamW with warmup-cosine;
- run length: `e400`;
- primary validation: `valid_iid`;
- diagnostic split: `valid_stress`.

The remaining IID error is still approximately 70%. V2 therefore closes as a
reproducible training and diagnostics baseline, not as a solved thermal-field
surrogate.

## Confirmed Conclusions

| finding | closeout conclusion |
|---|---|
| controlled training | YAML configuration, deterministic audit hooks, final/best export, and split-aware reporting are stable enough to carry into `main`. |
| diagnostics | Field-shape, condition, error-bin, background/bin0, and final-vs-best diagnostics are available for research-stage comparisons. |
| data protocol | Medium1024 Gap-A full1024 v2 with the stratified split is the default v2 protocol. |
| capacity history | B48, M2.5, and larger-capacity runs remain diagnostic history rather than the default mainline. |
| memorization | One-sample RIGNO fitting remains around 42% error, while the pointwise MLP fits the 1/4-sample cases below 20%. |
| primary bottleneck | Evidence points to graph coverage and the graph-to-output path, rather than simply insufficient width or training duration. |

V2 should not continue high-risk model-path changes. Those experiments belong
to the v3 graph-coverage work described in `v3_development_starting_plan.md`.
