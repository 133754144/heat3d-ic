# V4 P3c-4 Pilot64 Closeout

Read this file only for P3c pilot64 closeout, P3c-5 handoff, or split/QC
decisions.

## Scope

P3c-4 closes the 64-sample random-block pilot. It does not change solver
numerics, model code, loss code, or training configuration.

Generated artifact roots:

- `data/heat3d_v4_p3c_smoke16_v3/`
- `output/heat3d_v4_p3c_smoke16_v3/`
- `output/heat3d_v4_p3c_smoke16_v3_train50/`
- `data/heat3d_v4_p3c_pilot64_v0/`
- `output/heat3d_v4_p3c_pilot64_v0/`
- `data/heat3d_v4_p3c_pilot64_v1/`
- `output/heat3d_v4_p3c_pilot64_v1/`

## Smoke16 Training Check

`smoke16_v3` was generated with solver-control-volume weighted q calibration.
The local 50-epoch smoke training completed without NaN/Inf:

- `grad_finite=true`
- `loss_summary_finite=true`
- `predictions_finite=true`
- final prediction count: 16
- best prediction count: 16
- best epoch: 13
- best valid base MSE: `0.16725866496562958`
- final valid base MSE: `0.18019670248031616`

This is a path sanity check only, not a model-quality claim.

## Pilot64 Solver And Q Audit

`pilot64_v0` and `pilot64_v1` use the same generated scenes and solver labels;
`pilot64_v1` fixes split metadata and adds split/review audit.

`pilot64_v1` audit:

- sample count: 64
- solver pass rate: 1.0
- failure count: 0
- max absolute energy-balance residual: `2.3376856006507296e-12`
- max bottom Dirichlet error: `0.0`
- max absolute q total power error: `8.881784197001252e-16 W`
- max q power on boundary: `0.0 W`
- q boundary violation count: 0
- q side-boundary violation count: 0
- q deposited boundary node count: 0

Conclusion: P3c-4 solver/q/boundary audit passes for pilot64.

## Split Cleanup

`pilot64_v0` used index-based metadata split: first 48 samples were train and
last 16 samples were test. This is biased because q family, cooling, diag3, and
DeltaT patterns are generated in deterministic sequence.

`pilot64_v1` uses `deterministic_stratified_random_v0`: split assignment happens
after solver audit, with fixed seed, using these fields:

- `k_mode`
- `diag3_policy`
- `q_family`
- `cooling_regime`
- `DeltaT_bin`
- `high_deltaT_triage`
- `dataset_action`

Split counts:

| split | count |
| --- | ---: |
| train | 48 |
| test | 16 |

Field coverage summary:

| field | train | test | note |
| --- | --- | --- | --- |
| `k_mode` | scalar 38, diag3 10 | scalar 13, diag3 3 | covered |
| `diag3_policy` | scalar 38, mild 8, hbm_like_strong 2 | scalar 13, mild 2, hbm_like_strong 1 | covered |
| `cooling_regime` | nominal 16, strong 16, weak 16 | nominal 5, strong 5, weak 6 | covered |
| `dataset_action` | keep 46, review 2 | keep 15, review 1 | covered |
| `high_deltaT_triage` | not_high 46, high_unclassified 2 | not_high 15, high_unclassified 1 | covered |
| `q_family` | all 7 families present | all 7 families present | covered |
| `DeltaT_bin` | hard 21, low 1, nominal 20, reject_high 2, reject_low 1, review_high 3 | hard 6, low 1, nominal 7, reject_high 1, review_high 1 | `reject_low` total count is 1, so it cannot appear in both splits |

Conclusion: `pilot64_v1` removes the index split bias and is the preferred
pilot64 artifact for P3c-5 planning.

## Review Samples

`pilot64_v0` and `pilot64_v1` have the same three review samples. They are not
solver failures and not boundary/q-policy failures; they are high-DeltaT cases
that need parameter-space review before production-scale generation.

| sample | v1 split | DeltaT peak K | q family | reason tags | conclusion |
| --- | --- | ---: | --- | --- | --- |
| `sample_019` | train | 24.0443 | `dual_z_q_density` | high DeltaT, q power consistent, no boundary q, low-k/source overlap 0.5625 | review |
| `sample_039` | train | 17.6926 | `weak_background_hotspot_q_density` | high DeltaT, q power consistent, no boundary q, weak cooling | review |
| `sample_043` | test | 54.7204 | `multi_block_q_density` | high DeltaT, q power consistent, no boundary q, low-k/source overlap 0.9590, high integrated power 5.8645 W | review |

No sample is marked reject in P3c-4 because solver status, q power consistency,
and boundary-deposition checks pass. The three review samples should remain
flagged as high-amplitude pilot cases, not silently promoted into production QC.

## Random Split Assessment

If all samples are generated from one unified distribution and no stress split
is being defined, a fixed-seed random train/test split is acceptable. For small
pilot sets, plain random split can still miss rare strata, so P3c should use
deterministic stratified random split plus a post-split distribution audit. A
category with only one sample cannot be represented in both train and test; the
audit must report that explicitly.

## P3c-5 Recommendation

Proceed to P3c-5 with `pilot64_v1` as the handoff artifact, under two
conditions:

- Treat the three review samples as parameter-space review cases, not as
  automatic pass/fail for the dataset family.
- Use P3c-5 to select a validation subset and calibrate DeltaT QC policy before
  any 1024-sample production candidate.
