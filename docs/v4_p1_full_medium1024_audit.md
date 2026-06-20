# V4 P1 Full Medium1024 Audit

Read this file only for V4 full medium1024 range/OOD, final-probe amplitude,
or run-artifact provenance planning questions.

## Scope

This audit was run read-only on WSL2 after `git pull` on `research/v4`. It did
not train, evaluate a new checkpoint, start tmux, start a new run, or change
model/solver/loss/loader code.

Remote small outputs:

- `output/heat3d_v4_p1_full_medium1024_audit/training_path_audit.json`
- `output/heat3d_v4_p1_full_medium1024_audit/feature_manifest.json`
- `output/heat3d_v4_p1_full_medium1024_audit/amplitude_diagnostics.csv`

These outputs are ignored artifacts and are not tracked.

## Full Medium1024 Range Audit

The active V4 dataset path is
`data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2`.

| split | samples | k range | q max | top_h range | aspect | raw DeltaT max | normalized DeltaT max |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: |
| train | 768 | 2.125 to 290.989 | 1.585e8 | 420.399 to 1719.16 | 5.0 | 0.915 K | 20.21 |
| valid | 128 | 5.132 to 256.691 | 5.972e7 | 967.369 to 1710.39 | 5.0 | 0.858 K | 18.90 |
| final_probe | 10 | 0.668 to 423.366 | 1.940e8 | 450 to 3400 | 5.0 | 7.709 K | 175.21 |

BC relative temperature scalars are all 0 in train/valid/final-probe for the
audited data. Geometry extent/aspect ratio is not a final-probe OOD driver here:
all audited medium1024 train, valid, and final-probe samples have z extent
0.002 m and aspect ratio 5.0.

BC flag distribution differs sharply for final-probe under the current loader:

| flag | train mean | valid mean | final-probe mean |
| --- | ---: | ---: | ---: |
| `is_top` | 0.1667 | 0.1667 | 0.0 |
| `is_bottom` | 0.1667 | 0.1667 | 0.0 |
| `is_side` | 0.4375 | 0.4375 | 0.0 |
| `is_interior` | 0.3750 | 0.3750 | 1.0 |

This is not a geometry range issue. It is a final-probe metadata/BC-mask
compatibility issue: the final-probe metadata exposes boundary-region surfaces
without point indices, so the current V1 metadata loader does not recover the
top/bottom/side masks in the same way as medium1024 train/valid.

## Final-Probe OOD Conclusion

Relative to full medium1024 train, final-probe is still OOD.

OOD drivers:

- material/k: final-probe spans 0.668 to 423.366, outside train 2.125 to 290.989;
- q: final-probe max 1.94e8, above train max 1.585e8;
- BC scalar: final-probe top_h reaches 3400, above train max 1719.16;
- BC flags: final-probe flags collapse to all interior in the active loader path;
- amplitude: final-probe raw DeltaT max 7.709 K, far above train max 0.915 K.

Not an OOD driver in this audit:

- geometry extent/aspect ratio: final-probe matches train at z extent 0.002 m
  and aspect ratio 5.0.

Valid is in-distribution by these range checks.

## Amplitude Diagnostics

Amplitude diagnostics used existing V3 final-probe best predictions from:

`output/heat3d_v3_final_probe_S4mlp3discretebestFT2_e400_lr5e-6/best/predictions/s5_probe_predictions.npz`

No new inference was run. The audit only read existing predictions and labels.

| metric | min | median | max |
| --- | ---: | ---: | ---: |
| `RMSE_K` | 0.119 | 0.220 | 0.978 |
| `relRMSE_DeltaT` | 0.661 | 0.758 | 0.921 |
| `peak_error_K` | -7.013 | -2.258 | -0.957 |
| `mean_bias_K` | -0.291 | -0.130 | -0.024 |
| `scale_ratio` | 0.090 | 0.198 | 0.415 |
| `range_ratio` | 0.090 | 0.198 | 0.415 |
| `centered_corr` | 0.785 | 0.869 | 0.903 |
| `pred_deltaT_peak_K` | 0.404 | 0.645 | 0.733 |
| `label_deltaT_peak_K` | 1.638 | 2.903 | 7.709 |

Conclusion: the observed final-probe failure is primarily amplitude failure,
not shape failure. The model preserves field shape reasonably well
(`centered_corr` median 0.869) while predicting only about 20% of the label
DeltaT amplitude (`scale_ratio` and `range_ratio` medians 0.198) and
underpredicting peaks.

## Normalization Risk

The legacy normalization risk still holds and is stronger under the full audit:

- train raw DeltaT max is only 0.915 K, while final-probe reaches 7.709 K;
- normalized final-probe DeltaT reaches 175.21, far outside train max 20.21;
- k/q still use linear per-feature z-score, not log or physical-scale handling;
- BC flags are z-scored continuous channels, and final-probe BC masks are
  incompatible with train/valid under the current loader path;
- coordinate normalization is not the final-probe issue here, but it can still
  hide physical-scale changes in future datasets.

## Provenance Field Plan

No `run_registry.csv` field was added in this round. The registry/checker should
not be churned until a follow-up explicitly promotes these fields. Recommended
future result/provenance fields:

- `result_target_mode`
- `result_bridge_policy`
- `result_normalization_profile`
- `result_feature_manifest_hash`
- `result_dataset_split_hash`
- `result_final_probe_scale_ratio`
- `result_final_probe_range_ratio`
- `result_final_probe_centered_corr`
- `result_final_probe_mean_bias_K`
- `result_final_probe_peak_error_K`
