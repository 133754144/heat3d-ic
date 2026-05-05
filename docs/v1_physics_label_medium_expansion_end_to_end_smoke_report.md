# Heat3D v1 Medium Expansion End-to-End Smoke Report

## Stage Purpose

This report records the 24-sample `v1_multilayer_bc_eq_physics_label_medium_expansion_v2`
end-to-end smoke. The goal is to verify that the medium expansion physics-label
subset can flow through the existing v1 supervised route:

`relative BC features + zero_delta_u_bridge + normalized DeltaT target`

This is a research reference diagnostic. It is not a formal benchmark, not
model-performance evidence, not OOD generalization evidence, and not a
high-fidelity solver result.

## Subset Summary

- subset: `data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_expansion_v2`
- sample count: 24
- split counts: train 16, valid 4, test_id 2, test_ood_bc_candidate 1, test_ood_stack_candidate 1
- label source: reference solver v2 research-reference labels
- required label files: every sample has `temperature.npy` and `label_meta.json`

Coverage summary from the medium expansion checker:

- heat source patterns: centered 8, shifted 3, edge/corner 3, two-hotspot 3, broad-block 3, dual-active-layer 4
- k region modes: layerwise 11, blockwise 7, diagonal anisotropic diagnostic 3, interposer equivalent 3
- stack templates: baseline 11, compact 5, dual-active 4, interposer-like 3, held-out interposer candidate 1
- BC categories: nominal 14, low top HTC 4, high top HTC 5, held-out top HTC candidate 1

## Source And Solver Diagnostics

The medium expansion generator uses region-first source definitions with
volume-fraction source assignment and fixed-density `q_policy`.

Checker summary:

- `source_missed_count`: 0
- max `integrated_q_power_relative_error`: `4.319262e-16`
- max solver `residual_norm`: `6.538052e-16`
- max `bottom_dirichlet_error`: `0.0`
- medium expansion generation smoke: pass

These checks confirm source bookkeeping and solver metadata consistency at
smoke level. They are not formal energy-balance or grid-convergence evidence.

## Label Diagnostics

`scripts/check_heat3d_v1_label_diagnostics.py --subset ...medium_expansion_v2`
passed.

- diagnosed samples: 24
- status counts: pass 24
- label_meta present: 24
- warning samples: none
- fail samples: none

Computed diagnostics cover array sanity, temperature sanity, simple bottom
Dirichlet consistency, and label metadata sanity. PDE residual, top Robin flux,
side adiabatic flux, interface flux, and global energy balance remain
`not_computed` or `requires_numerical_operator`.

## Zero-Delta Bridge Smoke

`scripts/check_heat3d_v1_zero_delta_bridge.py --subset ...medium_expansion_v2`
passed.

- checked samples: 24
- relative condition feature contract: pass
- `legacy_inputs.u`: zero-delta field
- raw absolute BC temperatures excluded from model-facing relative features
- target temperature excluded from inputs
- `T_ref + DeltaT == target_temperature`: pass
- baseline-shift model-facing input invariance: pass

This confirms the medium expansion subset can use the current zero-delta bridge
contract. It does not prove model quality.

## Tiny Training Smoke

`scripts/check_heat3d_v1_zero_delta_tiny_training.py --subset ...medium_expansion_v2`
passed with default tiny-smoke settings.

- samples: 24
- graph-shape groups: one group, `N=384`, `F=11`
- steps: 3
- lr: `1e-5`
- seed: 0
- normalized loss: `1.23060596 -> 1.22273016`
- grad norms: finite
- raw DeltaT MSE after smoke steps: `3.7266307e-03`
- recovered temperature MSE after smoke steps: `3.7266419e-03`
- repeatability: pass

This is only a forward/backward/optimizer/loss contract smoke. It is not a
formal training experiment.

## Train/Valid Smoke

`scripts/check_heat3d_v1_small_train_valid_smoke.py --subset ...medium_expansion_v2`
passed.

- split source: `subset_sample_meta`
- train samples: 16
- valid samples: 4
- ignored diagnostic test samples: 4
- graph-shape groups: one train group and one valid group, both `N=384`, `F=11`
- train normalized loss: `1.22230577 -> 1.21504140`
- valid normalized loss: `0.86200345 -> 0.85253000`
- final train raw DeltaT MSE: `4.2994246e-03`
- final valid raw DeltaT MSE: `3.0166784e-03`
- final train recovered temperature MSE: `4.2994265e-03`
- final valid recovered temperature MSE: `3.0166553e-03`
- repeatability: pass
- checkpoint/log written: false

## Validation Metrics Smoke

`scripts/check_heat3d_v1_validation_metrics_smoke.py --subset ...medium_expansion_v2`
passed.

- evaluated splits: train + valid
- train samples: 16
- valid samples: 4
- diagnostic test samples ignored by default: 4
- label_meta in evaluated samples: 20 / 20
- repeatability: pass
- train normalized loss: `1.22230577 -> 1.19980037`
- valid normalized loss: `0.862003446 -> 0.833342969`
- per-sample metric rows: 40
- split summaries: 4

Mean recovered temperature RMSE:

- train zero_delta baseline: `6.25255451e-02`
- train tiny trained prediction: `6.18251845e-02`
- valid zero_delta baseline: `5.05292862e-02`
- valid tiny trained prediction: `5.27965352e-02`

The validation metrics loop is operational on the medium expansion subset.
The tiny prediction is not consistently better than the zero-delta baseline on
every split, and these values must not be interpreted as model performance.

## Backward Compatibility

The old default smoke paths still pass:

- `scripts/check_heat3d_v1_zero_delta_bridge.py`
- `scripts/check_heat3d_v1_validation_metrics_smoke.py`

The default path remains the original supervised-small smoke unless an explicit
subset is provided.

## Implementation Notes

This stage required additive subset compatibility fixes:

- v1 sample discovery now accepts sample directories by `sample_meta.json`, not
  only `sample_*` names.
- supervised v1 loading accepts medium pilot and medium expansion smoke stages.
- zero-delta and training/metrics scripts can use explicit subsets with
  non-`sample_*` identifiers such as `medium_000`.
- validation and train/valid split resolution can fall back to `sample_meta.json`
  in the explicit subset when the default manifest does not describe that subset.

No model core or v0 public entrypoint is modified.

## Non-Claims

This report does not claim:

- high-fidelity solver labels
- formal benchmark status
- model performance
- OOD generalization
- formal grid convergence
- formal energy balance

The `test_ood_*` samples remain diagnostic candidates only.

## Next Step

The recommended next step is to expand the region-first, volume-fraction,
reference-solver-v2 pipeline toward the planned 64-sample medium dataset, then
rerun the existing train/valid and validation metrics smoke before any model
comparison protocol.
