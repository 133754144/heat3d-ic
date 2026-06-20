# V4 P1 Semantic Normalization Design

Read this file only for V4 P1 normalization profile, training-semantics
cleanup, or model-lab merge review questions.

## Scope

This is an audit/design note. It does not change training defaults, model
structure, solver, loss, loader behavior, registry CSV, run artifacts, or
checkpoint behavior.

Current default remains:

`normalization_profile = legacy_zscore`

Opt-in profile under V4 runner plumbing:

`normalization_profile = semantic_normalization_v1`

`semantic_normalization_v1 = condition semantic normalization + coord provenance`

This is not a global coordinate-normalization fix. It changes the condition
feature transforms for k/q/BC/top_h/BC temperature scalars and records
coordinate physical extent/aspect-ratio provenance, but model coordinates still
use the legacy `train_minmax_to_unit_box` policy unless a later
coordinate-policy or position-encoding experiment explicitly changes it.

## Current Legacy Facts

The active V4 path uses the V1 medium controlled runner and extracted legacy
helper logic:

- `scripts/run_heat3d_v1_medium_controlled_training_export.py`
- `scripts/check_heat3d_v1_small_train_valid_smoke.py`
- `rigno/heat3d_v1_training_semantics.py`
- `rigno/heat3d_v1_normalization.py`
- `rigno/heat3d_v1_native_supervised.py`
- `rigno/dataset_Heat3D_v1.py`

Confirmed behavior:

| item | current behavior |
| --- | --- |
| coords | Train-only `coord_min` and `coord_span`; model coordinates are `2 * ((coords - coord_min) / coord_span) - 1`. Physical extent/aspect ratio is not a separate model input. |
| u | `zero_delta_u_bridge`; legacy `Inputs.u` is an all-zero delta-temperature field. |
| c feature source | Relative BC condition view: `k_x/k_y/k_z`, `q`, BC flags, `top_h`, `top_T_inf_minus_T_ref`, `bottom_T_fixed_minus_T_ref`. |
| c normalization | Every c feature uses the same per-feature train mean/std z-score path: `(raw_c - condition_mean) / condition_std`. |
| k | `k_x/k_y/k_z` are linearly z-scored per feature. No log scale or physical-scale encoding. |
| q | `q` is linearly z-scored. No log1p, source-power, or energy-scale transform. |
| BC flags | `is_top/is_bottom/is_side/is_interior` are included in c and z-scored, so binary flags become continuous values. |
| BC scalars | `top_h` and relative BC temperature scalars use the same c z-score mechanism as all other c channels. Constant channels get safe std `1.0`. |
| target | `target_delta_u = T - T_ref`; training target is normalized DeltaT: `(target_delta - target_delta_mean) / target_delta_std`. |
| recovery | `DeltaT_pred = pred_norm * target_delta_std + target_delta_mean`; `T_pred = T_ref + DeltaT_pred`. |

The issue is not missing per-channel z-score. The current path already z-scores
c per feature. The issue is missing semantic normalization: binary masks,
material properties, heat sources, convective coefficients, coordinate scale,
and target amplitude are all handled by generic linear statistics.

## Semantic Normalization V1

`semantic_normalization_v1` is an opt-in condition-feature profile with
coordinate provenance. It must not change `legacy_zscore` outputs unless the
caller explicitly selects the new profile.

| feature class | legacy_zscore | semantic_normalization_v1 plan |
| --- | --- | --- |
| coords | Train min/max to `[-1, 1]`; physical size is implicit. | Keep normalized coordinates unchanged. Record `physical_extent_m` and `aspect_ratio` as manifest/provenance only. A later coordinate-policy or position-encoding experiment may promote them to model inputs. |
| u | Zero delta bridge, not z-scored. | Keep `zero_delta_u_bridge` unless a separate bridge experiment is approved. |
| k | Linear per-axis z-score. | Use `log_k` or per-axis physical-scale transform; retain anisotropy information through `k_x/k_y/k_z` and optional ratios. |
| q | Linear z-score of volumetric source. | Use `log1p_q` or source-power scale, with explicit zero-source handling. Record q scale unit/policy. |
| BC flags | Binary flags are z-scored into continuous channels. | Keep `is_top/is_bottom/is_side/is_interior` as exact `0/1` mask channels. Do not z-score. |
| top_h | Linear z-score in the shared c path. | Use independent `top_h` scaler, preferably log or bounded physical scale. |
| BC temperature scalars | Relative scalars use shared c z-score; constant channels become zero with std `1`. | Keep relative-to-`T_ref` semantics, but use a separate scaler and record whether the channel is constant. |
| target | Normalized DeltaT using train mean/std. | Continue normalized DeltaT for base loss, but record raw K recovery, train DeltaT scale, and OOD scale diagnostics. |

Minimum profile contract:

```text
normalization_profile: legacy_zscore | semantic_normalization_v1
coord_policy: train_minmax_to_unit_box | unit_box_plus_extent_aspect
bridge_policy: zero_delta_u_bridge
target_mode: normalized_deltaT
target_recovery_policy: deltaT_norm_to_K_plus_T_ref
condition_feature_transform:
  k: linear_zscore | log_k
  q: linear_zscore | log1p_q
  bc_flags: zscore | binary_passthrough
  top_h: linear_zscore | independent_physical_scale
  bc_temperature_scalars: linear_zscore | independent_relative_scale
```

## Smoke-Path Cleanup

P1.1a extracts the current legacy behavior without enabling a new profile:

| logic | stable helper | current callers |
| --- | --- | --- |
| `zero_delta_u_bridge` choice | `rigno/heat3d_v1_training_semantics.py` | smoke helper, medium runner, P1 audit/final-probe smoke |
| train-only c/target/coord stats | `rigno/heat3d_v1_normalization.py` | smoke helper, medium runner, P1 audit/final-probe smoke |
| coordinate normalization | `rigno/heat3d_v1_normalization.py` | smoke helper, medium runner, P1 audit |
| c z-score and normalized target | `rigno/heat3d_v1_normalization.py` | smoke helper, medium runner, P1 audit |
| raw DeltaT/T recovery | `rigno/heat3d_v1_normalization.py` | smoke helper and medium runner metrics/export helpers |
| final-probe BC mask compatibility | final-probe eval script | final-probe adapter helper, separate from dataset loader |

Implemented module split:

- `rigno/heat3d_v1_training_semantics.py`: named route contracts, bridge
  policy, target mode, feature manifest, recovery policy strings.
- `rigno/heat3d_v1_normalization.py`: `legacy_zscore` implementation copied
  from the existing smoke helper: train-only stats, coords min/max scaling,
  condition z-score, target DeltaT normalization, and raw DeltaT/T recovery.

Equivalence check:

```bash
python scripts/check_heat3d_v1_training_semantics_equivalence.py --subset <subset>
```

The checker compares the pre-extraction reference formulas against the helper
for `u`, normalized coords, normalized `c`, normalized target, raw DeltaT
recovery, and raw temperature recovery. `semantic_normalization_v1` remains
disabled by default and is available only through explicit registry opt-in.

P1.1b runner/config plumbing:

- `dataset.normalization_profile` is the only new registry/YAML config
  dimension: `legacy_zscore` or `semantic_normalization_v1`.
- Missing or `legacy_zscore` keeps the original
  `scripts/run_heat3d_v1_medium_controlled_training_export.py` command path.
- `semantic_normalization_v1` selects
  `scripts/run_heat3d_v4_controlled_training.py`, a wrapper around the legacy
  runner that swaps in semantic train-only normalization stats and records the
  profile in future run provenance.
- The first registered semantic config is
  `V4P1_01_baseline_normalization`; it changes normalization profile and run
  identity/paths only.
- This is configuration plumbing and helper implementation, not performance
  evidence.

## Provenance Fields

Future runs should write these fields into `run_config.json`,
`loss_summary.json`, and eventually result registry fields. Do not add them to
`run_registry.csv` until a separate registry task approves the schema change.

| field | value for current legacy baseline | purpose |
| --- | --- | --- |
| `target_mode` | `normalized_deltaT` | Defines supervised target space. |
| `bridge_policy` | `zero_delta_u_bridge` | Explains `Inputs.u` and `T_ref` handling. |
| `normalization_profile` | `legacy_zscore` | Names the active transform profile. |
| `feature_manifest_hash` | hash of ordered feature names and transforms | Detects silent feature/transform drift. |
| `coord_policy` | `train_minmax_to_unit_box` | Records coordinate scaling and extent handling. |
| `condition_feature_transform` | per-feature transform map | Distinguishes z-scored flags from binary passthrough, log q, log k, etc. |
| `target_recovery_policy` | `deltaT_norm_to_K_plus_T_ref` | Defines raw K recovery from model output. |

## Decision

Keep `legacy_zscore` as the default V4 baseline unless a registry entry
explicitly selects `semantic_normalization_v1`. P1.1b adds the opt-in V4 runner
path and config plumbing; it does not change model structure, solver, loss, or
loader behavior.
