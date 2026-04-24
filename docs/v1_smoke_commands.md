# Heat3D v1 Smoke Commands

These commands are smoke checks for the v1 research scaffold. They are not
formal experiments and must not be used as performance evidence.

Run from the repository root.

## 1. Python Syntax Check

```bash
python3 -m py_compile \
  rigno/heat3d_v1_schema.py \
  rigno/dataset_Heat3D_v1.py \
  rigno/heat3d_v1_supervised.py \
  rigno/heat3d_v1_native_supervised.py \
  rigno/heat3d_v1_reference_solver.py \
  scripts/validate_heat3d_v1_schema.py \
  scripts/inspect_heat3d_v1_sample.py \
  scripts/check_heat3d_v1_loader.py \
  scripts/check_heat3d_v1_graphs.py \
  scripts/check_heat3d_v1_supervised_targets.py \
  scripts/check_heat3d_v1_supervised_batch.py \
  scripts/check_heat3d_v1_native_supervised_contract.py \
  scripts/check_heat3d_v1_tref_shift_invariance.py \
  scripts/check_heat3d_v1_relative_bc_features.py \
  scripts/check_heat3d_v1_zero_delta_bridge.py \
  scripts/check_heat3d_v1_zero_delta_tiny_training.py \
  tools/generate_heat3d_v1_metadata_smoke.py \
  tools/generate_heat3d_v1_supervised_smoke.py
```

Purpose: catch syntax/import-level breakage in v1 files.

Failure usually means a local code or dependency issue before any schema,
loader, graph, or training-smoke logic can be trusted.

## 2. Schema Validator

```bash
python3 scripts/validate_heat3d_v1_schema.py
```

Purpose: validate v1 metadata-only sample structure, array shape consistency,
boundary descriptors, interface descriptors, units, and split tags.

Failure usually means the metadata-first subset contract is broken.

## 3. Loader Smoke

```bash
python3 scripts/check_heat3d_v1_loader.py
```

Purpose: verify v1 sample loading and `native` / `diag3` thermal-conductivity
encoding behavior.

Failure usually means the metadata loader, feature naming, or k-field encoding
contract is broken.

## 4. Graph Smoke

```bash
python3 scripts/check_heat3d_v1_graphs.py
```

Purpose: verify single-sample graph construction from v1 coordinates while
keeping pure-physics features separate from topology construction.

Failure usually means the v1 loader output no longer matches the current
Heat3D graph builder expectations.

## 5. Supervised Target Sanity

```bash
python3 scripts/check_heat3d_v1_supervised_targets.py
```

Purpose: inspect `temperature.npy` smoke labels for shape, dtype, range,
bottom Dirichlet behavior, and basic smoke-level sanity.

Failure usually means the smoke target labels or metadata boundary assumptions
are inconsistent.

## 6. Supervised Batch Smoke

```bash
python3 scripts/check_heat3d_v1_supervised_batch.py
```

Purpose: verify that the two supervised smoke samples can form a tiny batch
with consistent condition / target contracts.

Failure usually means sample-level contracts do not batch cleanly.

## 7. Native Supervised Contract Smoke

```bash
python3 scripts/check_heat3d_v1_native_supervised_contract.py
```

Purpose: verify the native v1 contract:

```text
condition_features -> target_temperature
```

Failure usually means `temperature.npy` is leaking into inputs, feature names
are inconsistent, or the native target contract is broken.

## 8. T_ref Shift Invariance Smoke

```bash
python3 scripts/check_heat3d_v1_tref_shift_invariance.py
```

Purpose: verify the smoke solver's linear baseline-shift behavior under
`300 K -> 350 K` boundary-temperature shift.

Failure usually means either the smoke solver, temporary metadata shift, or
current baseline-shift assumption is inconsistent.

## 9. Relative BC Feature Smoke

```bash
python3 scripts/check_heat3d_v1_relative_bc_features.py
```

Purpose: compare raw absolute BC features with relative BC features under a
temporary baseline shift, and compare `tref_u_bridge` with
`zero_delta_u_bridge`.

Failure usually means relative feature construction, T_ref resolution, or
bridge tensor contracts are broken.

## 10. Zero-Delta Bridge Forward / Loss Smoke

```bash
python3 scripts/check_heat3d_v1_zero_delta_bridge.py
```

Purpose: verify the current recommended bridge without training:

```text
legacy_inputs.u = zero_delta_field
legacy_inputs.c = relative_condition_features
target = Delta T
```

Failure usually means graph construction, forward execution, target-delta
construction, or input leakage checks are broken.

## 11. Zero-Delta Tiny Training Smoke

```bash
python3 scripts/check_heat3d_v1_zero_delta_tiny_training.py
```

Purpose: verify forward, backward, optimizer update, normalized DeltaT loss,
raw DeltaT recovery, recovered-temperature MSE reporting, and checkpoint-free
repeatability for the current default smoke route.

This script includes repeatability checks by default.

Failure usually means the recommended v1 tiny-training smoke contract is not
stable enough for further development.

## Current Default Smoke Route

The current recommended supervised smoke route is:

```text
relative BC feature view
zero_delta_u_bridge
normalized DeltaT target
T_pred = T_ref + DeltaT_pred
```

The historical `u = k_x` route is retained only as compatibility smoke and is
not the recommended path for future v1 training work.
