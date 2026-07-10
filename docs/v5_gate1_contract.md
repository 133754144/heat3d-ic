# V5 Gate 1 Contract: Operator-Consistent Physics Scale

This document freezes the Gate 1 scope on `research/v5` after P0 commit
`e1889ac`. The machine-readable contract is
`configs/heat3d_v5/v5_gate1_contract.json`.

## Scope And Non-Goals

Gate 1 reads the frozen P5 dataset, derives deterministic scalar proxies, and
fits only scalar calibrations. It does not modify a model, loss, training
configuration, split, label, or P5 sample; train; generate data; call the
reference label solver; or enter Gate 2.

The P0-1 contract and audit remain unchanged. Its historical
`effective_source_power_W` is retained as `P_array`; P0 files are not renamed
or rewritten.

## Operator-Consistent Semantics

The authoritative source is `rigno/heat3d_v4_reference_solver.py`:

- `_assemble_triplets` replaces every bottom (`iz == 0`) row with
  `T = T_bottom` before it ever adds `q * CV_volume`.
- `_source_power_total` iterates only `iz = 1..end`.

Accordingly, Gate 1 records all three quantities per sample:

| name | definition | role |
| --- | --- | --- |
| `P_array` | `sum_all(q * CV_volume)` | P0-compatible array integral |
| `P_bottom` | `sum_bottom(q * CV_volume)` | source present on Dirichlet rows |
| `P_operator` | `sum_non_bottom(q * CV_volume)` | source term actually entering the operator RHS |

It verifies the coordinate bottom mask against both `bc_features.is_bottom` and
metadata, verifies bottom label temperature against `T_bottom`, and records
`T_inf - T_bottom`. A case is source-driven, BC-driven, mixed, or zero-drive
according to whether `P_operator` and that BC offset are active.

## Target And Candidate Bases

The Gate 1 target is

```text
s_y = CV-weighted RMS(temperature - T_bottom)
```

in K. It intentionally includes both source and BC amplitude.

The six candidates are constant, `P_operator`,
`q_rms * Lz^2 / kz_harmonic`, legacy `P_array * R_series`, source-centroid
two-path resistance, and a z-collapsed 1D finite-volume thermal proxy. The
1D network is an analytic deterministic proxy; it does not invoke the
reference label solver or write labels.

## Frozen Selection Protocol

- Fit every calibration only on `train` using log-space affine calibration.
- Select the best physical candidate only on `valid_iid` by log-RMSE.
- Inspect `hard_challenge_valid` only as OOD evidence.
- `test_iid` and `hard_challenge_test` are report-only after the decision; they
  cannot affect a formula, threshold, candidate selection, or calibration.
- A deterministic physics base is accepted only if its paired-bootstrap 95% CI
  for valid log-RMSE difference against constant has an upper endpoint below
  zero. Otherwise Gate 1 rejects all single proxies and defers global-scale
  learning to a later, explicitly authorized gate.

## Reproduction

Run the fixture suite first:

```bash
python3 -B scripts/check_heat3d_v5_gate1.py
```

On a server with P5 data and `rigno` activated, use the script's `--dry-run`
before the full read-only audit. The normal audit emits exactly three tracked
artifacts: the per-sample CSV table, closeout JSON, and closeout Markdown.
`--verify-summary` reconstructs the summary from the CSV without accessing the
P5 arrays.
