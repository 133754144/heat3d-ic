# V4 P3c-5 QC Freeze

Read this file only for P3c QC, candidate1024 acceptance, or P3b-lite
validation-subset decisions.

## Scope

P3c-5 freezes dataset QC policy before candidate1024 generation. It does not
change solver numerics, model code, loss code, loader semantics, or training
configuration.

## QC Classes

Accepted classes:

- `clean_keep`: solver, schema, q power, boundary, and DeltaT QC all pass.
- `physical_hard_keep`: high-DeltaT sample is physically explainable and all
  solver/q/boundary/schema checks pass.
- `review_hold`: accepted but flagged for validation-subset review.

Rejected class:

- `reject_resample`: candidate is excluded from manifest and may appear only in
  audit records.

## Reject/Resample

`reject_resample` is mandatory for:

- solver failure;
- NaN/Inf in generated arrays or labels;
- q power integration error beyond numerical tolerance;
- any q deposition on bottom/top/side boundary nodes;
- schema or loader-contract failure;
- `reject_low` DeltaT bin;
- high-DeltaT sample without a physical explanation.

Rejected candidates are not assigned sample IDs, do not enter
`manifest.json`, and are recorded only in `audit_summary.json`.

## Physical Hard Keep

High-DeltaT samples may be accepted as `physical_hard_keep` only when solver,
q-power, boundary, and schema checks pass and at least one reason applies:

- low-k trapped hotspot;
- weak cooling;
- multi-source or high-power bottleneck.

This keeps physically useful hard cases without silently accepting generator or
metadata failures.

## P3b-Lite Validation Subset

The P3b-lite subset is selected after accepted samples are split and audited.
It is a fixed audit/validation subset, not a stress split and not a pass/fail
gate.

Policy:

- use accepted samples only;
- fixed seed equals the dataset generation seed;
- target size is 64 samples, or all accepted samples if fewer than 64 exist;
- include `review_hold` samples first, capped by subset size;
- include deterministic representatives of `physical_hard_keep`, capped at one
  quarter of the subset;
- cover `qc_class`, `k_mode`, `diag3_policy`, `q_family`, `cooling_regime`,
  `DeltaT_bin`, and `high_deltaT_triage`;
- fill remaining slots by stable hash.

The selected IDs and coverage audit are written to `manifest.json` and
`audit_summary.json` under `p3b_lite_validation_subset`.

## Candidate1024 Gate

For `heat3d_v4_p3c_candidate1024_v0`:

- accepted sample count must be exactly 1024;
- rejected candidates are audit-only and excluded from manifest samples;
- train/test split uses deterministic stratified random assignment;
- full audit must include QC class counts, split distribution, P3b-lite subset,
  solver residuals, energy balance, boundary q checks, and q power consistency;
- `sha256_manifest.json` records the generated dataset files for local, remote,
  and Hugging Face consistency checks.
