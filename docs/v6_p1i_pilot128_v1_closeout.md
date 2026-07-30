# V6-P1i pilot128 v1 closeout

## Decision

The pilot is **physically valid but fails the preregistered distribution gate**.
Formal 1024 generation remains prohibited. No samples were filtered, replaced,
or resampled after solving, and no model training or inference was run.

## What passed

- 128/128 Sobol cases solved on the 240825-node layer-aligned FVM mesh.
- Maximum absolute energy-balance error: `1.06e-10`.
- Maximum normalized linear residual: `1.27e-10`.
- Every local source/conductivity region has projected support; minimum is 4
  nodes.
- All arrays and metrics are finite.
- All 12 equal-width intervals from 30 to 150 K are occupied.
- The peak-DeltaT KDE is unimodal, not a four-cluster distribution.

## What failed

- Only 108/128 cases (84.375%) fall inside 30--150 K; the contract requires
  at least 90%.
- Four cases are below the wider 20 K safety bound.
- Counts in the 12 primary bins are
  `[20, 25, 11, 16, 8, 11, 10, 2, 2, 1, 1, 1]`; the nonzero max/min ratio is
  25, above the frozen limit of 5.

Peak DeltaT is 15.070--142.399 K (median 49.534 K). Mean DeltaT is
10.309--121.344 K, and CV-RMS DeltaT is 10.475--121.927 K.

## Attribution and next decision

The strongest monotonic association with peak DeltaT is top Robin `h`
(Spearman -0.753), followed by package power (+0.619) and maximum source `q`
(+0.499). Bottom `h` is weak (+0.045) in this stack. This explains the
low-temperature pile-up: independently sampling a wide top-cooling range and
power range does not yield near-uniform temperature coverage.

If another pilot is authorized, the single recommended change is a global,
pre-solve continuous severity mapping that correlates power with cooling
strength. It must be frozen before solving and must not use per-sample
thermal-resistance inversion. No v2 config or 1024 dataset is created in this
closeout.

