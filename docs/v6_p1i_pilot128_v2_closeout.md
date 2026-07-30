# V6-P1i pilot128 v2 closeout

## Decision

V2 passes every frozen pilot gate and is eligible for a separately frozen
formal1024 protocol. Formal1024 remains closed until this pilot has been
reported and an explicit expansion decision is made.

No sample was filtered, replaced, or power-backsolved from its individual
thermal resistance. No model training or inference was run.

## Physical and numerical QC

- 128/128 Sobol cases solved on the 240825-node FVM mesh.
- Maximum absolute energy-balance error: `9.99e-11`.
- Maximum normalized linear residual: `1.10e-10`.
- Minimum solver CVs per source: 240.
- Minimum projected nodes per local source/k region: 4.
- All saved arrays and metrics are finite.

## Continuous temperature coverage

| metric | minimum | median | maximum |
|---|---:|---:|---:|
| peak DeltaT | 35.389 K | 93.672 K | 166.765 K |
| mean DeltaT | 26.521 K | 74.269 K | 132.757 K |
| CV-RMS DeltaT | 26.595 K | 75.130 K | 133.048 K |

- 124/128 (96.875%) peak values are inside 30--150 K.
- All 128 are inside the 20--180 K outer safety interval.
- Twelve-bin counts are `[5, 8, 12, 11, 13, 10, 9, 10, 13, 11, 12, 10]`.
- There are no empty bins; the max/min count ratio is 2.6.
- The largest sorted peak gap is 8.673 K.
- KDE finds two broad modes near 69.2 and 116.8 K, not a four-block pattern.

## Deconfounding effect

Continuous severity remains strongly monotonic with peak DeltaT
(Spearman 0.985), as intended. The direct top-h association falls from -0.753
in v1 to -0.035 in v2 after applying the single global power-law rule.
Background-kz correlations with peak DeltaT are all below 0.05 in magnitude;
they remain independently sampled physical variability rather than hidden
temperature selectors.

Train/valid/test were assigned from sample identity without reading labels.
Their descriptive peak medians are 92.19, 107.59, and 89.44 K. The formal1024
protocol should use a target-independent Sobol-stratified split assignment to
reduce the finite-16 pilot imbalance without consulting solved temperature.

## Expansion boundary

The generator, v2 config, seed, acceptance contract and their hashes may now be
frozen as the formal1024 parent. Expansion must still be a new dataset ID and
must recheck conservation, convergence, parameter coverage, temperature
coverage and train/valid/test consistency. This closeout does not generate it.

