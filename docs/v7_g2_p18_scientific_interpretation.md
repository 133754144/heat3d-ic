# V7 G2：common-valid scientific interpretation

All numbers below are valid-only measurements on the frozen 128 physical cases and
571,256-point temperature-space evaluator. They are not test or universal-domain
claims.

## Q1 — same 768 physical cases

`Heat3D-768-e600` reaches 0.709888 ± 0.007765% U-v2 full-field sample-first
relative RMSE, versus 1.305167 ± 0.268321% for `DeepOHeat-768` (best checkpoints).
On this valid cohort Heat3D is lower by 0.595279 percentage points (about 45.6%
relative to DeepOHeat-768). Heat3D has 892,776 parameters and 3.868 GiB peak device
memory, while DeepOHeat-768 has 3,303,680 parameters and 0.700 GiB peak device
memory. Mean training wall is 0.742 h/seed versus 0.660 h/seed. DeepOHeat-768
uses a PDE/BC full-mesh objective; Heat3D uses supervised labels on 1024 sparse
support plus deterministic U-v2 reconstruction. Thus this is a same-physical-
case/data-regime result, not same-information or same-compute evidence.

## Q2 — DeepOHeat data scale

`DeepOHeat-full-minus-valid128` (99,872 training cases) obtains 1.146688 ± 0.038323%
best, compared with 1.305167 ± 0.268321% for the 768-case cohort. The best-checkpoint
value improves by 0.158479 percentage points (12.1% relative), with similar mean
wall time per seed in these receipts because both use 100,000 iterations. Final
values are 1.507919 ± 0.218354% and 1.436807 ± 0.309476%, respectively; this
best/final difference is reported descriptively and does not change selection rules.

## Q3 — Heat3D small-data efficiency

On this common valid domain, Heat3D-768-e600 (0.709888%) is below the
full-minus-valid128 DeepOHeat value (1.146688%) while using 768 supervised physical
cases rather than 99,872 physics-informed source cases. This supports a
physical-case/data-regime efficiency observation, not a claim of equal information
or equal compute: supervision modality, full-mesh PDE sampling, sparse conditioning,
parameter count, and memory differ.

## Q4 — reconstruction bottleneck

P14.1 e200 gives native-1024 0.779942%, IDW dense 1.431502%, and U-v2 dense
0.814137%. The IDW ground-truth-support oracle floor is 1.222146%; a direct-query
U-v2 oracle is not defined. P15 e600 U-v2 improves to 0.709888%, while IDW improves
to 1.381139%. The pattern indicates that both model convergence and sparse-to-dense
reconstruction contribute: U-v2 has a much smaller gap to native prediction than
IDW, but the oracle is only a lower-bound diagnostic. We do not subtract oracle
errors to claim an additive decomposition.

## Boundary

The historical native-full DeepOHeat seed42 model is retained as `NATIVE_REFERENCE`
only because its original pool overlaps valid128. Official100, `test_iid`, and
sealed sets remain locked. No Therm-FM, multi-HTC, sparse-1024 DeepOHeat, or learned
decoder experiment is included in this cohort.
