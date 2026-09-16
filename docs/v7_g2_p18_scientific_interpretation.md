# V7 G2：common-valid scientific interpretation

All formal values below are valid-only measurements on the frozen 128 physical
cases and the 571,256-point temperature-space evaluator. They are not test or
universal-domain claims. U-v2 is named **direct-query dense inference** throughout.

## Checkpoint policy

- **Heat3D-768:** the formal row is the fixed e600 endpoint; it is not a
  validation-selected checkpoint. Its dense view is U-v2 direct-query dense
  inference on the full 571,256-point domain.
- **DeepOHeat-full-minus-valid128** and **DeepOHeat-768:** report both the
  validation-selected best checkpoint and the fixed final-100,000-iteration
  endpoint. The formal table does not use an ambiguous best-to-best rule;
  best-to-best is retained only as two explicitly labelled checkpoint views.

## Q1 — same 768 physical cases

`Heat3D-768-e600` reaches 0.709888 ± 0.007765% U-v2 direct-query dense
full-field sample-first relative RMSE, versus 1.305167 ± 0.268321% for
`DeepOHeat-768` (validation-selected best checkpoints). On this valid cohort
Heat3D is lower by 0.595279 percentage points (about 45.6% relative to
DeepOHeat-768). Heat3D has 892,776 parameters and 3.868 GiB peak device memory,
while DeepOHeat-768 has 3,303,680 parameters and 0.700 GiB peak device memory.
Mean training wall is 0.742 h/seed versus 0.660 h/seed. DeepOHeat-768 uses a
PDE/BC full-mesh objective; Heat3D uses supervised labels on 1024 sparse support
plus direct-query dense inference. This is same-physical-case/data-regime
evidence, not same-information or same-compute evidence.

## Q2 — DeepOHeat data scale

`DeepOHeat-full-minus-valid128` (99,872 training cases) obtains 1.146688 ±
0.038323% at its validation-selected best, compared with 1.305167 ± 0.268321%
for the 768-case cohort. The best-checkpoint value improves by 0.158479
percentage points (12.1% relative), with similar mean wall time per seed because
both use 100,000 iterations. Final values are 1.507919 ± 0.218354% and
1.436807 ± 0.309476%, respectively; this best/final contrast is descriptive and
does not change selection rules.

## Q3 — Heat3D small-data efficiency

On this common valid domain, Heat3D-768-e600 (0.709888%) is below the
full-minus-valid128 DeepOHeat value (1.146688%) while using 768 supervised
physical cases rather than 99,872 physics-informed source cases. This supports a
physical-case/data-regime efficiency observation, not equal information or equal
compute: supervision modality, full-mesh PDE sampling, sparse conditioning,
parameter count, and memory differ.

## Q4 — historical representation diagnostics

P14.1 e200 values (native-1024 0.779942%, IDW dense 1.431502%, U-v2
direct-query dense 0.814137%) and the IDW ground-truth-support oracle floor
(1.222146%) are historical diagnostics only. They are excluded from the formal
P18 table and main conclusion. The IDW floor is a lower-bound diagnostic; it is
not subtracted from prediction error, and no additive model-versus-mapping error
decomposition is claimed. P15 e600 formal interpretation is conservative:
**no clear boundary right-censoring, stable plateau unproven**; no extension
beyond e600 is authorized by this receipt.

## Validity boundary

The historical native-full DeepOHeat seed42 model is retained as
`NATIVE_REFERENCE` only because its original pool overlaps valid128; it is not a
held-out ranking. P17 explicitly excludes valid128 from its training pool.
`test_iid`, sealed, and DeepOHeat official100 remain locked. No Therm-FM,
multi-HTC, sparse-1024 DeepOHeat, or learned decoder experiment is included.
