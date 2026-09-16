# V7 G2-P18：publication limitations and interpretation

## Scope

The common-valid comparison uses the frozen Heat3D valid128 physical cases only.
Every model is evaluated in temperature space on the same 101×101×56 (571,256
point) domain with the pre-registered sample-first relative RMSE. `test_iid`, sealed
sets, and the DeepOHeat official100 evaluation remain locked.

## What is comparable

- `DeepOHeat-full-minus-valid128` excludes all 128 Heat3D valid source IDs from its
  99,872-case training pool. Its best valid RMSE is 1.146688 ± 0.038323% (sample SD).
- `DeepOHeat-768` and `Heat3D-768-e600` share the frozen 768/128 physical-case split.
  Their best full-field values are 1.305167 ± 0.268321% and 0.709888 ± 0.007765%,
  respectively.
- These are physical-case/data-regime comparisons. They are not same-information-
  budget or same-compute comparisons: DeepOHeat uses a full-mesh PDE/BC objective,
  while Heat3D uses supervised temperature labels at 1024 sparse support points and
  U-v2 direct-query dense inference.

## Checkpoint policy

The formal Heat3D row is the fixed e600 endpoint (not a validation-selected
checkpoint). DeepOHeat-full-minus-valid128 and DeepOHeat-768 report both their
validation-selected best and fixed final-100,000-iteration endpoints. No ambiguous
best-to-best primary comparison is used; the two DeepOHeat views are labelled
separately.

## What is not rankable

The historical DeepOHeat native-full seed42 model trained on the original 100,000
pool, which contains the Heat3D valid128 rows. It is retained as `NATIVE_REFERENCE`
and is not ranked on valid128. The official100 test remains sealed.

## Historical diagnostic and latency limits

P14.1 e200 Heat3D U-v2 direct-query dense value is 0.814137 ± 0.006054%, versus
IDW 1.431502 ± 0.016591%; the IDW oracle floor is 1.222146% and a U-v2 oracle is
not defined for direct query. These are historical lower-bound diagnostics only,
not formal comparison rows and not an additive decomposition of model error and
any mapping error.

Heat3D U-v2 direct-query dense end-to-end latency includes full-query graph construction
and direct-query forward (about 1,008 s per valid128 evaluation in P15). Comparable dense inference
latency was not recorded in the DeepOHeat P14/P17 receipts, so no latency Pareto
claim is made. Same-hardware remeasurement is required before such a claim.

## Training-budget limits

DeepOHeat runs use 100,000 iterations with 50 sampled function instances per
iteration (5.0M sampled function instances processed); this is not a claim of 5.0M
independent collocation points. Heat3D uses 600 epochs with B24. The cohorts should
therefore be described as data-regime and physical-case efficiency evidence, not as
equal training-budget experiments.

## Unopened data and excluded tracks

No test/sealed/official100 data were opened in P14–P18. Therm-FM, multi-HTC,
Geo-FNO, and DeepOHeat-v2 are outside this final common-valid cohort by protocol.
