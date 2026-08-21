# V6 publication benchmark lifecycle-schema closeout

## Scope

This amendment changes only benchmark result lifecycle/schema handling. It does
not change the frozen model, checkpoint, dataset, graph, padding envelope,
sampling, metrics, or timing boundary. No GPU smoke matrix or model inference
was run while sealing the schema.

## Frozen mode contract

Each formal route process emits all lifecycle keys. Inapplicable keys are JSON
`null`; they are never synthesized from warmup, replay, cache-hot, or an empty
series.

| Mode | Required | Must be null |
|---|---|---|
| Serial | cold, fresh/Q1, cache-hot, resident | Q2 submit-to-result, inter-completion, throughput, B16-to-B32 |
| Q2 | submit-to-result, inter-completion, throughput, B16-to-B32 | cold, fresh/Q1, cache-hot, resident |

FVM Serial uses an in-process persistent P1 service. Cache-hot and resident are
now independent pools; resident is prepared-system solve-only. FVM Q2 uses the
frozen persistent P2 contract and has all Serial-only fields set to `null`.
Neural Q2 processes do not run the Serial cache-hot/resident/replay loops.

Formal 32-sample route cells must have status `passed`. Provenance is split into
`formal_measurement_attempted`, `formal_matrix_completed`, and
`publication_results_generated`; the last field can become true only in the
collector after a complete 30-cell matrix.

## Regression gate

The schema-only gate instantiates five routes times two modes (10 fixtures),
serializes every payload with non-finite values forbidden, validates required
and null fields, checks the Q2 B16-to-B32 requirement, and parses each fixture
through the real collector normalization path. Negative empty-series and
missing/non-finite Q2 marginal fixtures fail closed. It performs no dataset,
GPU, graph, checkpoint, or model work.

Result: `benchmark_lifecycle_schema = GO` and
`ready_for_authoritative_valid32 = GO`. Publication timing remains
`NO_GO_ready_for_full_valid32` until a new, indivisible 30/30 measurement and
collector closeout both pass.
