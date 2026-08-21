# V6 Authoritative valid32 Attempt 3 Closeout

## Decision

- `benchmark_lifecycle_schema = GO`
- `ready_for_authoritative_valid32 = GO` before launch
- Attempt 3: `formal_measurement_attempted=true`
- Attempt 3: `formal_matrix_completed=false`
- Attempt 3: `publication_results_generated=false`
- `publication_timing_freeze = NO_GO`

No training, accuracy tuning, test/sealed access, checkpoint change, dataset
change, graph/padding-policy change, or route-semantic change occurred.

## Lifecycle-schema amendment

The five routes now share one formal mode contract. Serial owns cold,
fresh/Q1, cache-hot, and resident. Q2 owns submit-to-result,
inter-completion, throughput, and B16-to-B32. Inapplicable fields are JSON
`null`; empty timing statistics and fallback to warmup/replay are forbidden.
FVM Serial uses separate cache-hot and prepared-system solve-only resident
pools. The 10 route/mode fixtures and an expanded synthetic 30-cell collector
parse passed without GPU or model execution.

## Attempt 3

Attempt 3 started from a new directory and reused no Attempt 2 cell. The first
seed (`20260814`) completed all 10 route/mode cells. Seed `20260815` completed
all five Serial cells. The next process,
`E16384_reconstruction / 20260815 / Q2`, exited with:

`RuntimeError: Q2 lifecycle did not produce one passed order`.

The attempt therefore completed 15/30 cells and attempted 16/30 processes.
The formal orchestrator stopped immediately; the remaining 14 cells were not
started. The failing Q2 route did not write a cell JSON, so the inner exception
caught by its Q2 worker loop is not available beyond the outer hard-gate
message. CUDA timer accuracy warnings occur in its log, but the contract did
not fail merely because those warnings appeared.

Per the preregistered rules, no single cell was rerun, no implementation or
statistics were changed after failure, no abnormal value was removed, and the
collector/authoritative checker were not run. The 15 partial cells cannot be
used for publication latency or speedup claims.

## Evidence

- lifecycle seal: `configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_pre_measurement_seal_lifecycle_schema.json`
- Attempt 3 raw/logs: `configs/heat3d_v6_p1i/v6_p1i_publication_authoritative_valid32_failure_f1b55a0_attempt3_raw/`
- failure manifest: `configs/heat3d_v6_p1i/v6_p1i_publication_authoritative_valid32_failure_f1b55a0_attempt3_manifest.json`
- machine closeout: `configs/heat3d_v6_p1i/v6_p1i_publication_authoritative_valid32_attempt3_closeout.json`
