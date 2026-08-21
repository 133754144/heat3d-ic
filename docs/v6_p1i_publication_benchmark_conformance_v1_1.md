# V6 Publication Benchmark Implementation Conformance v1.1

## Decision

- `benchmark_protocol_v1_1 = GO`
- `benchmark_implementation_freeze = GO`
- `publication_timing_freeze = NO_GO_pending_full_valid32`

This closeout qualifies the benchmark implementation only. It does not publish
latency, speedup, accuracy, or full-population results.

## Frozen lifecycle and workload

The five real routes are `E16384 + reconstruction`, `U-v2 16384 +
reconstruction`, `U-v2 direct240825`, `E240825 direct` as an architecture
control, and `FVM240825` as the reference solver. All use the service boundary
`in-memory k/q/BC -> synchronized 240825-node result`.

The low-cost conformance run used four frozen-valid32 inputs per order and
three preregistered order seeds. Serial and Q2 used separate Python services,
so the complete smoke consisted of `5 routes x 3 seeds x 2 modes = 30`
independent PIDs. For a fixed seed, all routes and both modes received the same
ordered sample IDs. Neural routes used the same one-worker CPU/KD-tree/graph/
reconstruction policy. FVM serial used a persistent P1 service and Q2 used a
persistent P2 service, each worker restricted to one CPU thread.

The neural JIT envelope came from the lowest-SHA train sample
`v6p1if1_0389`; no target was read. Its static padding is the elementwise
maximum of the frozen valid envelope and the train warmup's actual edge counts.
No timed valid graph or packing shape was prewarmed. FVM has no JIT warmup and
did not pre-solve a timed case.

## Timing and residual semantics

Cold/fresh, cache-hot, and resident-core pools are explicitly separate. The
conformance smoke measures real fresh serial/Q2 execution but deliberately
marks repeat-case cache-hot as not measured; it does not synthesize a cache-hit
number. Q2 has no serial prepass and is a real depth-two service.

Every device span ends in explicit synchronization. Service residual is
computed as `E2E - sum(directly timed named stages)`. There is no artificial
`other` stage. The unchanged gate is
`0 <= residual <= max(0.025 s, 0.05 * E2E)`. Whole-service peak RAM is reported;
for FVM it is the aggregate of the parent service and all persistent workers.

## Exactness provenance

The checker compares candidate and reference hashes directly rather than
trusting prewritten pass booleans.

- Native 1024 metadata SHA:
  `ef9769719ee4271fe3c49c6e319731a341d97ed00b4feb3cc3f7895a6f467d4b`.
- Native 1024 real-edge SHA:
  `562d8640b1e0bd97e7b768c5ed6fc84baa3435297ddbd2de874bb77f5fac648d`.
- 16384 and 240825 each have nine direct audits: six E
  route/mode/seed replays and three U serial candidate/reference audits.
- Graph metadata/edge hashes and prepared-payload hashes are recorded and
  compared separately. The checkpoint before/after hash gate also passes.

## Artifacts and scope

- Protocol:
  `configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_standard_v1_1.json`
  (`325dd80dffadae0f56c547ec84902a717a59615c9d32bac2036d121bae17790b`).
- Raw real-route smoke:
  `configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_conformance_v1_1_smoke.json`
  (`257fe38b21d8aee630569a7e96eaaa390b4db346d7fe82aecac15f4e7770e3c1`).
- Real-route harness:
  `scripts/smoke_heat3d_v6_publication_benchmark_conformance_v1_1.py`.
- Fail-closed checker:
  `scripts/check_heat3d_v6_publication_benchmark_conformance_v1_1.py`.

The successful run accessed one train input for target-free warmup and four
valid-IID inputs per lifecycle cell. It did not train, read test/sealed roles,
run full valid32/96 timing, tune accuracy, or generate formal speedups.
Earlier path/envelope/readiness failures remain under `/tmp` and are excluded
from the frozen successful artifact.
