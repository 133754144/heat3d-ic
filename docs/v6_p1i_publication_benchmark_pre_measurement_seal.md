# V6 Publication Benchmark Pre-measurement Seal

## Decision

- `pre_measurement_seal = GO`
- `ready_for_authoritative_valid32 = GO`
- `publication_timing_freeze = NO_GO_ready_for_full_valid32`

This seal does not contain publication latency, speedup, or a new full-valid32
measurement. It freezes the implementation and analysis contract that must be
used by the later formal measurement.

## Historical golden exactness

The reference is the immutable Git object at commit
`519963c3ce2f0a04c9b51ec2998c13ca0753190a`, not a second invocation of the
current implementation. Its tracked conformance artifact has SHA256
`257fe38b21d8aee630569a7e96eaaa390b4db346d7fe82aecac15f4e7770e3c1`.

Twelve route/seed records cover the serial exactness witness for the four
neural routes at 16384 and 240825. Each record binds native-1024 and query
metadata/edge component hashes separately from the prepared-payload hash. The
seal checker reads the historical blob through `git show`, recomputes the
candidate/reference records, and compares the SHA values directly. A
same-launch replay remains a useful secondary diagnostic but is not the sole
reference.

## Frozen analysis

Each route and seed is summarized independently with median and p95. Fresh/Q1
uncertainty uses the same 32 ordered sample IDs for FVM and each neural route,
then performs a paired workload bootstrap within each seed using seed
`20260821` and 20,000 resamples. The three randomized lifecycles are summarized
with median and min--max only. Q2 throughput and B16-to-B32 likewise report the
three-repeat median/range and do not call an n=3 bootstrap a 95% interval.
Pooled-96 ratios and post-hoc changes to aggregation are forbidden.

## Frozen workload and lifecycle

The five routes retain 30 independent lifecycles: five routes, three order
seeds, and separate serial/Q2 processes. The formal runner uses 32 distinct
valid inputs only after this seal. Its pools are disjoint:

- cold first case and fresh distinct cases belong to the serial lifecycle;
- repeat-case cache-hot is a separate post-fresh repeat pool;
- resident core uses a separately prepared valid payload or FVM system;
- Q2 runs in its own process and performs no serial prepass;
- true B16-to-B32 is obtained from the real Q2 completion trace.

FVM fresh/Q1 uses persistent in-process P1 with one thread. FVM Q2 and
B16-to-B32 use persistent P2 with one thread per process. Resident FVM is
prepared-system solve-only. The common service boundary remains in-memory
`k/q/BC` to synchronized 240825-node result.

## Runtime-state and memory semantics

All five route-specific padding and tensor envelopes are frozen in the
machine-readable seal with canonical hashes. A train-only target-free static
JIT envelope remains permitted; graph or packing shapes from timed cases may
not be prewarmed.

Neural routes and FVM serial use the single service process HWM. FVM Q2 cannot
sample a reliable simultaneous parent-plus-worker RSS peak, so its memory
field is explicitly named `summed_process_HWM_upper_bound_bytes`. It must not
be reported as simultaneous aggregate RSS.

## Scope

No training, test/sealed access, accuracy tuning, full-valid32 timing, or
formal latency/speedup generation occurred. The recent real-route conformance
smoke from commit `519963c3` was reused. The only new execution is the tracked
one- or two-case FVM in-process persistent-P1 sanity bound into the seal.

Machine-readable files:

- `configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_pre_measurement_protocol.json`
- `configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_pre_measurement_seal.json`
- `scripts/collect_heat3d_v6_publication_benchmark_v1_1.py`
- `scripts/check_heat3d_v6_publication_benchmark_pre_measurement.py`
