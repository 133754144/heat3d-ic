# V6 P1i P6 + U2 runtime/throughput closeout

All neural rows use frozen seed0 valid32. No training or test/sealed access occurred. Accuracy and timing are never interchanged across workload semantics.

| system | full PG (%) | raw CV (K) | fresh B1 med/p95 (s) | highest common B4 samples/s | peak VRAM at B4 (GB) |
|---|---:|---:|---:|---:|---:|
| E16384 | 2.702270 | 2.284806 | 0.778409/0.819143 | 188.56 | 1.835 |
| E32768 | 2.672444 | 2.208698 | 0.861476/0.947140 | 117.01 | 2.560 |
| U1-32768 | 2.748528 | 2.304401 | 0.944714/0.994337 | 188.82 | 2.401 |

## Frozen gates

- Accuracy: U1−E32768 = 0.076084 pp, within +0.1 pp: PASS.
- Matched B1 E2E: E/U1 = 0.912x versus required 1.2x: FAIL.
- Resident throughput at highest common batch B4: U1/E = 1.614x versus required 1.2x: PASS.
- Both runtime gates were preregistered as mandatory. U1-32768 is therefore NO-GO for production replacement; U1-240825 was not executed.

## Workload semantics

`single-case latency` is the fresh matched continuous pipeline. `resident inference throughput` uses already prepared/device-resident groups. `production batch throughput` includes streamed prepared-host H2D plus forward/reconstruction and is reported separately in the CSV.
FVM known-topology/new-physics B1 median/p95 is 1.573042/1.833450 s. Its 32-case result is serial, one process and one thread; it is not presented as a parallel batch result.

## Decision

The paper mainline remains E16384 for the production accuracy/latency route, with E32768 retained as a reference. U1 is a throughput-oriented diagnostic, not a production replacement.
