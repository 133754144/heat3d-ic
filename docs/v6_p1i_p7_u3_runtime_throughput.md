# V6 P1i P7 + U3 runtime/throughput closeout

All accuracy cells are frozen seed0 valid32. No training or test/sealed access occurred. Historical U2 remains unchanged.

## Primary comparison

| route | full PG (%) | raw CV (K) | B1 E2E median/p95 (s) | resident throughput | peak VRAM |
|---|---:|---:|---:|---:|---:|
| E16384 | 2.702270 | 2.284806 | 0.778409/0.819143 | B8 190.499 samples/s | 2.820 GB |
| E32768 | 2.672444 | 2.208698 | 0.861476/0.947140 | B2 117.797 samples/s | 1.777 GB |
| U1-32768 | 2.748527 | 2.304393 | 0.778012/0.851240 | B16 174.954 samples/s | 6.904 GB |
| U-direct240825 | 2.818838 | 2.318104 | 0.591519/0.613574 | B1 17.146 samples/s | 6.098 GB |

## P7 fresh throughput

E16384 fresh distinct-case throughput peaks at B32 = 1.826 samples/s. Parallel FVM saturates at 4 processes = 0.544 samples/s, a semantically matched 3.36x throughput ratio. Fresh CPU preprocessing remains the dominant stage.

## U3 exact adapter and direct output

The 1024→1024 identity gate passed bitwise for all 32 samples. Replacing the redundant full local graph build reduced `dummy_local_p2r` from 0.007356 s to 0.002595 s (2.83x) without changing checkpoint parameters or outputs.
U1-32768 completed actual B1/B4/B8/B16 attempts without OOM; best resident throughput is B16 = 174.954 samples/s. Its optimized B1 E2E is 0.778012 s, essentially tied with E16384 (0.778409 s), while PG is 2.748527% (+0.046256 pp).
The independent 1024→240825 direct smoke passed, followed by valid32: PG 2.818838%, B1 E2E 0.591519 s (1.32x versus E16384), peak VRAM 6.098 GB. It avoids adaptive high-N support and reconstruction, but the PG penalty is +0.116568 pp.

## Decision

P7 is GO. The paper production mainline remains E16384. U3 is an engineering-feasibility GO but a production-replacement NO-GO: U1-32768 does not dominate E16384, and U-direct240825's faster B1 result comes with a valid32 PG penalty exceeding the earlier +0.1 pp non-inferiority reference. A new independently preregistered confirmation would be required before promoting the direct route.

`single-case latency`, `fresh distinct-case batch throughput`, `streamed prepared-host throughput`, and `resident inference throughput` are separate workload semantics and are not interchanged.
