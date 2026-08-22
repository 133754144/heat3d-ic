# V6 publication authoritative valid32 — Attempt 4 closeout

Attempt 4 passed all 30 independent route/seed/lifecycle cells.  The frozen
collector and final checker passed without deleting observations, rerunning a
cell, or changing the model, graph, padding, collector or statistics.  Training,
test and sealed roles remained closed.

## Runtime isolation gate

The required non-authoritative `E16384_reconstruction / seed20260815 / Q2`
diagnostic completed 32/32.  All service residuals passed; graph/payload/golden
audit was outside the production timing and Q2 refill boundary; service HWM was
captured before audit buffers.  The diagnostic JSON SHA256 is
`8ab4b89754f132001c9cea9cc1b759d84859f20196d0ecca4c3848577cb5c708`.

## Frozen publication timing

The table reports the median of the three independent lifecycle repeats.  Q1
and fresh are single-case seconds; Q2 is samples/s.  Speedup is paired against
FVM by sample and seed before the three-seed median.

| Route | Fresh/Q1 (s) | Q2 throughput (samples/s) | B16→B32 marginal (s/case) | Fresh speedup | Q2 speedup |
|---|---:|---:|---:|---:|---:|
| E16384 + reconstruction | 0.883247 | 1.72633 | 0.567698 | 2.001× | 2.005× |
| U-v2 16384 + reconstruction | 0.866177 | 1.75948 | 0.561611 | 1.976× | 2.053× |
| U-v2 direct 240825 | 1.30909 | 1.14757 | 0.855329 | 1.292× | 1.340× |
| E direct 240825 control | 1.31023 | 1.31089 | 0.730085 | 1.322× | 1.531× |
| FVM 240825 reference | 1.70068 | 0.857039 | 1.15262 | 1.000× | 1.000× |

Fresh/Q1 paired workload uncertainty uses 20,000 bootstrap resamples with seed
`20260821` inside each lifecycle seed.  Q2 and B16→B32 use only the preregistered
three-repeat median and min–max; they are not labelled as 95% confidence
intervals.

## Evidence

- Raw 30-cell JSON/log bundle and collector outputs:
  `configs/heat3d_v6_p1i/v6_p1i_publication_authoritative_valid32_attempt4_04dc85c_raw/`
- SHA manifest:
  `configs/heat3d_v6_p1i/v6_p1i_publication_authoritative_valid32_attempt4_04dc85c_manifest.json`
- Machine closeout:
  `configs/heat3d_v6_p1i/v6_p1i_publication_authoritative_valid32_attempt4_closeout.json`

Final decision: `publication_timing_freeze = GO`.
