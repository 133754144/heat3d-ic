# V6 P1i P5-S2 support-ordering closeout

Status: **PASS**. All 64 valid32 route/sample prefixes and downstream exact gates passed.

## B8192_adaptive

| Quantity | P5-S median (s) | P5-S2 median (s) | Speedup |
|---|---:|---:|---:|
| support ordering | 0.190695 | 0.182083 | 1.047x |
| continuous preprocessing | 0.424971 | 0.421205 | 1.009x |

### Ordering profile

| Stage | median (s) | p95 (s) |
|---|---:|---:|
| mask_seconds | 0.000298 | 0.000329 |
| sha256_seconds | 0.098132 | 0.099426 |
| sort_seconds | 0.059948 | 0.061076 |
| inner_interleave_seconds | 0.007124 | 0.008062 |
| outer_interleave_seconds | 0.006200 | 0.006312 |

Remaining continuous-stage bottleneck: `support_ordering` (0.182083 s).

## E32768_adaptive

| Quantity | P5-S median (s) | P5-S2 median (s) | Speedup |
|---|---:|---:|---:|
| support ordering | 0.320050 | 0.227747 | 1.405x |
| continuous preprocessing | 0.600888 | 0.514411 | 1.168x |

### Ordering profile

| Stage | median (s) | p95 (s) |
|---|---:|---:|
| mask_seconds | 0.000296 | 0.000357 |
| sha256_seconds | 0.096606 | 0.101444 |
| sort_seconds | 0.059719 | 0.066891 |
| inner_interleave_seconds | 0.036150 | 0.037262 |
| outer_interleave_seconds | 0.025179 | 0.025691 |

Remaining continuous-stage bottleneck: `support_ordering` (0.227747 s).

## Decision

The cached lazy-interleave implementation is retained. Further Python micro-optimization stops: remaining ordering time is attributed by the frozen profile, and C/C++ or semantic/approximate changes are out of scope.
