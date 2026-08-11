# V6 P1i P5-S lazy-prefix closeout

Status: **PASS**; geometry cache preparation 0.023714 s.

## B8192_adaptive

| Stage | Full-order baseline median (s) | Lazy-prefix median (s) | Speedup |
|---|---:|---:|---:|
| support_ordering | 0.718437 | 0.190695 | 3.767x |
| cv_redistribution | 0.043714 | 0.042903 | 1.019x |
| regional_prepare | 0.002000 | 0.002169 | 0.922x |
| coverage | 0.015276 | 0.014583 | 1.048x |
| p2r | 0.016206 | 0.016395 | 0.988x |
| r2r | 0.025392 | 0.025820 | 0.983x |
| r2p | 0.002371 | 0.002424 | 0.978x |
| packing | 0.007765 | 0.007873 | 0.986x |
| graph_total | 0.080025 | 0.081459 | 0.982x |
| reconstruction_map | 0.104549 | 0.102707 | 1.018x |
| continuous_total | 0.953724 | 0.424971 | 2.244x |

Remaining bottleneck: `support_ordering` (0.190695 s); ordering secondary=False.

## E32768_adaptive

| Stage | Full-order baseline median (s) | Lazy-prefix median (s) | Speedup |
|---|---:|---:|---:|
| support_ordering | 0.716373 | 0.320050 | 2.238x |
| cv_redistribution | 0.050107 | 0.049475 | 1.013x |
| regional_prepare | 0.002136 | 0.002262 | 0.945x |
| coverage | 0.048493 | 0.046036 | 1.053x |
| p2r | 0.029105 | 0.029074 | 1.001x |
| r2r | 0.010005 | 0.010569 | 0.947x |
| r2p | 0.003469 | 0.003625 | 0.957x |
| packing | 0.008083 | 0.008147 | 0.992x |
| graph_total | 0.110488 | 0.111544 | 0.991x |
| reconstruction_map | 0.111405 | 0.110634 | 1.007x |
| continuous_total | 0.996802 | 0.600888 | 1.659x |

Remaining bottleneck: `support_ordering` (0.320050 s); ordering secondary=False.

## Decision

Every valid32 prefix is bitwise equal to the historical full-order prefix and frozen support. No C/C++ or approximate q-cluster cache was implemented. Further support-order micro-optimization stops in this phase; U1 may proceed only because all hard gates passed.
