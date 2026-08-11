# V6 P1i P5 adaptive preprocessing closeout

Status: **PASS**. Frozen valid32; no inference, training, test or sealed access.

## B8192_adaptive

| Stage | Reference median (s) | Candidate median (s) | Speedup |
|---|---:|---:|---:|
| support_ordering | 2.138885 | 0.716342 | 2.986x |
| cv_redistribution | 0.053047 | 0.040834 | 1.299x |
| regional_prepare | 0.002010 | 0.001785 | 1.126x |
| coverage | 0.037789 | 0.015186 | 2.488x |
| p2r | 0.016904 | 0.015614 | 1.083x |
| r2r | 0.025308 | 0.025034 | 1.011x |
| r2p | 0.017007 | 0.002336 | 7.280x |
| reconstruction_map | 0.141874 | 0.101904 | 1.392x |
| packing | 0.004104 | 0.007523 | 0.545x |
| graph_total | 0.109973 | 0.073925 | 1.488x |
| total_adaptive_preprocessing | 2.443445 | 0.932484 | 2.620x |

Remaining largest named stage: `support_ordering` (0.716342 s median).

## E32768_adaptive

| Stage | Reference median (s) | Candidate median (s) | Speedup |
|---|---:|---:|---:|
| support_ordering | 2.138885 | 0.716342 | 2.986x |
| cv_redistribution | 0.072481 | 0.047186 | 1.536x |
| regional_prepare | 0.002455 | 0.002018 | 1.216x |
| coverage | 0.119554 | 0.044784 | 2.670x |
| p2r | 0.029521 | 0.027672 | 1.067x |
| r2r | 0.010399 | 0.010093 | 1.030x |
| r2p | 0.029278 | 0.004661 | 6.281x |
| reconstruction_map | 0.175206 | 0.110678 | 1.583x |
| packing | 0.004926 | 0.008044 | 0.612x |
| graph_total | 0.203357 | 0.102751 | 1.979x |
| total_adaptive_preprocessing | 2.588093 | 0.976452 | 2.651x |

Remaining largest named stage: `support_ordering` (0.716342 s median).

## Timing semantics and decision

Reference and candidate graph paths were warmed separately for every exact
shape. `graph_total` is one continuous `build_metadata` span followed by a full
metadata `block_until_ready`; the total is ordering + CV + synchronized graph +
reconstruction map. Qualification, hashes, inference and serialization are
excluded.

All 64 route/sample cells passed support-order, CV, canonical-graph and
reconstruction-map exact gates. Candidate packing alone regressed by about 3 ms
and is therefore **NO-GO as a standalone optimization**. Exact reverse reuse is
retained because synchronized graph total still improves 1.49x for B8192 and
1.98x for E32768. Keep the CPU sparse path; do not resume GPU-tiled or
graph-policy search. The next justified engineering step is batch/offline
parallel support-order preparation; inference batching remains a separate later
study.
