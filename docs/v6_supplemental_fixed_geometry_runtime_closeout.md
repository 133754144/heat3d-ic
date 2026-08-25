# V6/P1i fixed-geometry supplemental runtime closeout

Status: **COMPLETED / PASS**. This is a train-input-only runtime supplement; it does not alter frozen V6/P1i scientific results and does not create an FVM speedup claim.

## Frozen design

- `v6p1if1_0079`: source count 3, k regions 7
- `v6p1if1_0971`: source count 5, k regions 4
- `v6p1if1_0393`: source count 7, k regions 3
- `v6p1if1_0056`: source count 10, k regions 5

`K_only` keeps q byte-exact and changes k at preregistered formal-distribution quantiles. `K_plus_Q_scale` uses the same k sweep and positive alpha={0.8,0.95,1.05,1.2}; q mask and normalized spatial distribution are invariant. All inputs remain inside the formal P1i contract.

## Correctness

All 4 routes passed 32/32 cases. Cached and standard paths have exact support/CV, graph, reconstruction-map and prepared-payload hashes. Checkpoint parameters are unchanged; no temperature, valid, test or sealed labels were opened.

## Runtime results (seconds per case)

| Sweep | Route | Reuse mode | Median | P95 | Samples/s | Speedup vs fresh | Median bottleneck |
|---|---|---|---:|---:|---:|---:|---|
| K_only | E16384_reconstruction | fresh_new_case | 0.5441 | 0.5829 | 1.83 | 1.00x | support_plus_cv |
| K_only | E16384_reconstruction | graph_only_reuse | 0.2706 | 0.2778 | 3.94 | 2.01x | reconstruction_map |
| K_only | E16384_reconstruction | full_static_reuse | 0.0669 | 0.0709 | 15.04 | 8.14x | query_dynamic_pack |
| K_plus_Q_scale | E16384_reconstruction | fresh_new_case | 0.6654 | 0.6843 | 1.59 | 1.00x | support_plus_cv |
| K_plus_Q_scale | E16384_reconstruction | graph_only_reuse | 0.2692 | 0.2787 | 3.92 | 2.47x | reconstruction_map |
| K_plus_Q_scale | E16384_reconstruction | full_static_reuse | 0.0646 | 0.0685 | 15.54 | 10.30x | query_dynamic_pack |
| K_only | U_v2_16384_reconstruction | fresh_new_case | 0.5390 | 0.5545 | 1.85 | 1.00x | support_plus_cv |
| K_only | U_v2_16384_reconstruction | graph_only_reuse | 0.2066 | 0.2111 | 4.85 | 2.61x | reconstruction_map |
| K_only | U_v2_16384_reconstruction | full_static_reuse | 0.0682 | 0.0709 | 14.71 | 7.90x | h2d_enqueue |
| K_plus_Q_scale | U_v2_16384_reconstruction | fresh_new_case | 0.5400 | 0.5480 | 1.85 | 1.00x | support_plus_cv |
| K_plus_Q_scale | U_v2_16384_reconstruction | graph_only_reuse | 0.2054 | 0.2096 | 4.88 | 2.63x | reconstruction_map |
| K_plus_Q_scale | U_v2_16384_reconstruction | full_static_reuse | 0.0680 | 0.0730 | 14.65 | 7.94x | h2d_enqueue |
| K_only | U_v2_direct240825 | fresh_new_case | 0.9668 | 1.1963 | 1.01 | 1.00x | query_graph |
| K_only | U_v2_direct240825 | graph_only_reuse | 0.1274 | 0.1455 | 7.77 | 7.59x | query_group_pack |
| K_only | U_v2_direct240825 | full_static_reuse | 0.1237 | 0.1405 | 8.00 | 7.81x | query_dynamic_pack |
| K_plus_Q_scale | U_v2_direct240825 | fresh_new_case | 0.9433 | 1.1508 | 1.04 | 1.00x | query_graph |
| K_plus_Q_scale | U_v2_direct240825 | graph_only_reuse | 0.1212 | 0.1283 | 8.17 | 7.78x | query_group_pack |
| K_plus_Q_scale | U_v2_direct240825 | full_static_reuse | 0.1157 | 0.1200 | 8.63 | 8.15x | query_dynamic_pack |
| K_only | E240825_direct_control | fresh_new_case | 1.0030 | 1.0827 | 0.98 | 1.00x | query_graph |
| K_only | E240825_direct_control | graph_only_reuse | 0.4262 | 0.5309 | 2.25 | 2.35x | query_group_pack |
| K_only | E240825_direct_control | full_static_reuse | 0.3867 | 0.4843 | 2.49 | 2.59x | query_dynamic_pack |
| K_plus_Q_scale | E240825_direct_control | fresh_new_case | 1.0033 | 1.0947 | 0.99 | 1.00x | query_graph |
| K_plus_Q_scale | E240825_direct_control | graph_only_reuse | 0.4272 | 0.5187 | 2.26 | 2.35x | query_group_pack |
| K_plus_Q_scale | E240825_direct_control | full_static_reuse | 0.3851 | 0.4616 | 2.49 | 2.61x | query_dynamic_pack |

## Static setup and amortization

- `E16384_reconstruction`: setup median 6.750 s; break-even repeated cases [12, 15].
- `U_v2_16384_reconstruction`: setup median 5.058 s; break-even repeated cases [11].
- `U_v2_direct240825`: setup median 5.221 s; break-even repeated cases [7].
- `E240825_direct_control`: setup median 6.478 s; break-even repeated cases [11].

The CSV includes amortized latency at 4, 16, and 100 repeated cases. 

## Prior devbox continuity check

- `E16384_reconstruction`: current 0.544 s vs historical 0.832 s (0.654x).
- `U_v2_16384_reconstruction`: current 0.539 s vs historical 0.811 s (0.665x).
- `U_v2_direct240825`: current 0.967 s vs historical 1.171 s (0.826x).
- `E240825_direct_control`: current 1.003 s vs historical 1.188 s (0.844x).

Fresh times are compared with the prior devbox lifecycle table; the comparison is hardware/runtime continuity only because the sample populations differ.
