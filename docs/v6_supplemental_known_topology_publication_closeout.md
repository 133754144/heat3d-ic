# V6 supplemental known-topology/new-physics closeout

Execution commit: `8a812619ab0112b4ecfc37ef18189f731180059d`. S0/S1/S2/S3 all PASS. The historical `171cd5e...` timing attempt remains invalidated and none of its timing results are reused.

The benchmark uses four frozen train-only geometries. It does not train, read labels, or access valid/test/sealed roles. Fresh and known-topology spans retain the frozen publication boundary from in-memory k/q/BC to synchronized 240825-node output.

| Sweep | Route | Fresh median/p95 (s) | Known median/p95 (s) | Known throughput (sample/s) | Median speedup |
|---|---|---:|---:|---:|---:|
| K_only | E16384_reconstruction | 0.881575 / 0.936587 | 0.110406 / 0.117190 | 9.018837 | 7.985x |
| K_only | U_v2_16384_reconstruction | 0.833712 / 0.860644 | 0.091892 / 0.094002 | 10.867798 | 9.073x |
| K_only | U_v2_direct240825 | 1.356760 / 1.631548 | 0.214807 / 0.250510 | 4.589252 | 6.316x |
| K_only | E240825_direct_control | 1.252546 / 1.411133 | 0.484351 / 0.581549 | 1.972401 | 2.586x |
| K_plus_Q_scale | E16384_reconstruction | 0.879804 / 0.932007 | 0.111101 / 0.114763 | 8.964225 | 7.919x |
| K_plus_Q_scale | U_v2_16384_reconstruction | 0.831805 / 0.864915 | 0.092380 / 0.096610 | 10.827130 | 9.004x |
| K_plus_Q_scale | U_v2_direct240825 | 1.477953 / 1.885117 | 0.220280 / 0.267383 | 4.454772 | 6.709x |
| K_plus_Q_scale | E240825_direct_control | 1.257209 / 1.346801 | 0.485888 / 0.595347 | 1.950704 | 2.587x |

Fresh E16384 is dominated by support/CV, group packing, and reconstruction-map construction; known-topology E16384 is dominated by dynamic anchor lookup and frozen packing. Fresh direct routes are dominated by query-graph construction; after reuse, direct routes are dominated by dynamic query packing and neural forward.

No resident optimization, FVM, training, accuracy selection, or label-bearing evaluation was performed.
