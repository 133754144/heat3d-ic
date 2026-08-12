# V6 P1i P7 fresh-batch closeout

All runs use frozen valid32 on WSL2. Accuracy is unchanged and reused from the frozen E16384 route. JIT, metrics, hashes, and serialization are outside fresh spans.

| system | batch/processes | wall (s) | samples/s | avg/case (s) |
|---|---:|---:|---:|---:|
| E16384 (fresh_distinct_case_batch) | 1 | 0.753846 | 1.327 | 0.753846 |
| E16384 (fresh_distinct_case_batch) | 2 | 1.277622 | 1.565 | 0.638811 |
| E16384 (fresh_distinct_case_batch) | 4 | 2.360189 | 1.695 | 0.590047 |
| E16384 (fresh_distinct_case_batch) | 8 | 4.498714 | 1.778 | 0.562339 |
| E16384 (fresh_distinct_case_batch) | 16 | 9.035549 | 1.771 | 0.564722 |
| E16384 (fresh_distinct_case_batch) | 32 | 17.523875 | 1.826 | 0.547621 |
| FVM (parallel_independent_known_topology_new_physics) | 1 | 100.266721 | 0.319 | 3.133335 |
| FVM (parallel_independent_known_topology_new_physics) | 2 | 63.055958 | 0.507 | 1.970499 |
| FVM (parallel_independent_known_topology_new_physics) | 4 | 58.857232 | 0.544 | 1.839288 |
| FVM (parallel_independent_known_topology_new_physics) | 8 | 61.762583 | 0.518 | 1.930081 |

Best fresh neural throughput: B32 = 1.826 samples/s. Saturated FVM: 4 processes = 0.544 samples/s. Semantically explicit throughput ratio = 3.36x.

Fresh neural throughput is dominated by CPU preprocessing, while resident throughput excludes that work. These figures are not interchangeable. No training or test/sealed access occurred.
