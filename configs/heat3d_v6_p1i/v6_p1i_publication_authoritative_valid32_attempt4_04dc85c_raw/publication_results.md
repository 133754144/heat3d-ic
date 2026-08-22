# V6 authoritative valid32 publication timing

- Status: `collected_authoritative_valid32_without_pooled_96`
- Population: frozen valid32; test/sealed unopened.
- Fresh/Q1 CI: paired 32-sample workload bootstrap within each seed; seed 20260821; 20,000 resamples.
- Three lifecycle repeats and Q2/B16→B32: median and min–max only; no n=3 bootstrap CI.

## Three-lifecycle summary

| Route | Metric | Median | Min | Max |
|---|---|---:|---:|---:|
| E16384_reconstruction | B16_to_B32_marginal | 0.567698 | 0.559984 | 0.586382 |
| E16384_reconstruction | Q1 | 0.883247 | 0.879188 | 0.892221 |
| E16384_reconstruction | Q2_submit | 1.15231 | 1.13135 | 1.1577 |
| E16384_reconstruction | Q2_throughput | 1.72633 | 1.69727 | 1.72653 |
| E16384_reconstruction | fresh | 0.883247 | 0.879188 | 0.892221 |
| E240825_direct_control | B16_to_B32_marginal | 0.730085 | 0.712431 | 0.755867 |
| E240825_direct_control | Q1 | 1.31023 | 1.29266 | 1.31132 |
| E240825_direct_control | Q2_submit | 1.54008 | 1.44667 | 1.54578 |
| E240825_direct_control | Q2_throughput | 1.31089 | 1.28385 | 1.34692 |
| E240825_direct_control | fresh | 1.31023 | 1.29266 | 1.31132 |
| FVM240825_reference | B16_to_B32_marginal | 1.15262 | 1.12876 | 1.1653 |
| FVM240825_reference | Q1 | 1.70068 | 1.6966 | 1.71021 |
| FVM240825_reference | Q2_submit | 2.29441 | 2.28499 | 2.31038 |
| FVM240825_reference | Q2_throughput | 0.857039 | 0.856195 | 0.861036 |
| FVM240825_reference | fresh | 1.70068 | 1.6966 | 1.71021 |
| U_v2_16384_reconstruction | B16_to_B32_marginal | 0.561611 | 0.54337 | 0.568663 |
| U_v2_16384_reconstruction | Q1 | 0.866177 | 0.865998 | 0.879279 |
| U_v2_16384_reconstruction | Q2_submit | 1.13436 | 1.11493 | 1.14195 |
| U_v2_16384_reconstruction | Q2_throughput | 1.75948 | 1.74886 | 1.79448 |
| U_v2_16384_reconstruction | fresh | 0.866177 | 0.865998 | 0.879279 |
| U_v2_direct240825 | B16_to_B32_marginal | 0.855329 | 0.843338 | 0.878692 |
| U_v2_direct240825 | Q1 | 1.30909 | 1.30156 | 1.3957 |
| U_v2_direct240825 | Q2_submit | 1.71634 | 1.70902 | 1.75954 |
| U_v2_direct240825 | Q2_throughput | 1.14757 | 1.13157 | 1.16905 |
| U_v2_direct240825 | fresh | 1.30909 | 1.30156 | 1.3957 |

## Paired speedups

Speedup is paired by sample ID and seed before any across-seed summary.

