# V6 authoritative valid32 publication timing

- Status: `collected_authoritative_valid32_without_pooled_96`
- Population: frozen valid32; test/sealed unopened.
- Fresh/Q1 CI: paired 32-sample workload bootstrap within each seed; seed 20260821; 20,000 resamples.
- Three lifecycle repeats and Q2/B16→B32: median and min–max only; no n=3 bootstrap CI.

## Three-lifecycle summary

| Route | Metric | Median | Min | Max |
|---|---|---:|---:|---:|
| E16384_reconstruction | B16_to_B32_marginal | 0.538317 | 0.536566 | 0.558185 |
| E16384_reconstruction | Q1 | 0.831871 | 0.820655 | 0.862344 |
| E16384_reconstruction | Q2_submit | 1.12341 | 1.0926 | 1.12352 |
| E16384_reconstruction | Q2_throughput | 1.80534 | 1.78796 | 1.82232 |
| E16384_reconstruction | fresh | 0.831871 | 0.820655 | 0.862344 |
| E240825_direct_control | B16_to_B32_marginal | 0.635957 | 0.616885 | 0.691596 |
| E240825_direct_control | Q1 | 1.18803 | 1.17494 | 1.19874 |
| E240825_direct_control | Q2_submit | 1.31284 | 1.2568 | 1.35384 |
| E240825_direct_control | Q2_throughput | 1.50077 | 1.44358 | 1.55495 |
| E240825_direct_control | fresh | 1.18803 | 1.17494 | 1.19874 |
| FVM240825_reference | B16_to_B32_marginal | 1.04815 | 1.03448 | 1.06694 |
| FVM240825_reference | Q1 | 1.43313 | 1.42125 | 1.44096 |
| FVM240825_reference | Q2_submit | 2.08769 | 2.03809 | 2.10687 |
| FVM240825_reference | Q2_throughput | 0.947023 | 0.932219 | 0.960566 |
| FVM240825_reference | fresh | 1.43313 | 1.42125 | 1.44096 |
| U_v2_16384_reconstruction | B16_to_B32_marginal | 0.530551 | 0.522555 | 0.535492 |
| U_v2_16384_reconstruction | Q1 | 0.810669 | 0.807535 | 0.815 |
| U_v2_16384_reconstruction | Q2_submit | 1.06859 | 1.06494 | 1.07469 |
| U_v2_16384_reconstruction | Q2_throughput | 1.86883 | 1.85598 | 1.88094 |
| U_v2_16384_reconstruction | fresh | 0.810669 | 0.807535 | 0.815 |
| U_v2_direct240825 | B16_to_B32_marginal | 0.761542 | 0.742658 | 0.762661 |
| U_v2_direct240825 | Q1 | 1.17107 | 1.15102 | 1.19563 |
| U_v2_direct240825 | Q2_submit | 1.48058 | 1.47969 | 1.48299 |
| U_v2_direct240825 | Q2_throughput | 1.3266 | 1.31692 | 1.34364 |
| U_v2_direct240825 | fresh | 1.17107 | 1.15102 | 1.19563 |

## Paired speedups

Speedup is paired by sample ID and seed before any across-seed summary.

