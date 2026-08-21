# V6 Authoritative valid32 Attempt 2 Closeout

## Decision

- `envelope_qualification = GO` (the two frozen full-valid32 graph runs were reused)
- `padding_numerical_equivalence = GO`
- `ready_for_authoritative_valid32 = GO` before launch
- `publication_timing_freeze = NO_GO`

No training, accuracy tuning, test access, sealed-set access, checkpoint change,
graph-policy change, or route-semantic change occurred.

## Final padding gate

Padding capacity is the elementwise maximum of the previous same-semantic
frozen capacity, the qualified valid32 maximum, and the target-free train-only
warmup.  It is monotonic and changes only masked dummy capacity.  The gate used
`v6p1if1_0308` and max-edge witness `v6p1if1_0029` across all four neural
routes.  Every required comparison passed the frozen rule:

`max_abs_K <= max(1e-3, 20 * same_shape_floor_K)`.

Numeric results were persisted before gate evaluation.  Padding-adjusted
prepared-payload golden records were regenerated without model inference, and
their real graph hashes match the historical frozen graph witnesses.

## Authoritative Attempt 2

Attempt 2 started from a new output directory and did not reuse Attempt 1.  It
completed the five Serial cells for seed `20260814`.  The sixth process,
`E16384_reconstruction / 20260814 / Q2`, failed while assembling its JSON result:

`IndexError: index -1 is out of bounds for axis 0 with size 0`.

The Q2-only process has no Serial resident/cache-hot pool, but the E result
assembly path unconditionally called the statistics helper on the empty
`valid_resident_values` array.  This is a result-assembly contract failure, not
a padding, accuracy, or measured-performance rejection.

Per the preregistered contract, the failed cell was not repaired or rerun and
the remaining 24 cells were not started.  Attempt 2 is therefore:

- `attempted=true`
- `completed=false`
- `publication_results_generated=false`
- completed cells: `5/30`
- attempted processes: `6/30`

The formal collector and authoritative result checker are ineligible and were
not run.  No latency or speedup from this incomplete attempt may be published.

## Evidence

- padding gate: `configs/heat3d_v6_p1i/v6_p1i_publication_final_padding_gate_raw/`
- final-padding seal: `configs/heat3d_v6_p1i/v6_p1i_publication_benchmark_pre_measurement_seal_final_padding.json`
- Attempt 2 raw/logs: `configs/heat3d_v6_p1i/v6_p1i_publication_authoritative_valid32_failure_72e2308_attempt2_raw/`
- failure manifest: `configs/heat3d_v6_p1i/v6_p1i_publication_authoritative_valid32_failure_72e2308_attempt2_manifest.json`
- machine closeout: `configs/heat3d_v6_p1i/v6_p1i_publication_authoritative_valid32_attempt2_closeout.json`
