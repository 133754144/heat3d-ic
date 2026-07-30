# V6-P1i formal1024 v0 closeout

Status: **generated completely; qualification failed**.

The frozen 1024-case Sobol design was solved once on local CPU. All 1024
samples and the pre-solve 768/128/128 split are retained. No sample was
filtered, replaced, or reassigned, and no seed or physical parameter was
changed after observing temperatures.

## Result

| item | result |
|---|---:|
| generation | 1024/1024 complete |
| elapsed | 2110.260 s |
| peak DeltaT min / median / max | 33.140 / 95.581 / 174.483 K |
| mean DeltaT min / median / max | 25.525 / 75.006 / 142.915 K |
| CV-RMS DeltaT min / median / max | 25.954 / 75.871 / 143.210 K |
| peak 30--150 K | 973/1024 = 95.020% |
| peak outside 20--180 K | 0 |
| 12-bin counts | 40, 87, 81, 90, 82, 90, 80, 87, 91, 84, 92, 69 |
| bin max/min ratio | 2.300 |
| KDE modes | 2 |
| max energy-balance relative error | 1.395e-10 |
| max linear residual | 1.222e-10 |

Two frozen gates failed:

- the largest adjacent sorted peak gap is 5.743 K, above the 5.0 K limit;
- the maximum pre-solve parameter KS is 0.177083, above 0.15. The worst
  comparison is `bottom_h_W_m2K`, train versus valid_iid.

Temperature split QC passed: maximum temperature-metric KS is 0.1328125 and
the maximum peak-median relative difference is 0.08110. This does not override
the failed pre-solve split gate.

## Auditor implementation amendment

The frozen auditor still contained a pilot-only `len(rows) == 128` assertion
and a fixed Markdown denominator. It failed before computing any metric. The
implementation was minimally amended to read `sample_count` from the already
frozen acceptance JSON. Physics, samples, split assignment, thresholds, and
metric formulas were not changed. The freeze manifest retains original hashes
and records the effective implementation hashes.

## Decision

`heat3d_v6_p1i_continuous_physics1024_v0` is an auditable failed
qualification artifact, not a training dataset. It must not be silently
repaired by deleting samples or changing the seed. Any successor requires a
new version and a new preregistration.

No training or model inference was run, and existing frozen V6 assets were not
modified.
