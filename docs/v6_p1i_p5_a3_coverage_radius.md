# V6 P1i P5-A3 coverage-radius closeout

## Profiling correction

The historical `coverage_radius_seconds` started from the beginning of regional
node preparation. The timer now starts only after regional preparation ends.
The two stages are therefore disjoint; candidate regional preparation remained
about 0.00228 s median in this run.

## Exact-equivalence

For B8192 and E32768 on frozen valid32, all 64 actual graphs passed:

- identical nearest-regional assignments;
- bitwise-identical coverage radii and radii SHA256;
- identical final graph hashes.

The reference and candidate both shortlist 16 float64 KD-tree neighbors and
reapply the frozen input-dtype distance/tie rule. The only implementation
difference is `cKDTree.query(workers=1)` versus `workers=-1`.

## Corrected timing

| Route | Reference coverage median (s) | Candidate median (s) | Speedup |
|---|---:|---:|---:|
| B8192 | 0.035153 | 0.019897 | 1.767x |
| E32768 | 0.113395 | 0.049144 | 2.307x |
| pooled | 0.086411 | 0.033740 | 2.561x |

The CPU-resident parallel exact path is promoted (`GO`). No prediction,
temperature, test or sealed role was accessed.
