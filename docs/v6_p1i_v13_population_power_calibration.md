# V6-P1i v13 population power calibration

V13 retains one monotonic six-knot population power curve and continuous
independent jitter. It recalibrates the global rule against the eight complete
v5--v12 pilot populations after v12 failed only the frozen histogram occupancy
ratio gate.

The optimization was seeded with `20260731` and never used v13 outputs. All
eight completed populations pass the frozen gates counterfactually with
conservative margins: worst nonzero-bin ratio 3.2, minimum bin count 5,
maximum core gap 5.131 K, maximum full gap 11.958 K, and minimum primary-range
fraction 93.75%.

V13 uses the unseen Sobol seed `612815`. The rule is fixed before generation,
does not read a case temperature, performs no sample-specific thermal
resistance inversion, and permits no filtering or replacement.

The first input-only preflight found a 1.6687 W design below the 1.7 W
literature-contract floor. Before freeze and before any solve, the lowest
base-power knot was raised from 3.17294 W to 3.24 W. This adjustment used no
temperature or other target value.
