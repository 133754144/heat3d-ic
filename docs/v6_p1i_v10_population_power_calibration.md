# V6-P1i v10 population power calibration

V10 keeps one population-level rule and uses a new unseen Sobol seed. It sets
base power to 2.7--10.7 W, the monotonic severity exponent to 0.86, and the
top-h exponent to 0.68. The latter is still weaker than the pre-v3 value 0.79,
but reduces the low-h tail exposed by v9.

The rule has the best aggregate frozen-gate score over completed v5--v9
populations: four pass all counterfactual temperature gates; v7 alone misses
the full-gap limit by 0.211 K. No case-specific Rth inversion, filtering,
replacement or target-dependent split is permitted.
