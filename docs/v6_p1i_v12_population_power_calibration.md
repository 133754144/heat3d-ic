# V6-P1i v12 population power calibration

V12 replaces the single severity exponent with one fixed monotonic six-knot
population curve. The curve, top-h exponent and nonzero independent-jitter
width were optimized jointly against complete v5--v11 pilot populations.

All seven populations pass the frozen temperature and single-latent gates
counterfactually. The worst occupancy ratio is 4.5, core gap 6.818 K, full gap
13.965 K, severity Spearman 0.865, and single-latent R² 0.780.

V12 uses a new unseen seed. The rule never reads a case temperature, performs
no Rth inversion, and permits no filtering, replacement or target-dependent
split.
