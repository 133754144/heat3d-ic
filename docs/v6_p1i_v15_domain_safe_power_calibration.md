# V6-P1i v15 domain-safe power calibration

The first formal1024_v1 input-only preflight found a 1.6089 W point below the
1.7 W literature floor. It ran no solver and produced no dataset.

V15 therefore returns to a new 128-sample pilot and recalibrates the single
global power rule using the completed v5--v12 and v14 populations. Unlike prior
calibrations, the rule is constrained analytically over every allowed
severity, top-h, and independent-jitter value. Its theoretical power range is
1.7282--19.4888 W, inside the frozen 1.7--20 W contract.

The aborted v13 partial population is excluded. V15 uses the unseen seed
`612818`; no v15 target is available before freeze, and the rule performs no
per-sample Rth inversion, filtering, or replacement.
