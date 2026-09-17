# P23-R Table B — valid-only full-field transfer

Status: **`BLOCKED_TABLE_B_NOT_FROZEN`**.

The common domain and metrics are frozen at 128 `valid_iid` cases and
240,825 nodes per case.  Therm-FM seeds 0/1/2 were replayed and pass the P22
metric sanity check; the available seed-mean diagnostics are recorded in the
JSON receipt.  Heat3D V6 seed0 was replayed through the original deterministic
V6 mapping, but the exact frozen seed1/seed2 parameter artifacts are missing.

Therefore a three-seed mean±SD, paired bootstrap, and Table B ranking are not
emitted.  Recovering those exact artifacts is the only next action; no
retraining, substitution, or test unlock is allowed.
