# V5 Gate 4A Offline Learned Scale-Correction Contract

Gate 4A is an offline feasibility study. It keeps the V4P5_02 shape frozen and
does not edit RIGNO, the model, the loss, formal training configuration, P5
data, labels, or split assignment.

## Frozen Definitions

`s_phys` is the uncalibrated Gate 1 corrected
`raw_z_collapsed_1d_operator_K`. The supervised scalar target is
`delta_s = log(s_true / s_phys)`, and each correction uses
`s_hat = s_phys * exp(delta_s_hat)`. A frozen V4 best/final shape is combined
with `s_hat` in raw temperature space before applying only the prescribed
Dirichlet projection.

## Inputs And Leakage Boundary

The global set contains effective total power, `s_phys`, q-weighted local and
inverse conductivity, q/low-k overlap, source concentration and centroid,
source-layer and global conductivity descriptors, anisotropy, geometry, and
BC parameters. The optional latent is a mean pool of frozen V4
`rnodes_processed` features. The exporter verifies that the associated model
prediction reproduces the frozen raw-temperature archive before accepting it.

No target scale, target residual, oracle metric, Gate 1 calibrated prediction,
or other label-derived field may be an input. Each model learns input
normalization only from the rows named by its protocol fit roles.

## Protocols

`clean_only_zero_shot` fits only `train` and chooses on `valid_iid`; hard roles
are post-selection reports. `hard_adapted` fits `train + hard_train_holdout`,
chooses on `hard_challenge_valid`, and must retain the `valid_iid` clean guard.
`test_iid` and `hard_challenge_test` never enter feature construction,
standardization, hyperparameter choice, thresholds, or model selection.

The comparison includes physics-only, ridge, and a small MLP with global
physics features alone or those features plus frozen pooled latent. The
machine-readable contract fixes the candidate grid, selection, clean guard,
and paired-bootstrap rules.
