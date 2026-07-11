# V5 Gate 1 Final Correction Contract

This final correction preserves the old Gate 1 result at commit `6f9a4b2` and
adds a separate final table and closeout. It does not overwrite the prior
layer-averaged 1D CSV, JSON, or Markdown.

## Corrected `z_collapsed_1d_operator`

For each adjacent z pair and every x-y column, Gate 1 now uses the exact V4
z-face definition:

```text
G_column = harmonic_mean(kz_lower, kz_upper) * dx_cv * dy_cv / (z_upper - z_lower)
G_layer_face = sum_xy(G_column)
```

This uses actual node spacing, not a half-control-volume approximation. The
collapsed network retains the V4 source and boundary meanings: bottom rows are
Dirichlet replacement rows, so bottom q is excluded; all non-bottom `q*CV`
enters the RHS; and top Robin adds both `h*A` and
`h*A*(T_inf-T_bottom)` in relative-temperature form.

## Candidate And Unit Contract

The final comparison retains constant, power-only,
`q_rms_lz2_over_kz`, source-centroid two-path, legacy `P_array*R_series`, and
the old layer-averaged 1D proxy. It adds `z_collapsed_1d_operator`.

Raw columns carry their physical units: notably `raw_power_only_W` is W, while
both 1D proxy raw values are K. Every calibrated prediction remains named
`pred_<candidate>_K`.

## Selection And Closeout

Calibration is train-only; `valid_iid` selects; `hard_challenge_valid` is OOD
inspection; test roles are report-only. The closeout must contain paired
bootstrap comparisons of every physical candidate versus constant, the selected
candidate versus power-only, and winner versus runner-up. If the winner and
runner-up are statistically tied, only an operator-consistent, nonuniform-grid
generalizability rationale may break the tie.

The machine-readable contract is
`configs/heat3d_v5/v5_gate1_final_correction_contract.json`.
