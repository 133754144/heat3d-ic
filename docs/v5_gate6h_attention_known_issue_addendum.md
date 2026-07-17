# Gate 6H Attention Known-Issue Addendum

This is a post-freeze engineering addendum. It does not rewrite the Gate 6H
history or move `v5-gate6h-frozen`. No training has been started by this
change.

## Known issue and versioned fix

The historical `bugged_v1` QK regional schema used
`q >= quantile(q, 0.75)` for its fourth channel. Sparse source fields often
have a zero 75th percentile, so zero-source nodes are incorrectly included in
the `q_high` set. The old schema remains available and is still the default so
old checkpoints replay unchanged.

The sole repaired candidate, `V4P5_32_gate6h_attention_sparse_safe_v2_e600`,
explicitly selects `sparse_safe_v2`:

- remove `q_high_inverse_kz_overlap`;
- use the regional mean of `q > 1e-12` as `source_present_fraction`;
- rename `source_z_normalized` to `region_z_normalized` without changing its
  values;
- retain the other continuous q/k and BC features and the 11-wide schema.

The input-only audit read `coords.npy`, `k_field.npy`, and `q_field.npy` for
672 train plus 128 `valid_iid` samples. It read no temperature/target files.
Across those 800 samples:

- 82.125% had a zero q 75th percentile;
- 86.994% of zero-q nodes were false-positive `q_high` nodes;
- old overlap covered 40.018% of all nodes;
- 88.642% of old overlap nodes were outside the actual source.

The new source-presence variance is `0.09843`, versus `0.24004` for the old
overlap feature. Its Pearson correlation is `0.91024` with continuous
`log1p_q_relative`, `-0.00029` with `log_inverse_kz_relative`, and `0.69976`
with `log1p_q_inverse_kz_relative`. Full variances and correlation matrices
are in
`configs/heat3d_v5/gate6h/attention_sparse_safe_v2_feature_audit.json`.

## Candidate contract

V32 inherits the frozen V13 runtime contract: clean data and split, RIGNO
capacity, graph, losses `1.5/0.5/1/1`, AdamW warmup-cosine schedule, seeds,
train B28, and e600. Validation and prediction use B32.

Its only effective architecture change from V13 is
`scale_attention_mode=physics_gate`; pooled-latent stop-gradient is explicitly
false. Scale pooling remains `mean` and scale-head depth remains 1. The
versioned input-schema change is `qk_region_feature_version=sparse_safe_v2`.

Checkpoint contract:

- `params_best.pkl`: legacy `valid_base_mse` best;
- `params_best_valid_point_global.pkl`: true-RMS point-global best and the
  only checkpoint eligible for advancement;
- `params_best_valid_sample_first.pkl`: sample-first CV-relative best, with
  raw CV RMSE K as the preregistered tie-break;
- `params_final.pkl`: final e600 diagnostic.

After all four checkpoint choices are frozen, valid-only diagnostics record
normalized attention entropy, maximum attention weight, and Pearson
correlations of attention weight with source fraction and the three continuous
q/k features. Test, hard roles, and sealed IID are forbidden until checkpoint
freezing; none was accessed in this preparation.

## Historical comparison limit

V13 remains the frozen historical reference:

- valid point-global true-RMS relative RMSE: `23.70068%`;
- valid sample-first CV-relative RMSE: `20.31646%`;
- valid raw CV-weighted RMSE: `0.167982 K`.

This is a non-contemporaneous baseline comparison because V13 was trained
under the earlier code revision. V32 must use the frozen V5 metric formulas,
and no test result may select a checkpoint or change the promotion decision.

## Manual launch

After pulling the preparation commit and activating `rigno`, the authorized
manual command is:

```bash
python scripts/run_heat3d_v4_config.py --config /home/xyh/myCodeGitOnly/heat3d-ic/configs/heat3d_v5/generated/V4P5_32_gate6h_attention_sparse_safe_v2_e600.yaml
```
