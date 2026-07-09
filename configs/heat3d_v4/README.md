# Heat3D V4 Config Registry

Read this directory only for V4 registry, YAML generation, dry-run, or launch
readiness tasks.

`v4_run_registry.json` is the authoritative registry. It stores one resolved
`baseline` plus per-run `overrides`; `runs.*` must not repeat full rows.
`run_registry.csv` is the only CSV file and is a resolved audit mirror generated
from the JSON registry. Do not add compact CSV variants. The CSV is split into
configuration fields first and result fields last.
`metrics_v0.json` is the V4 metrics contract referenced by the registry through
`metrics_profile` and `metrics_contract`.
`normalization_profile` is a configuration field. Missing or `legacy_zscore`
uses the legacy V1 controlled runner; `semantic_normalization_v1` selects the
V4 controlled runner wrapper.
The registry/CSV configuration fields also include provenance fields:
`runner_family`, `target_mode`, `bridge_policy`, `input_feature_schema`,
`coord_policy`, `extent_feature_policy`, `condition_feature_transform`,
`target_recovery_policy`, and
`feature_manifest_hash`. These are audit metadata written to generated YAML
`metadata`; `input_feature_schema`, `coord_policy`, and
`extent_feature_policy` are also mirrored into `dataset` and passed to the V4
runner when they are non-default. `feature_manifest_hash` may remain `planned`
until a real manifest hash writer exists.
For V4 semantic-normalization ablations, `condition_feature_transform` is also
mirrored into `dataset.condition_feature_transform` and passed to the V4 runner.
Supported semantic ablations are BC flags only, q only, k only, and full
semantic v1.
Dataset identity is explicit in the registry/CSV through `dataset_name`,
`subset_path`, `manifest_path`, and `split_map_path`; these are mirrored into
generated YAML `dataset` fields. V4 P3 candidate1024 formal configs use the
tracked train768/valid128/test128 split map. The older test-as-valid map is a
smoke legacy reference only.
V4 P5 clean-nohard configs use
`candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json`;
its regular train/valid/test keys contain no `physical_hard_keep`, while the
original hard samples remain under three explicit holdout/challenge keys.
Overrides may only use resolved configuration column names. To add another
controlled field, first extend the resolved audit CSV configuration columns and
checker; do not add arbitrary dotted YAML overrides.
The controlled loss-weight fields are `background_relative_weight`,
`background_over_weight`, `strong_q_weight`, and `hotspot_weight`; they map to
the matching `loss.*` YAML keys and default to `0.0`.
For V4P4 hotspot/strong-q planned configs, nonzero `hotspot_weight` or
`strong_q_weight` must be paired with `loss_mode=hotspot_strong_q`; under plain
`mse` those weights are not part of the runner's total loss.
Continuation runs use the controlled fields `init_checkpoint` and
`checkpoint_load_strict`; they map to `run.init_checkpoint` and
`run.checkpoint_load_strict`.
Sample-weight runs use the controlled fields `sample_weight_policy`,
`sample_weight_json`, `sample_weight_default`, and
`sample_weight_normalize`; they map to `run.sample_weight_*` YAML keys.
`sample_weight_policy=hard_sample_list` requires a tracked JSON weight file.
The optional upstream-onecycle fields `lr_init`, `lr_peak`, `lr_base`,
`lr_lowr`, `pct_start`, and `pct_final` map to `optimizer.*` YAML keys and
should stay blank for non-onecycle configs.

Result fields are CSV-only audit fields that are filled after training or
post-run review. `prepare_heat3d_v4_run.py` preserves existing result values
when regenerating the CSV mirror and leaves result fields blank for newly added
configs. `check_heat3d_v4_registry.py` requires result columns to exist but does
not treat blank result values as errors.

Result metric columns intentionally include V2/V3 audit axes: best/final
MSE/RMSE/MAE, raw DeltaT MSE/RMSE/MAE, valid_iid and stress scalar summaries,
field-shape diagnostics (`corr`, `amp`, variance, top-k), zRMSE, top5/top10,
strong-q, peak/p95/p99/hotspot diagnostics, low-DeltaT background
overprediction (`bin0`, `le0.05`), and final-probe
RMSE/relRMSE/Tmax/probe-family summaries. These columns are for post-run audit
entry; blank values are expected before a run is reviewed.
Additional split-aware audit columns after `result_notes` are also result
fields; they record valid/test IID scalar, shape, background, peak, and
final-probe summaries and must be preserved by the checker.

Workflow:

1. Register the run in `v4_run_registry.json` by adding only its overrides.
2. Update the resolved CSV audit mirror from the JSON registry.
3. Generate inherited YAML from `V4_base.yaml`; generated YAML stores only
   executable overrides that differ from the base.
4. Run `scripts/check_heat3d_v4_registry.py`; it validates the metrics contract,
   legal selection metric, provenance fields, registry mirror, generated YAML,
   seed fields, unmapped-field warnings, and path conflicts.
5. Run `scripts/prepare_heat3d_v4_run.py --dry-run` before any launch handoff;
   the dry-run output must show provenance fields, `normalization_profile`,
   dataset identity fields, `metrics_profile`, `metrics_contract`,
   `selection_metric`, and selected training script.
6. Start tmux training only when the current user request explicitly approves a
   launch on a named server. Report the log path for live output.

Remote run helpers:

- `scripts/heat3d_v4_remote_run.py --host devbox check`
- `scripts/heat3d_v4_remote_run.py --host devbox launch --config-id <config_id>`
- `scripts/heat3d_v4_remote_run.py --host devbox monitor --config-id <config_id>`
- `scripts/heat3d_v4_remote_run.py --host devbox sync-command --config-id <config_id> --target-host wsl2`
- `scripts/summarize_heat3d_v4_run_result.py --config-id <config_id> --update-csv`

`launch` fetches and fast-forwards the requested branch, starts a detached tmux
session, and runs the tracked YAML through `scripts/run_heat3d_v4_config.py`.
`sync-command` prints a non-overwriting server-to-server `rsync --ignore-existing`
command for the full `output_dir`; it does not sync outputs to the Codex host.

Standard local check:

```bash
python3 scripts/prepare_heat3d_v4_run.py --write-csv-mirror --write-yaml --dry-run
```

The V4 standard task is:

```text
coords + k(x) + q(x) + BC -> T(x)
```

Coordinate encoding fields:

- `node_coordinate_encoding=raw` is the default and preserves baseline node
  coordinate features.
- `raw_plus_fourier` appends Fourier features to the current
  `train_minmax_to_unit_box` coordinates while retaining raw `x,y,z`.
- This is not a physical coordinate scale fix and does not change Heat3D
  `periodic=False`, graph topology, edge indices, distance logic, solver, loss,
  or dataset.

Split-map fields:

- `split_map_path` is a resolved configuration field in the V4 registry/CSV.
  Post-training diagnostics should use the active split map explicitly instead
  of relying only on `sample_meta["split"]`.
- `configs/heat3d_v4/candidate1024_v0_train768_valid128_test128_stratified_seed0.json`
  is the formal candidate1024_v0 train/valid/test split map.
- `configs/heat3d_v4/candidate1024_v0_test_as_valid_iid_split_map.json` remains
  only as a legacy smoke bridge and should not be used for formal V4 P3 runs.
- `configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json`
  is the P5 clean-IID split. See `docs/v4_p5_clean_nohard_dataset.md` for the
  clean_iid, hard_challenge, and all_iid reporting relationship.
- V4 formal training defaults to `prediction_split=valid_iid`; use
  `prediction_split=all` only for explicit export/audit requests because it
  forces full `all_groups` construction.

Input feature and coordinate policy fields:

- Default `input_feature_schema=legacy_bc_flags`,
  `coord_policy=train_minmax_to_unit_box`, and `extent_feature_policy=none`
  preserve the V4P1_12 input path.
- `boundary_distance_replacement` removes
  `is_top/is_bottom/is_side/is_interior` and adds
  `d_xmin/d_xmax/d_ymin/d_ymax/d_bottom/d_top`; distances are normalized by the
  sample's per-axis physical extent before condition normalization.
- `sample_local_isotropic` normalizes each sample's `x/y/z` by one shared
  max-axis scale. For that policy, `Inputs.x_inp/x_out` and graph metadata are
  built from the same normalized coordinates.
- `log_extent_broadcast` appends
  `log_Lx/log_Ly/log_Lz/log_Lx_over_Lz/log_Ly_over_Lz` as condition features.

Decoder bypass fields:

- `decoder_bypass_mode=none` is the default and preserves the baseline model.
- `post_decoder_residual` adds an opt-in normalized-DeltaT residual after the
  decoder.
- `decoder_bypass_features=full_condition` resolves feature indices from
  `feature_names`; missing required condition features are an error.
- `zero_residual` initializes the bypass output layer to zero.

Metrics policy:

- default checkpoint selection is `valid_base_mse`;
- runner progress no longer emits the legacy `iid_err` label. Use
  `raw_rmse_K`, `recovered_T_rmse_K`, and `rel_rmse_v4_pct` instead.
- `mse`, `rmse`, and `mae` are overall model-performance report metrics;
- raw DeltaT metrics report physical-scale error and stay separate from
  normalized validation;
- final-probe/OOD, region/hotspot, and diagnostic metrics are report or
  diagnosis aids and do not replace default checkpoint selection;
- metrics should be computed per sample first, then summarized by split or group
  with mean, median, and standard deviation.

For the full metrics profile, see `docs/v4_metrics_contract.md`.
