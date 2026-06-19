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
Overrides may only use resolved configuration column names. To add another
controlled field, first extend the resolved audit CSV configuration columns and
checker; do not add arbitrary dotted YAML overrides.

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

Workflow:

1. Register the run in `v4_run_registry.json` by adding only its overrides.
2. Update the resolved CSV audit mirror from the JSON registry.
3. Generate inherited YAML from `V4_base.yaml`; generated YAML stores only
   executable overrides that differ from the base.
4. Run `scripts/check_heat3d_v4_registry.py`; it validates the metrics contract,
   legal selection metric, registry mirror, generated YAML, seed fields,
   unmapped-field warnings, and path conflicts.
5. Run `scripts/prepare_heat3d_v4_run.py --dry-run` before any launch handoff;
   the dry-run output must show `metrics_profile`, `metrics_contract`, and
   `selection_metric`.
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

Metrics policy:

- default checkpoint selection is `valid_base_mse`;
- `mse`, `rmse`, and `mae` are overall model-performance report metrics;
- raw DeltaT metrics report physical-scale error and stay separate from
  normalized validation;
- final-probe/OOD, region/hotspot, and diagnostic metrics are report or
  diagnosis aids and do not replace default checkpoint selection;
- metrics should be computed per sample first, then summarized by split or group
  with mean, median, and standard deviation.

For the full metrics profile, see `docs/v4_metrics_contract.md`.
