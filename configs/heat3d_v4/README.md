# Heat3D V4 Config Registry

Read this directory only for V4 registry, YAML generation, dry-run, or launch
readiness tasks.

`v4_run_registry.json` is the authoritative registry. It stores one resolved
`baseline` plus per-run `overrides`; `runs.*` must not repeat full rows.
`run_registry.csv` is the only CSV file and is a resolved audit mirror generated
from the JSON registry. Do not add compact CSV variants.
`metrics_v0.json` is the V4 metrics contract referenced by the registry through
`metrics_profile` and `metrics_contract`.
Overrides may only use resolved CSV column names. To add another controlled
field, first extend the resolved audit CSV columns and checker; do not add
arbitrary dotted YAML overrides.

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
