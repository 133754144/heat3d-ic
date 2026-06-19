# Heat3D V4 Config Registry

Read this directory only for V4 registry, YAML generation, dry-run, or launch
readiness tasks.

`v4_run_registry.json` is the authoritative registry. It stores one resolved
`baseline` plus per-run `overrides`; `runs.*` must not repeat full rows.
`run_registry.csv` is the only CSV file and is a resolved audit mirror generated
from the JSON registry. Do not add compact CSV variants.
Overrides may only use resolved CSV column names. To add another controlled
field, first extend the resolved audit CSV columns and checker; do not add
arbitrary dotted YAML overrides.

Workflow:

1. Register the run in `v4_run_registry.json` by adding only its overrides.
2. Update the resolved CSV audit mirror from the JSON registry.
3. Generate inherited YAML from `V4_base.yaml`; generated YAML stores only
   executable overrides that differ from the base.
4. Run `scripts/check_heat3d_v4_registry.py`.
5. Run `scripts/prepare_heat3d_v4_run.py --dry-run` before any launch handoff.
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
