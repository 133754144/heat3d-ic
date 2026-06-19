# Heat3D V4 Config Registry

Read this directory only for V4 registry, YAML generation, dry-run, or launch
readiness tasks.

`v4_run_registry.json` is the authoritative registry. `run_registry.csv` is an
audit mirror generated from the JSON registry and checked for exact row parity.

Workflow:

1. Register the run in `v4_run_registry.json`.
2. Update the CSV audit mirror from the JSON registry.
3. Generate inherited YAML from `V4_base.yaml`; generated YAML stores only fields
   that differ from the base.
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
