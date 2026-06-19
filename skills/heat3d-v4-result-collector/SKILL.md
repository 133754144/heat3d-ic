---
name: heat3d-v4-result-collector
description: Use after a Heat3D V4 run finishes to read loss_summary.json and related diagnostic payloads, then update only configs/heat3d_v4/run_registry.csv result_* columns. Use for devbox/WSL/local result collection, CSV audit refresh, and final V4 run reporting without editing model configuration fields.
---

# Heat3D V4 Result Collector

## Contract

Read `AGENTS.md`, workflow Quick Contract, `WF-EVAL`, `WF-SYNC` when syncing
artifacts, and `WF-REPORT`. This skill reads existing run outputs; it must not
train, evaluate, or change model/config fields unless the user explicitly asks.

## Workflow

1. Identify `config_id` and run location from `run_registry.csv`.
2. If collecting on the remote, activate `rigno` first and run:

```bash
python -B scripts/summarize_heat3d_v4_run_result.py --config-id <config_id> --source-label devbox
```

3. To update the tracked CSV from an available run directory:

```bash
python3 -B scripts/summarize_heat3d_v4_run_result.py --config-id <config_id> --source-label <local-or-remote-label> --update-csv
```

Use `--run-dir <path>` only when the output directory differs from the registry.

## Result Columns

The script updates `RESULT_FIELDNAMES` only. It preserves all configuration
columns and leaves unavailable result fields blank.

It fills direct `loss_summary.json` fields for status, source, commit, run/log
paths, checkpoints, best epoch, valid/base MSE, RMSE, raw DeltaT MSE/RMSE,
valid_iid, valid_stress, hotspot, and diagnostics/final-probe status.

When `loss_summary.json` references post-training diagnostics or final-probe
metrics, it also fills available field-shape, region, bin0/le0.05, top-k,
zRMSE, peak, and probe RMSE columns. For unlabeled diagnostics, prefer the
`best` entry and fall back to `final`.

## Validation

Run after CSV updates:

```bash
python3 -B scripts/check_heat3d_v4_registry.py
git diff --check
```

Do not commit `output/`, checkpoints, predictions, logs, or synced artifacts.
