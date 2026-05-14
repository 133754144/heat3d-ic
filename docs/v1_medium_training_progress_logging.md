# Heat3D v1 Medium Training Progress Logging

This note documents startup progress logging for the V1 medium controlled
training/export runner. It is diagnostic tooling only; it is not a formal
benchmark or model-performance claim.

## Motivation

Full1024 `medium1024_gapA` generation, generated-subset checks, label
diagnostics, metadata coverage, and a short 2-epoch training smoke have run on
SSH. The training chain is viable, but the runner previously printed its first
user-visible line only after dataset scanning, eager sample loading,
normalization, and grouped JAX array/graph construction. On a large subset this
made it hard to tell whether the run was loading data, initializing the model,
evaluating the first loss, running the first training step, exporting
predictions, or stalled.

## What The Runner Logs

`scripts/run_heat3d_v1_medium_controlled_training_export.py` now emits compact
stage markers before the first epoch report when progress logging is enabled:

- `[startup] script start ...`
- `[startup] output dir ready ...`
- `[startup] loading dataset ...`
- `[startup] dataset loaded: sample_count=... split_counts=...`
- `[startup] computing train-only target normalization ...`
- `[startup] target normalization done ...`
- `[startup] building grouped JAX arrays and graphs ...`
- `[startup] groups built ...`
- `[startup] initializing model parameters ...`
- `[startup] model parameters initialized`
- `[startup] computing initial train/valid losses ...`
- `[train] epoch loop start ...`
- `[train] epoch ... start lr=...`
- `[train] epoch ... metrics computed`
- `[export] building recovered predictions ...`
- `[export] writing run_config.json and loss_summary.json ...`
- `[export] saving predictions ...`
- `[done] script complete`

The existing `--log-mode compact`, `--log-mode full`, and `--log-mode quiet`
behavior remains. Progress logging is enabled by default for compact/full and is
suppressed in quiet mode. It can also be disabled explicitly with
`--no-progress-log`.

All runner output goes through a small flushed print wrapper, so progress lines
should appear promptly when the command is run through `tee`. For SSH runs,
`PYTHONUNBUFFERED=1 python -u ...` is still a useful belt-and-suspenders option
when diagnosing terminal buffering.

## Current Caution

The `medium1024_gapA` diversity diagnostics currently show substantial true
`q_field`, `k_field`, and `temperature` repetition. That means further long
training is not the next priority. The next research-stage task should improve
generator diversity before treating full1024 training results as meaningful
controlled experiments.
