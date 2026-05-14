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
- `[startup] group build train: start samples=...`
- `[startup] group build train: sample scan grouped=...`
- `[startup] group build train: group ... arrays+graph start ...`
- `[startup] group build train: group ... arrays+graph built ...`
- `[startup] group build valid: ...`
- `[startup] group build all: ...`
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

`--progress-detail` controls the grouped JAX array/graph startup logs:

- `off`: only the outer `building grouped...` and `groups built...` messages.
- `basic`: default; prints train/valid/all group-build starts, sample-scan
  completion, and per-group arrays+graph start/done messages.
- `verbose`: additionally prints sample-scan checkpoints such as
  `256/1024`, `512/1024`, `768/1024`, and `1024/1024` for full1024 runs.

The long full1024 startup stage mainly calls the small-train helper logic that
groups examples by shape/signature, builds metadata signatures from coordinates,
stacks condition/target arrays into grouped JAX arrays, and constructs graph
topology for each group. The new logging wraps that path without changing the
arrays, graph construction, loss, optimizer, normalization, or predictions
schema.

The runner also emits a final startup timing line similar to:

`[startup-summary] dataset_load=... normalization=... group_build=... model_init=... initial_loss=...`

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
