# Heat3D v3 Checkpoint / Warm-Start / Upstream Audit Note

## Current Gap

Existing B6/S2/S3 long runs saved predictions and diagnostics, but did not save
model params. They cannot be re-run on a new target probe or new test set
without repeating training.

## New Default

Future controlled runner outputs now save:

- `params_final.pkl`
- `params_best.pkl`

Each checkpoint stores params, model config, train-only normalization, epoch,
best/final record, and run metadata. Optimizer state is intentionally not
saved.

## Warm-Start Next Step

Warm-start is not implemented in this change. A later step should add
`--init-checkpoint`, load params only, rebuild the optimizer from scratch, and
record the checkpoint source in `run_config.json` and `loss_summary.json`.

## Upstream Alignment Items

The upstream RIGNO path has capabilities or design differences that remain to
audit locally:

- edge masking behavior;
- checkpoint/resume flow;
- decoder/regional path details;
- whether Heat3D graph repair changes should remain strictly opt-in.

This note is preparation for P3/P4 audit work only. It is not a model,
decoder, loss, or objective change.
