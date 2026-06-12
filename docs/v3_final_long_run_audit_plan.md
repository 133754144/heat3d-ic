# Heat3D v3 Final Long-Run Audit Plan

Purpose: consolidate completed v3 long-run diagnostics and prepare the same
read-only result path for S2/S3 once their WSL2 runs are synced. This is a
diagnostic tracking plan, not a benchmark claim.

## Current Matrix

| run | seed | schedule | epochs | current status | role |
| --- | ---: | --- | ---: | --- | --- |
| W1 seed1 warmup-flat | 1 | `upstream_onecycle`, `1e-4 -> 1e-3`, flat | 1200 | completed and diagnosed | tests early warmup repair for seed1 |
| L2 seed1 constant | 1 | `constant`, `lr=1e-3` | 1200 | completed and diagnosed | seed1 repaired constant-lr reference |
| S1 seed1 constant | 1 | `constant`, `lr=1e-3` | 1600 | completed and diagnosed | checks whether L2 path was still undertrained |
| B6 seed0 warmup-cosine | 0 | `warmup_cosine`, `lr=5e-4`, `min_lr=5e-5` | 400 | completed and diagnosed | strongest seed0 baseline |
| S2 seed0 constant | 0 | `constant`, `lr=1e-3` | 1200 | pending WSL2 result | isolates seed0 versus schedule |
| S3 seed0 warmup-cosine | 0 | `warmup_cosine`, `lr=1e-3`, `min_lr=1e-4` | 1200 | pending WSL2 result | tests seed0 L3-style schedule |

All rows are diagnostic results or pending diagnostics. None should be
described as publication-ready or as a formal benchmark.

## Unified Audit Fields

The long-run audit table should include:

- run name and output directory
- seed, `model_seed`, `batch_order_seed`, `graph_seed`
- schedule type, `lr`, `min_lr`, `warmup_epochs`, epoch count
- graph policy
- best epoch
- final and best `valid_iid`
- final and best `valid_stress`
- final and best DeltaT RMSE / MAE
- field centered correlation and top-k overlap
- bin0 signed bias and overprediction ratio
- weakest split groups
- weakest condition groups
- final/best ratio
- trusted/caveat note

`scripts/summarize_heat3d_v3_long_run_audit.py` reads existing
`loss_summary.json`, `run_config.json`, and final/best diagnostics JSON files
and writes ignored JSON/Markdown summaries. It does not import JAX, load
predictions, build graphs, or start training.

## S2/S3 Intake

When S2/S3 finish and their output is available:

1. Confirm `loss_summary.json`, `predictions.npz`, and `best_predictions.npz`
   exist.
2. Run the same final/best diagnostics used for S1/W1 if any diagnostics JSON
   is missing.
3. Run:

```bash
python3 scripts/summarize_heat3d_v3_long_run_audit.py
```

4. Compare S2 against B6 to decide whether seed0 strength is mainly seed path
   or schedule path.
5. Compare S3 against S2 and B6 to test whether seed0 benefits from the
   stronger warmup-cosine setup.

Do not write conclusions for S2/S3 before their completed diagnostics exist.

## Mechanism Diagnostics Design

The next mechanism audit should be checkpoint-level first, replay-level only if
needed. Preferred read-only metrics:

- milestone loss curve at e20/e50/e100/e200/e400/e800/e1200/e1600
- `valid_iid` and `valid_stress` trajectory
- best epoch movement
- prediction amplitude ratio
- prediction std / target std, if available from diagnostics
- centered field correlation
- top-k overlap
- bin-level bias and overprediction ratio
- `high_dynamic_range_power_cases` error
- `multi_block_power` error
- `diag3`, low-k barrier / TIM, and extreme `top_h` condition errors

If short replay is later justified, add transient instrumentation for:

- encoder / processor / decoder grad norm
- latent RMS and activation norm
- update-to-param ratio
- decoder input-output sensitivity

Those replay hooks should be scoped as audit hooks, not model changes.

## P3 Audit Questions

The current evidence points away from blind schedule search and toward model
path explanation. P3 should answer:

- whether q/k/BC channels are effectively used by the encoder
- whether regional message passing propagates heat-source and boundary
  information enough for local hotspots
- whether the decoder is biased toward smooth fields instead of local shape
- whether `diag3` and high-contrast k expose k encoding or edge-message
  weakness
- whether extreme `top_h` cases expose weak BC scale conditioning
- whether multi-block and high-dynamic-range power cases need a stronger local
  power path

Forbidden at this stage: decoder changes, pointwise skip, loss/objective
changes, and new long training.
