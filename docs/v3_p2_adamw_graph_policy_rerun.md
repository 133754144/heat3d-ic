# Heat3D v3 P2-redux AdamW Graph Policy Rerun

## Purpose

P3-c showed that prior P2/P3 small-sample RIGNO fitting was confounded by
manual GD and low learning rate. This rerun re-evaluates graph policy impact
with B96-style AdamW defaults and larger RIGNO capacity, without changing model,
decoder, loss, objective, graph semantics, or running the full dataset.

## Default Changes

The controlled runner now defaults to AdamW instead of manual GD:

- optimizer: `adamw`
- lr: `3e-4`
- lr_schedule: `warmup_cosine`
- warmup_epochs: `10`
- min_lr: `1e-6`
- weight_decay: `1e-4`
- gradient_clip_norm: `1.0`

`manual_gd` remains available as an explicit CLI choice for debug and legacy
reproduction. Explicit YAML/CLI optimizer settings still take precedence because
the config command builder passes optimizer fields as CLI arguments.

The v3 P2/P3 small training scripts now also default to AdamW. P2-redux uses the
B96-style model config:

- node_latent_size: `128`
- edge_latent_size: `128`
- processor_steps: `6`
- mlp_hidden_layers: `2`
- concatenate_t / concatenate_tau / conditioned_normalization: `false`
- p_edge_masking: `0.0`

## Commands

Local compatibility check:

```bash
python3 scripts/run_heat3d_v3_p2_adamw_graph_policy_rerun.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --sample-count 1 \
  --policy legacy \
  --epochs 2 \
  --output-json output/heat3d_v3_p2_adamw_rerun/local_e2.json
```

Devbox reruns:

```bash
python3 scripts/run_heat3d_v3_p2_adamw_graph_policy_rerun.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --sample-count 1 \
  --policy all \
  --epochs 300 \
  --output-json output/heat3d_v3_p2_adamw_rerun/sample1_b96_adamw_e300.json

python3 scripts/run_heat3d_v3_p2_adamw_graph_policy_rerun.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --sample-count 4 \
  --policy all \
  --epochs 300 \
  --output-json output/heat3d_v3_p2_adamw_rerun/sample4_b96_adamw_e300.json

python3 scripts/run_heat3d_v3_p2_adamw_graph_policy_rerun.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --sample-count 16 \
  --policy all \
  --epochs 300 \
  --output-json output/heat3d_v3_p2_adamw_rerun/sample16_b96_adamw_e300.json
```

Optional optimizer contrast:

```bash
python3 scripts/run_heat3d_v3_p2_adamw_graph_policy_rerun.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --sample-count 1 \
  --policy all \
  --optimizer adam \
  --lr 1e-3 \
  --epochs 1000 \
  --output-json output/heat3d_v3_p2_adamw_rerun/sample1_adam_lr1e3_e1000.json
```

## Local Check

Passed with `sample_count=1`, `policy=legacy`, `epochs=2`.

| policy | best relative RMSE | best loss | p2r/r2p zero | edge ratio p2r/r2p | finite |
| --- | ---: | ---: | ---: | ---: | --- |
| legacy | 64.74% | 9.820259e-01 | 58 / 58 | 1.000 / 1.000 | true |

This is only a compatibility check. It confirms the B96-style AdamW path,
coverage reporting, edge-ratio reporting, metrics, and ignored JSON output work
locally.

## Devbox Results

All runs used supervised-small train-only samples and wrote JSON only under
ignored `output/heat3d_v3_p2_adamw_rerun/`.

### B96 AdamW, 300 epochs

| sample_count | policy | best relative RMSE | final relative RMSE | best loss | best epoch | p2r/r2p zero | edge ratio p2r/r2p | finite |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | legacy | 25.89% | 25.89% | 1.570592e-01 | 300 | 58 / 58 | 1.000 / 1.000 | true |
| 1 | nearest_repair | 25.83% | 25.83% | 1.563841e-01 | 300 | 0 / 0 | 1.468 / 1.475 | true |
| 1 | discrete_radius | 57.01% | 57.01% | 7.614899e-01 | 300 | 0 / 0 | 2.621 / 2.664 | true |
| 4 | legacy | 56.32% | 56.32% | 6.321909e-01 | 300 | 222 / 222 | 1.000 / 1.000 | true |
| 4 | nearest_repair | 26.47% | 26.47% | 1.404833e-01 | 300 | 0 / 0 | 1.500 / 1.507 | true |
| 4 | discrete_radius | 59.35% | 59.35% | 7.064050e-01 | 300 | 0 / 0 | 2.768 / 2.806 | true |
| 16 | legacy | 62.01% | 62.01% | 7.714413e-01 | 297 | 908 / 908 | 1.000 / 1.000 | true |
| 16 | nearest_repair | 61.20% | 61.20% | 7.586823e-01 | 300 | 0 / 0 | 1.506 / 1.519 | true |
| 16 | discrete_radius | 61.33% | 61.33% | 7.619693e-01 | 300 | 0 / 0 | 2.753 / 2.826 | true |

### Adam lr=1e-3, 1000 epochs, sample_count=1

| policy | best relative RMSE | final relative RMSE | best loss | best epoch | p2r/r2p zero | edge ratio p2r/r2p | finite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 54.19% | 54.19% | 6.880412e-01 | 1000 | 58 / 58 | 1.000 / 1.000 | true |
| nearest_repair | 9.84% | 9.84% | 2.268889e-02 | 1000 | 0 / 0 | 1.468 / 1.475 | true |
| discrete_radius | 54.14% | 54.14% | 6.867742e-01 | 1000 | 0 / 0 | 2.621 / 2.664 | true |

## Initial Interpretation

Changing the default optimizer away from manual GD is necessary. B96 AdamW
reduces 1-sample legacy error from the old low-lr regime to about 25.9%, but it
still does not reach the v3 <=20% small-fitting gate.

Nearest repair is the strongest graph-policy candidate. It eliminates p2r/r2p
zero coverage with about 1.47x-1.52x p2r/r2p edge cost. Under B96 AdamW it gives
only a tiny 1-sample gain, a large 4-sample gain, and a tiny 16-sample gain.
Under the Adam lr=1e-3 1-sample contrast, nearest repair reaches 9.84% while
legacy remains at 54.19%, so zero coverage can be a real bottleneck in at least
some optimizer settings.

Discrete coverage radius is not recommended as the next default candidate. It
also removes zero coverage, but costs about 2.6x-2.8x p2r/r2p edges and does
not improve fitting in these runs. The current discrete policy likely changes
the support distribution too aggressively, not just the uncovered-node cases.

The combined evidence does not support "zero coverage is the only bottleneck."
It supports a narrower conclusion: explicit nearest repair is useful and should
remain the P2/P3 graph candidate, but 16-sample fitting still points to model
path, routing, capacity, optimizer schedule, or decoder bottlenecks.

## Follow-Up: Adam lr=1e-3, 1000 epochs

Purpose: rerun only `legacy` and `nearest_repair` for `sample_count=1/4/16`
with B96-style model capacity and Adam lr=1e-3 constant. `discrete_radius` is
paused as a mainline candidate.

Commands:

```bash
python3 scripts/run_heat3d_v3_p2_adamw_graph_policy_rerun.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --sample-count 1 \
  --policy legacy,nearest_repair \
  --optimizer adam \
  --lr 1e-3 \
  --lr-schedule constant \
  --epochs 1000 \
  --output-json output/heat3d_v3_p2_adamw_rerun/followup_sample1_adam_lr1e3_e1000.json

python3 scripts/run_heat3d_v3_p2_adamw_graph_policy_rerun.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --sample-count 4 \
  --policy legacy,nearest_repair \
  --optimizer adam \
  --lr 1e-3 \
  --lr-schedule constant \
  --epochs 1000 \
  --output-json output/heat3d_v3_p2_adamw_rerun/followup_sample4_adam_lr1e3_e1000.json

python3 scripts/run_heat3d_v3_p2_adamw_graph_policy_rerun.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --sample-count 16 \
  --policy legacy,nearest_repair \
  --optimizer adam \
  --lr 1e-3 \
  --lr-schedule constant \
  --epochs 1000 \
  --output-json output/heat3d_v3_p2_adamw_rerun/followup_sample16_adam_lr1e3_e1000.json
```

### Adam lr=1e-3 constant, 1000 epochs

| sample_count | policy | best relative RMSE | final relative RMSE | best loss | best epoch | p2r/r2p zero | edge ratio p2r/r2p | finite |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | legacy | 37.04% | 37.04% | 3.215249e-01 | 1000 | 58 / 58 | 1.000 / 1.000 | true |
| 1 | nearest_repair | 37.84% | 37.40% | 3.267605e-01 | 986 | 0 / 0 | 1.468 / 1.475 | true |
| 4 | legacy | 50.49% | 55.88% | 5.060120e-01 | 912 | 222 / 222 | 1.000 / 1.000 | true |
| 4 | nearest_repair | 58.48% | 56.55% | 6.228310e-01 | 684 | 0 / 0 | 1.500 / 1.507 | true |
| 16 | legacy | 54.78% | 54.92% | 6.034866e-01 | 998 | 908 / 908 | 1.000 / 1.000 | true |
| 16 | nearest_repair | 55.30% | 55.39% | 6.162985e-01 | 998 | 0 / 0 | 1.506 / 1.519 | true |

Adam lr=1e-3 constant does not support nearest_repair as stable improvement.
It removes zero coverage, but the best relative RMSE is slightly worse than
legacy for all three sample counts in this run.

### Optional B96 AdamW warmup-cosine, 1000 epochs

| sample_count | policy | best relative RMSE | final relative RMSE | best loss | best epoch | p2r/r2p zero | edge ratio p2r/r2p | finite |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | legacy | 19.68% | 19.68% | 9.077220e-02 | 1000 | 58 / 58 | 1.000 / 1.000 | true |
| 1 | nearest_repair | 19.62% | 19.62% | 9.017006e-02 | 1000 | 0 / 0 | 1.468 / 1.475 | true |
| 4 | legacy | 55.39% | 55.39% | 6.117184e-01 | 995 | 222 / 222 | 1.000 / 1.000 | true |
| 4 | nearest_repair | 17.76% | 17.76% | 6.383359e-02 | 1000 | 0 / 0 | 1.500 / 1.507 | true |
| 16 | legacy | 60.81% | 60.81% | 7.416919e-01 | 999 | 908 / 908 | 1.000 / 1.000 | true |
| 16 | nearest_repair | 59.71% | 59.71% | 7.225689e-01 | 1000 | 0 / 0 | 1.506 / 1.519 | true |

B96 AdamW e1000 gives a more favorable but still mixed result: nearest_repair is
slightly better for 1-sample, strongly better for 4-sample, and only slightly
better for 16-sample. This supports nearest_repair as the B96 A/B candidate, but
not as a complete fix for the 16-sample fitting bottleneck.
