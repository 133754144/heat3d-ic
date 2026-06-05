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

## Local Check

Passed with `sample_count=1`, `policy=legacy`, `epochs=2`.

| policy | best relative RMSE | best loss | p2r/r2p zero | edge ratio p2r/r2p | finite |
| --- | ---: | ---: | ---: | ---: | --- |
| legacy | 64.74% | 9.820259e-01 | 58 / 58 | 1.000 / 1.000 | true |

This is only a compatibility check. It confirms the B96-style AdamW path,
coverage reporting, edge-ratio reporting, metrics, and ignored JSON output work
locally.

## Devbox Results

Pending.

## Initial Interpretation

Pending. The decision point is whether nearest repair or discrete radius
improves best relative RMSE after optimizer/capacity confounding is removed.
