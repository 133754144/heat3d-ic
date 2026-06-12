# Heat3D v3 P3 Model Path / Decoder Audit

Purpose: define the read-only P3 audit path for explaining why some RIGNO
settings remain seed-sensitive and weak on local or OOD thermal structure. This
document is not a model-change proposal and does not claim benchmark results.

## Current Model Path

The current Heat3D controlled runner builds RIGNO inputs from per-sample
bridges and then batches shape-compatible examples:

1. Dataset/sample bridge provides coordinates, raw condition features, target
   raw DeltaT, and reference temperature.
2. The runner concatenates raw `u`, raw condition channels `c`, raw coordinates,
   and target DeltaT across a batch.
3. Condition channels are normalized with train-only `condition_mean/std`.
4. Target is normalized as `normalized_deltaT`.
5. Coordinates are normalized before model input; graph metadata is built from
   the sample coordinates through `Heat3DGraphBuilder`.
6. RIGNO receives `Inputs(u, c, x_inp, x_out)` plus `p2r/r2r/r2p` graphs and
   outputs normalized DeltaT.

The model path in `rigno/models/rigno.py` is:

- concatenate `inputs.u` and `inputs.c` into physical-node features
- encoder: pmesh to rmesh through `p2r`
- processor: regional message passing on `r2r`
- decoder: rmesh to pmesh through `r2p`, with latent pnodes also passed into
  the decoder
- output: normalized DeltaT at physical nodes

## Input Feature Path

The v3 long-run configs use:

- `k_encoding_mode=diag3`, so anisotropic or region-varying conductivity is
  represented through condition channels rather than a separate physics solver
  path.
- Heat-source / power features in the same per-node condition tensor.
- BC flags/scalars through `relative_bc_features`.
- `zero_delta_u_bridge`, so the model predicts DeltaT from condition fields
  rather than from a nonzero prior temperature field.

Audit implication: q/k/BC compete inside the same encoder input tensor. If
feature scaling or encoder sensitivity is uneven, the model can fit smooth
global fields while underusing local power, anisotropic k, or extreme BC
signals.

## Graph And Message-Passing Path

The graph builder constructs:

- `pmesh`: physical nodes from normalized Heat3D coordinates
- `rmesh`: subsampled regional nodes
- `p2r`: physical-to-regional encoder graph
- `r2r`: regional processor graph
- `r2p`: regional-to-physical decoder graph

Current v3 repaired runs use:

- `radius_policy=legacy_kdtree_mean4`
- `coverage_repair_policy=nearest_rnode`
- `repair_p2r=true`
- `repair_r2p=true`
- `min_physical_coverage=1`

This guarantees at least nearest regional coverage for uncovered physical
nodes, but it does not guarantee strong multi-hop receptive field, balanced
edge degree, or adequate interface/local-hotspot message capacity. The B88
long-run configs use `processor_steps=6`; that may still limit information
transport across rnodes, especially for multi-block power, thin layers,
interfaces, or extreme BC changes.

Edge features are structural: relative coordinate vector and distance,
normalized by a graph-level max edge length, plus support-radius features on
sender/receiver node sets. There is no explicit physics-aware edge feature for
material interface jumps, BC category, or source intensity.

## Decoder Path

The decoder takes processed regional latents and latent physical nodes:

- `updated_latent_rnodes` from the regional processor
- `latent_pnodes` from the p2r encoder
- `r2p` graph structural features

There is no direct post-processor q/k/BC bypass into the output head. q/k/BC can
affect the output only through encoded physical latents, regional latents, and
the r2p decoder. This is a plausible bottleneck for:

- local hotspot shape recovery
- high-dynamic-range source patterns
- multi-block power fields
- sharp material-interface effects
- extreme top-convection boundary cases

If the decoder relies too much on smooth regional messages or encoded pnode
shortcuts, it may reproduce global field trends while suppressing local peak
shape and sharp amplitude variation.

## Mapping Current Weaknesses To Bottlenecks

Known diagnostic weaknesses from W1/L2/S1/B6:

- OOD stack and OOD combined splits remain harder than train/valid.
- OOD BC is consistently a high-RMSE split in seed1 runs.
- `high_dynamic_range_power_cases` and `multi_block_power` are common weak
  source categories.
- `diag3` cases are harder than `iso1`, indicating anisotropic k encoding or
  edge-message use may be weak.
- `low_k_barrier_or_TIM_variation` and `high_contrast_interface_k` remain
  difficult, consistent with interface transport limitations.
- `very_high_top_h`, `very_low_top_h`, and held-out top-h categories expose BC
  conditioning and scaling sensitivity.

Possible bottleneck mapping:

- OOD stack / combined: limited regional receptive field or missing structural
  stack features in messages.
- OOD BC / extreme top-h: BC scale conditioning may not be preserved through
  encoder/processor/decoder.
- high-dynamic-range power: local q information may be smoothed by p2r/r2r/r2p.
- multi-block power: processor and decoder may struggle with multiple separated
  sources.
- diag3 / high-contrast k: k channels may be underused, or edge features may
  lack material-interface awareness.
- low-k barrier / TIM: rmesh message passing may not represent thin interface
  resistance sharply enough.

## Prediction-Level Mechanism Diagnostics

Before any model change, the next diagnostics should decompose prediction
errors by amplitude, shape, and condition group:

- amplitude ratio
- prediction std / target std
- centered spatial correlation
- z-score RMSE, when available
- hotspot centroid distance
- peak error
- top-k overlap
- bin0 signed bias and overprediction ratio
- condition-wise amplitude/shape decomposition
- split/source/k/BC group errors for high-dynamic-range power, multi-block
  power, `diag3`, low-k barrier / TIM, and extreme top-h cases

These metrics can be computed from saved predictions and metadata; they should
not require a training replay.

## Checkpoint / Replay Diagnostics Design

If prediction-level diagnostics are insufficient, a short controlled replay can
instrument model internals without changing model semantics:

- encoder / processor / decoder grad norm
- latent RMS and activation norm
- processed-rnode update magnitude
- update-to-param ratio
- decoder input-output sensitivity
- q/k/BC ablation sensitivity before and after a short fit
- zero/shuffle processed-rnode and latent-pnode ablations

Replay should be scoped to a short audit. It should not become a new training
claim and should not introduce decoder changes, pointwise skip, or objective
changes.

## Current Boundaries

Allowed now:

- read code
- read existing output
- summarize diagnostics
- design audit hooks and metrics

Not allowed in this P3 preparation step:

- decoder changes
- pointwise skip
- model architecture changes
- loss/objective changes
- new long training
- claims that S1/W1/B6 are publication-ready benchmarks
