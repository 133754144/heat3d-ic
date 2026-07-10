# Heat3D-IC Project Overview

Heat3D-IC adapts a graph neural operator workflow to steady thermal fields in
synthetic multilayer 3D IC-like structures. V4 closes with a frozen clean-IID
baseline, an explicit hard-challenge protocol, and a documented negative
Fourier ablation. The project remains a research codebase, not a thermal
signoff or deployment system.

For the public entry points and compact baseline table, see
[README.md](README.md). For exact V4P5 best/final metrics, see
[docs/v4_closeout.md](docs/v4_closeout.md).

## Phase Evolution

| phase | contribution | current relevance |
| --- | --- | --- |
| V0 | established the inherited Heat3D execution path and legacy input bridge | historical compatibility evidence |
| V1 | audited training semantics, normalization, target recovery, and feature provenance | foundation for controlled V4 runs |
| V2 | formalized metric, graph, batch, and split diagnostics | retained audit methods |
| V3 | expanded controlled training comparisons and final-probe analysis | historical baseline context |
| V4 | introduced registry-driven controls, semantic normalization, formal candidate/P5 splits, split-aware diagnostics, and P5 clean/hard evaluation | current closed stage |

## Current V4 Task

```text
[k(x), q(x), boundary conditions, geometry/extent features]
    -> steady DeltaT(x) / T(x)
```

The standard model statement remains:

```text
coords + k(x) + q(x) + BC -> T(x)
```

`k(x)`, `q(x)`, and boundary conditions are active model inputs. Geometry and
extent information is represented through the active coordinate and condition
policies. Layer-stack, interface, layer-ID, region-ID, and material-ID metadata
support dataset generation and evaluation grouping; they are not standard
default model-input features.

## P5 Dataset And Split

V4 closes on the ignored local dataset
`data/heat3d_v4_p5_clean_nohard_v0` and its tracked split map:

`configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json`.

- clean training/validation/test: 672 / 128 / 128;
- hard train holdout/challenge validation/challenge test: 121 / 12 / 12;
- clean roles contain no `physical_hard_keep` samples;
- all-IID is a reporting union, never a replacement training split.

The P5 dataset preserves original hard samples as a challenge rather than
deleting or relabeling them. It adds 49 solver-accepted clean replacements to
complete the clean split sizes.

## RIGNO Adaptation And Training Semantics

The project keeps the upstream RIGNO graph-operator core while adding:

- native Heat3D sample loading and the legacy zero-DeltaT bridge where required;
- non-periodic graph construction with discrete physical coverage and repair;
- semantic normalization for k/q/BC condition semantics;
- sample-local isotropic coordinates and log-extent broadcast features;
- a post-decoder residual using normalized condition features;
- run registry provenance, inherited YAML, result audit CSV, and checkpoint
  metadata;
- checkpoint selection by normalized `valid_base_mse` and sample-first
  split-aware reporting.

Raw DeltaT and recovered-temperature metrics are physical-scale reports. They
do not replace the normalized selection metric. The metrics contract requires
sample-first aggregation before split/group summaries; point-global flattened
values are retained only as separately labeled cross-checks.

## V4 Final Baseline

The frozen clean baseline is
`V4P5_02_clean_baseline_raw_B28_e600`:

- raw coordinates, plain MSE, B28, seed 0, 600 epochs;
- selected best checkpoint at epoch 405;
- P5 clean-nohard training and formal split;
- best clean valid/test sample-first RMSE: 0.119 / 0.153 K;
- best clean valid/test point-global raw DeltaT RMSE: 0.170 / 0.236 K.

The final epoch-600 checkpoint is retained for trajectory comparison, but does
not replace the selected best checkpoint. The V4P5_03 `raw_plus_fourier`,
frequency-4 run completed without OOM but is a negative ablation: it worsens
clean-IID and P02/P06 behavior despite a local P09 improvement.

## Hard Challenge

Hard challenge samples remain difficult. For the selected P5_02 checkpoint,
sample-first RMSE is 4.709 K on hard validation and 3.683 K on hard test. The
hard cohort contributes nearly all all-IID point MSE and shows a severe
high-amplitude/top5/strong-q scale failure. This is why clean-IID, hard
challenge, all-IID, and fixed final-probe results are reported separately.

The hard tail should be investigated as a controlled curriculum, fine-tune, or
feature-redesign line after the clean baseline is held fixed. It should not be
silently mixed into clean-IID training or used to reinterpret clean metrics.

## V5 Planned Direction

V5 is planned, not implemented by the V4 closeout:

1. q/target decomposition and a global physics-scale branch;
2. shape-scale decomposition for amplitude failures;
3. a bottom-Dirichlet hard constraint;
4. discrete physical residual metrics;
5. controlled hard-tail curriculum or fine-tune studies; and
6. multi-seed evaluation.

Upstream attribution and license obligations remain unchanged in
[ATTRIBUTION.md](ATTRIBUTION.md).
