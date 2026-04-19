# Heat3D-IC: Steady Thermal Field Prediction with Graph Neural Operators

Heat3D-IC is a research codebase for steady 3D thermal field prediction from
heterogeneous thermal conductivity fields and localized heat source fields using
graph neural operators.

The current public version provides a minimal runnable workflow for a simplified
3D steady heat-conduction setting. It is designed as a foundation for continued
research, not as a complete industrial 3D IC, chiplet, TSV, or package thermal
simulation platform.

## Project Overview

The current workflow learns the operator

```text
[thermal conductivity field k(x), localized heat source field q(x)] -> steady temperature field T(x)
```

on a fixed 3D point cloud. The main project work is the task adaptation around a
graph neural operator core:

1. synthetic 3D heat data generation,
2. local Heat3D data loading,
3. 3D graph construction for the point cloud,
4. coefficient-field to temperature-field training,
5. checkpointed evaluation with reproducible splits and metrics.

For a longer technical summary, see [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md).
For upstream inheritance and citation requirements, see
[`ATTRIBUTION.md`](ATTRIBUTION.md).

## What Is Kept From The Upstream Core

This repository retains a minimal upstream graph-operator core needed by the
Heat3D workflow:

- region interaction graph operator and graph builder logic,
- graph network components,
- typed graph data structures,
- operator input structures,
- shared model utilities.

The generic upstream benchmark data interface, training CLI, testing CLI,
plotting utilities, and broad PDE experiment workflow are not part of the
current public entry path.

## What Is Newly Added For Heat3D-IC

Heat3D-specific components:

- Heat3D dataset adapter: loads Heat3D samples and exposes model-compatible
  temperature, coordinate, coefficient, and graph metadata fields.
- Heat3D graph builder wrapper: builds non-periodic 3D graph metadata for the
  Heat3D point cloud.
- Heat3D training/evaluation pipeline: provides deterministic splits,
  Heat3D-specific normalization, steady-output prediction, checkpoint IO, and
  evaluation metrics.
- `scripts/inspect_heat3d_dataset.py`: checks dataset loading and single-sample
  graph construction.
- `scripts/check_heat3d_batch_graphs.py`: checks batched graph metadata and
  graph construction.
- `scripts/train_heat3d_operator.py`: trains and saves a Heat3D graph neural
  operator checkpoint.
- `scripts/evaluate_heat3d_operator.py`: loads a checkpoint and evaluates the
  stored test split.

## Current Scope And Limitations

Current scope:

- steady-state 3D heat field prediction,
- synthetic fixed-grid 3D samples,
- heterogeneous thermal conductivity fields and localized heat source fields,
- supervised coefficient-to-temperature operator learning,
- minimal reproducible train/evaluate workflow.

Current limitations:

- The first public dataset is a prototype feasibility dataset, not a complete
  real 3D IC thermal dataset.
- TSVs, micro-bumps, BEOL interconnects, package layers, anisotropic materials,
  and detailed boundary heat transfer are not explicitly modeled.
- The current model is not physics-informed: it does not include PDE residual,
  interface heat-flux continuity, or boundary-condition losses.
- The current task is steady-state only, with no heat capacity term, time
  sequence, or autoregressive rollout.
- No deployment-level or industrial-accuracy claim is made.

## Installation

Create a fresh environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` currently requests `jax[cuda12]`. For CPU or a different
accelerator stack, adjust the JAX package according to the official JAX
installation instructions for your platform.

## Dataset

`dataset_3d_heat/` is intentionally not tracked by Git. It should be downloaded
or generated separately and placed at the repository root:

```text
dataset_3d_heat/
  sample_000/
    coords.npy
    temperature.npy
    k.npy
    source.npy
    edge_index.npy
  sample_001/
    ...
```

The public dataset entry point is:

- Hugging Face repo: `133754144X/heat3d-thermal-simulation`
- Dataset title: `Heat3D Thermal Simulation Dataset: Synthetic 3D Heat-Conduction Data for Operator Learning`
- Current public subset: `subsets/v0_unitcube_demo/`

The current public subset is a prototype synthetic feasibility dataset. It uses
a fixed UnitCube grid with heterogeneous thermal conductivity fields and
localized heat source fields, and stores the resulting 3D steady temperature
field. It is intended to validate the algorithm workflow, not to represent the
final full 3D IC task setting.

To run the current scripts, download the files under
`subsets/v0_unitcube_demo/samples/` and place the `sample_xxx/` directories under
`./dataset_3d_heat/`.

The current loader uses:

- `coords.npy`: 3D node coordinates,
- `temperature.npy`: target steady temperature field,
- `k.npy`: thermal conductivity field,
- `source.npy`: heat source field.

`edge_index.npy` may be present from data generation, but the current graph
operator path builds its own regional graph metadata from the 3D coordinates.

## Minimal Run Commands

Run from the repository root:

```bash
python3 scripts/inspect_heat3d_dataset.py
python3 scripts/check_heat3d_batch_graphs.py
```

Minimal train/eval smoke loop that writes only to `/tmp`:

```bash
python3 scripts/train_heat3d_operator.py \
  --epochs 1 \
  --batch-size 1 \
  --n-train 1 \
  --n-valid 1 \
  --n-test 1 \
  --output-dir /tmp/heat3d_smoke \
  --checkpoint-name smoke.pkl

python3 scripts/evaluate_heat3d_operator.py \
  --checkpoint /tmp/heat3d_smoke/smoke.pkl \
  --output-dir /tmp/heat3d_smoke_eval \
  --batch-size 1
```

These commands verify the execution path only. A one-epoch, one-sample run is
not a model-quality experiment.

## Training

Default training reads `./dataset_3d_heat/` and writes to
`./output/heat3d_ic/`:

```bash
python3 scripts/train_heat3d_operator.py
```

Common options:

```bash
python3 scripts/train_heat3d_operator.py \
  --data-dir dataset_3d_heat \
  --output-dir output/heat3d_ic \
  --checkpoint-name heat3d_operator_best.pkl \
  --epochs 30 \
  --batch-size 4 \
  --n-train 160 \
  --n-valid 20 \
  --n-test 20
```

The checkpoint stores model parameters, normalization statistics, split indices,
model configuration, graph builder configuration, and training arguments.

## Evaluation

Evaluate a checkpoint:

```bash
python3 scripts/evaluate_heat3d_operator.py \
  --checkpoint output/heat3d_ic/heat3d_operator_best.pkl \
  --output-dir output/heat3d_ic_eval \
  --batch-size 4
```

A small reference checkpoint is retained at
`output/heat3d_ic/heat3d_operator_best.pkl`. It is provided to exercise the
load/evaluate path. Evaluation still requires the matching local
`dataset_3d_heat/` directory.

The evaluator reports aggregate regression metrics and median relative L1 error
for operator-learning comparison.

## Citation / Attribution

This project is derived from a minimal subset of RIGNO:

- Upstream repository: <https://github.com/camlab-ethz/rigno>
- Paper: <https://arxiv.org/abs/2501.19205>

Please cite RIGNO when using the retained upstream model core:

```bibtex
@inproceedings{mousavi2025rigno,
  title         = {RIGNO: A Graph-based framework for robust and accurate operator learning for PDEs on arbitrary domains},
  author        = {Sepehr Mousavi and Shizheng Wen and Levi Lingsch and Maximilian Herde and Bogdan Raonic and Siddhartha Mishra},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {38},
  year          = {2025}
}
```
