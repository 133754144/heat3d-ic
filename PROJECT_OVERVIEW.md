# Heat3D-IC Project Overview

Heat3D-IC is a research codebase for steady 3D thermal field prediction with
graph neural operators. The current public version focuses on a simplified
synthetic setting and provides the minimum runnable workflow needed to support
future extensions.

## Problem Setting

The current supervised learning task is:

```text
[k(x), q(x)] -> T(x)
```

where:

- `k(x)` is the thermal conductivity field,
- `q(x)` is the localized heat source field,
- `T(x)` is the steady-state temperature field,
- `x` is a 3D coordinate on a fixed UnitCube point cloud.

The first public dataset is prototype synthetic data. It is suitable for
checking feasibility and execution flow, but it is not intended to represent a
complete final 3D IC/chiplet/package thermal dataset.

## Current Workflow

The public workflow is:

1. Generate or download `dataset_3d_heat/`.
2. Inspect local Heat3D sample loading:
   `scripts/inspect_heat3d_dataset.py`.
3. Verify batched graph construction:
   `scripts/check_heat3d_batch_graphs.py`.
4. Train a steady graph neural operator:
   `scripts/train_heat3d_operator.py`.
5. Reload the checkpoint and evaluate the stored split:
   `scripts/evaluate_heat3d_operator.py`.

The supporting implementation includes a Heat3D dataset adapter, a 3D graph
builder wrapper, and a Heat3D training/evaluation pipeline. Upstream inheritance
and license obligations are documented separately in `ATTRIBUTION.md`.

## Main Technical Contributions In This Repository

The current contribution is the Heat3D task adaptation and runnable workflow:

- a data representation for 3D heat samples using coordinates, conductivity,
  heat source, and temperature arrays,
- a 3D graph-construction wrapper for non-periodic point-cloud heat data,
- graph metadata reuse for fixed-coordinate samples,
- coefficient-field normalization for inputs and temperature-field
  normalization for outputs,
- a steady-output training/evaluation path instead of time marching,
- checkpoint payloads that store parameters, normalization statistics, split
  indices, model configuration, graph builder configuration, and training
  arguments,
- evaluation metrics including MSE, RMSE, MAE, relative L1/L2, NRMSE, R2, and
  median relative L1.

## Current Limitations

This repository does not claim:

- full industrial 3D IC/chiplet/package thermal simulation,
- explicit TSV, micro-bump, interposer, BEOL, or package-layer modeling,
- transient thermal simulation,
- physics-informed training,
- deployment-level reliability,
- superiority over other thermal simulation or neural-operator methods without
  controlled experiments.

## Future Work

Planned research directions include:

- publishing and versioning the synthetic UnitCube dataset,
- adding reproducible experiment configs,
- adding baseline comparisons and runtime measurements,
- using FEM adjacency or material-interface edges as additional graph structure,
- adding PDE residual and interface-continuity losses,
- extending the synthetic data generator to more complex geometries, boundary
  conditions, package scenarios, TSVs, micro-bumps, and chiplet stacks.
