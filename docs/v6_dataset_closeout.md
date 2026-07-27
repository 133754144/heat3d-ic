# V6 dataset closeout

## Frozen lifecycle

`heat3d_v6_p1h_shared_support1024_v0` is the sole default V6-layer canonical
dataset. It contains 1,024 samples in 128 group-locked groups, with
`train=768`, `valid_iid=128`, and sealed `test_iid=128`. Every sample has the
same ordered 1,024 solver-node coordinates and graph.

`heat3d_v6_p1g_geometry_deconfounded1024_v0` is retained as
`archived_geometry_adaptive_baseline`. Its data and manifest remain intact,
and the immutable V6_01/V6_02 configs retain their historical P1g bindings.
V6_03 is the canonical P1h model configuration; V6_04 is the DualAttention
ablation. No historical configuration or run artifact was overwritten.

The machine-readable default and lifecycle sources are:

- `configs/heat3d_v6/v6_layer_canonical_default.yaml`
- `configs/heat3d_v6/v6_training_dataset_lifecycle.csv`
- `configs/heat3d_v6/v6_run_artifact_freeze.json`

## Integrity

- manifest SHA256:
  `324ca50a85698223d36c12a05d3e26b5cbc9aa00b559d067619baeb37f11e9d5`
- `full_fields.h5` SHA256:
  `f58141b3f365c5c90a57ec3802ae57c7e7afbf83ba0ab988060a617164b14c00`
- shared 1,024-coordinate SHA256:
  `2bda8e710c8c9f15b180783dc12132280253124688d3e3177296e97527798745`
- shared training graph SHA256:
  `6d3d62830755872194766aad2a8ac7b0f1fabec57840dac78fcb2642a6ed771c`

## Common-domain valid probe

The fixed 4,096-node probe is selected from the public solver mesh using only
stack geometry, layer/interface locations, Robin surfaces, and the two
source-allowed layer domains. It does not inspect temperature, source values,
predictions, errors, or split labels. The frozen support covers all 9 layers,
all 8 interfaces, 256 top nodes, 256 bottom nodes, and 2,048 nodes in the two
source-allowed layers.

Only the 128 `valid_iid` cases are evaluated. The evaluator uses full-field
solver `q` and temperature values at the frozen indices and reports
CV-weighted point-global and sample-first errors, peak and source-region
errors, layer mean/drop errors, and top/bottom surface errors. `test_iid`,
hard roles, training, checkpoint selection, and parameter updates are outside
this closeout.

Machine-readable support and results:

- `configs/heat3d_v6/v6_valid_common_probe4096.json`
- `configs/heat3d_v6/v6_common_valid_probe4096_results.json`
- `docs/v6_common_valid_probe4096_results.md`

Point-global checkpoint results on the common probe:

| Model | Point-global CV | Sample-first CV | Raw CV RMSE | Role |
|---|---:|---:|---:|---|
| V6_02 | 216.636406% | 218.865342% | 89.415544 K | historical P1g adaptive-support baseline |
| V6_03 | 1.851389% | 1.855461% | 0.764151 K | canonical P1h model configuration |
| V6_04 | 1.834161% | 1.813109% | 0.757040 K | DualAttention ablation |

V6_03 and V6_04 both transfer cleanly from the shared 1,024-node training
support to the independent 4,096-node solver support. V6_04 is lower than
V6_03 by 0.017228 percentage points in point-global CV error and by 0.042352
percentage points in sample-first CV error, but this single valid-only
diagnostic does not change its preregistered ablation role. V6_02 fails on the
P1h common domain with a large positive temperature bias; this is evidence of
the P1g geometry-adaptive support/domain mismatch, not a rewrite of its
historical 1,024-node P1g result.

## Hugging Face mirror

The canonical dataset is mirrored without deleting or overwriting any other
subset at:

`133754144X/heat3d-thermal-simulation/subsets/heat3d_v6_p1h_shared_support1024_v0/`

The mirror contains all 1,024 sample directories, `manifest.json`,
`full_fields.h5`, and a subset README. Remote verification and the final Hub
commit are recorded in
`configs/heat3d_v6/v6_hf_sync_receipt.json`.

The verified Hub HEAD is
`58a1d78673ec7d10ee9bbb3f9717a3b7f06d2384`.
