# Heat3D v2 medium1024 split protocol review

Scope: split-protocol review only. No new labels, no training, no model/loss changes.

## Current split source

`medium1024_gapA_full1024_v2` stores split membership in each sample's metadata:

- `sample_meta.json: split`
- `metadata.json: split`

The v2 training runner currently reads these fields through `_subset_split_ids(...)`, then uses only `train` and `valid` for controlled training. The source generation plan is defined in:

- `configs/heat3d_v1_physics_label_medium1024_gapA_manifest.json`
- `tools/generate_heat3d_v1_physics_label_medium.py`

The manifest also defines candidate OOD splits. It explicitly marks candidate OOD splits as diagnostics-only, but the plain `valid` split was not separated into IID validation and stress validation.

## Current valid behavior

The existing `valid` split behaves like a stress/OOD diagnostic rather than an IID learning-validation split:

| category | old train | old valid |
|---|---:|---:|
| low_power | 1 | 113 |
| diag3 | 72 | 127 |
| high_top_h | 116 | 127 |
| low_k_barrier_or_TIM_variation | 2 | 67 |
| raw DeltaT mean | 0.02929 | 0.01096 |
| low DeltaT fraction <=0.01 K | 0.390 | 0.809 |

This explains why B192 training can reduce train loss while `valid_loss` rises: validation is dominated by low-power, low-DeltaT, high-top-h, diag3 and barrier/TIM-variation cases that are nearly absent from train.

## Suitability

The current `valid` split is not suitable as the only model-learning validation signal. It is useful as a stress diagnostic, but it should not be the only selection target for research-stage controlled training.

## Dataset decision

A new dataset is not required for the immediate fix. The full 1024 samples already contain the relevant categories:

- low-power cases exist in the full set;
- diag3 cases exist in the full set;
- high-top-h cases exist in the full set;
- high-contrast and barrier/TIM k-region cases exist in the full set.

The problem is the split protocol, not missing labels.

## Minimal fix

Add an external split map without moving sample directories or editing labels:

- keep all 1024 sample IDs;
- keep held-out BC/stack candidates as explicit OOD test splits;
- create `valid_iid` from the regular train-like distribution;
- create `valid_stress` for low-power/high-top-h/diag3/barrier-TIM stress cases;
- ensure train contains enough low-power, diag3 and barrier/TIM samples.

This preserves the existing dataset while separating IID validation from stress diagnostics.
