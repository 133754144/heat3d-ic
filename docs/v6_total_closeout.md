# V6 total closeout

Status: **preregistered_pending_hard_stress**.

This is the authoritative V6 summary. Earlier phase reports remain evidence records and are not rewritten.

## Canonical dataset and model

- Dataset: `heat3d_v6_p1h_shared_support1024_v0`, 1024 cases, 128 geometry groups, group-locked 768/128/128 splits.
- Model: `V6_03_V5best_P1h`; seed0 e111 point-global checkpoint is the reference. Seeds 1/2 are replications.
- `V6_04_V5best_P1h_DualAttention` remains a registered ablation.
- Applicability is limited to the P1h source-aware support family.

## Frozen production workflow

1024 source-aware conditioning anchors provide Global Context and scale. Added 4096/8192/16384 source-aware query nodes use the frozen Anchor-derived workflow, sparse KD-tree edge-list graph cache, and layer/interface-aware reconstruction to 240825 solver nodes.

- 4096: default/hotspot-oriented.
- 8192: balanced full-field.
- 16384: maximum full-field accuracy.
- 32768: experimental and excluded from formal holdout/hard tables.

## Three-seed valid_iid full-field performance

| Mode | Full RMSE mean±std K | Point-global mean±std % |
|---:|---:|---:|
| 4096 | 1.4966±0.0593 | 3.6908±0.1462 |
| 8192 | 1.3001±0.0890 | 3.2062±0.2196 |
| 16384 | 1.2236±0.0934 | 3.0174±0.2304 |

## Governance

- The previously opened test split is a `corrected confirmatory holdout`, not a pristine test set.
- The wrong-ladder temporary outputs are excluded by SHA and were never used for selection.
- FVM results are `legal structured-FVM mesh sensitivity`, not a matched-accuracy claim.
- CPU→CPU and GPU→CPU speedups use the same 240825-node CPU FVM cold/warm denominator and remain nonmatched-DOF.
- Canonical P1h contains no true OOD role. The preregistered hard corner is an input-defined in-distribution stress subset.

## Remaining limits

- Only the P1h source-aware support family is covered.
- Added query nodes still participate in the frozen joint encoder/processor path.
- The corrected confirmatory holdout is not a fresh untouched test set.
- Canonical P1h has no registered distribution-shift OOD role.
- FVM/model comparisons use nonmatched degrees of freedom and hardware.
- No uncertainty calibration or real-package experimental validation is claimed.
