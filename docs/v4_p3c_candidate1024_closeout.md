# V4 P3c Candidate1024 Closeout

## Published Dataset

- Dataset ID: `heat3d_v4_p3c_candidate1024_v0`
- Local data root: `/Users/xuyihua/.codex/worktrees/cf01/3D IC Heat/data/heat3d_v4_p3c_candidate1024_v0/`
- Local audit root: `/Users/xuyihua/.codex/worktrees/cf01/3D IC Heat/output/heat3d_v4_p3c_candidate1024_v0/`
- devbox data root: `~/myCodeGitOnly/heat3d-ic/data/heat3d_v4_p3c_candidate1024_v0/`
- wsl2 data root: `~/myCodeGitOnly/heat3d-ic/data/heat3d_v4_p3c_candidate1024_v0/`
- Hugging Face path: `https://huggingface.co/datasets/133754144X/heat3d-thermal-simulation/tree/main/subsets/heat3d_v4_p3c_candidate1024_v0`

Hash checks matched across local, devbox, and wsl2:

- `manifest.json`: `736336bab9f055232bf7b40643bc5e144e3b60b7e546822b30816a18a2fb2515`
- `audit_summary.json`: `2d88977044e701bd1969bfa1e570ac3938ef96da0bf59734cc775d7ed6a4b175`
- `sha256_manifest.json`: `4159d277a9c2810e4b0cac870f3e7b9d389bec60c22880143e1f2ee49036ed0d`

`sha256_manifest.json` covers 9218 dataset files. The Hugging Face subset path
contains the same dataset plus the sha manifest file.

## Dataset Summary

- Accepted samples: 1024
- Rejected candidates: 23
- Candidate count consumed: 1047
- Split: train 768, test 256
- P3b-lite reference subset: 64 samples
- Solver pass rate: 1.0
- Failure count: 0
- NaN/Inf audit: pass
- Max absolute energy-balance residual: `6.06159566984843e-12`
- Max bottom Dirichlet error: `0.0`
- Max q total power error: `1.7763568394002505e-15 W`
- Max q boundary power: `0.0 W`

QC classes:

- `clean_keep`: 766
- `physical_hard_keep`: 145
- `review_hold`: 113

DeltaT bins:

- `low`: 64
- `nominal`: 389
- `hard`: 313
- `review_high`: 113
- `reject_high`: 145
- `reject_low`: 0

Generation coverage:

- q families: compact hotspot 148, dual-z 145, elongated 148, multi-block 150,
  TSV-adjacent 144, weak-background hotspot 139, weak background 150
- cooling regimes: weak effective air 349, nominal package 337, strong forced
  or effective heatsink 338
- k modes: scalar 225, diag3 799
- diag3 policy: mild 637, HBM-like strong 162, scalar 225

Accepted high-DeltaT triage:

- not high DeltaT: 879
- physical low-k enclosed compact hotspot: 97
- physical multi-source or high-power bottleneck: 28
- physical weak-cooling high DeltaT: 20

Reject reason counts:

- `unclassified_high_deltaT`: 22
- `reject_low`: 1

Review reason counts:

- `review_high_deltaT_bin`: 113

## Consumer-Side Smoke

devbox smoke used the synchronized dataset at
`~/myCodeGitOnly/heat3d-ic/data/heat3d_v4_p3c_candidate1024_v0/`.

Preflight checks:

- manifest, audit, and sha manifest hashes matched the local values above.
- accepted samples were 1024.
- manifest split was train 768 / test 256.
- required sample files loaded: `coords.npy`, `layer_id.npy`, `region_id.npy`,
  `material_id.npy`, `k_field.npy`, `q_field.npy`, `bc_features.npy`,
  `sample_meta.json`, `temperature.npy`.
- `Heat3DV1NativeSupervisedDataset` loaded 1024 samples.
- sample shape: coords `(1024, 3)`, features `(1024, 11)`, target `(1024, 1)`.
- active features: `k_x`, `k_y`, `k_z`, `q`, `is_top`, `is_bottom`, `is_side`,
  `is_interior`, `top_h`, `top_T_inf`, `bottom_T_fixed`.

Existing V4 runner compatibility:

- The published dataset uses train/test splits.
- The current V4 runner expects `train` plus `valid_iid` when a split-map is
  provided.
- The smoke therefore used a temporary devbox split-map that maps dataset
  `test` to runner `valid_iid` without modifying the dataset.

Training smoke results:

- Command path: `scripts/run_heat3d_v4_controlled_training.py`
- Output root: `~/myCodeGitOnly/heat3d-ic/output/heat3d_v4_p3c_candidate1024_v0_training_smoke_b8/`
- Epochs: 1
- Batch sizes: train 8, validation 16, prediction 16
- Selection metric: `valid_base_mse`
- Feature mode: relative BC features, diag3 k encoding, zero-delta bridge
- Target mode: normalized DeltaT
- Split-map counts: train 768, valid_iid 256
- Final valid_iid loss: `1.1415960050653666`
- Final valid_iid base MSE: `1.1415960050653666`
- Final valid_iid raw DeltaT MSE: `7.42358826007694`
- Grad finite: true
- Predictions: `predictions.npz` and `best_predictions.npz` each contain 256
  finite prediction entries.
- Checkpoints were intentionally not written for this smoke.

Batch-size note:

- A first smoke attempt with train/valid/prediction batch size 44 failed during
  backprop with GPU out-of-memory.
- Batch size 8 for training and 16 for validation/prediction passed.

2026-07-03 local consumer refresh:

- Current branch: `research/v4`.
- Dataset root: `data/heat3d_v4_p3c_candidate1024_v0/`.
- Readable files: `manifest.json`, `audit_summary.json`,
  `sha256_manifest.json`.
- Hashes matched the published closeout values above.
- Accepted sample files checked: 1024 sample directories, with no missing
  required files.
- Manifest split: train 768, test 256.
- Runner split bridge:
  `configs/heat3d_v4/candidate1024_v0_test_as_valid_iid_split_map.json`
  maps the published test split to `valid_iid` as 768 train / 256 valid_iid.
- `Heat3DV1NativeSupervisedDataset` loaded 1024 samples with coords
  `(1024, 3)`, condition features `(1024, 11)`, and target `(1024, 1)`.
- Required arrays were finite and matched expected point/target shapes.
- Local 1-epoch smoke used
  `scripts/run_heat3d_v4_controlled_training.py` with V4P3 semantic wrapper
  settings, B8 train, B16 validation/prediction, no final-probe, no
  post-training diagnostics, no checkpoints, and prediction split `valid_iid`.
- Ignored smoke output:
  `output/heat3d_v4_p3c_candidate1024_consumer_smoke_20260703/run/`.
- Smoke result: `status_ok=true`, `grad_finite=true`, best epoch 1,
  final/best valid_base_mse `1.2061412334442139`, final/best raw DeltaT MSE
  `7.843317031860352`.
- `predictions.npz` and `best_predictions.npz` each contain 256 finite
  `(1024, 1)` prediction arrays.

Local refresh conclusion: candidate1024_v0 passes consumer-side manifest,
loader, batch-build, one-epoch train/eval, and prediction-export smoke on the
current `research/v4` code path. The metric is smoke-only and must not be used
as a model-quality claim.

## Known Limitations

- This is a 1-epoch consumer smoke, not a performance claim.
- The existing runner still uses `valid_iid` naming, so a tracked formal split
  bridge or runner update is needed for native train/test semantics.
- The runner builds `all` groups during startup even when prediction is limited
  to `valid_iid`; candidate1024 startup therefore has noticeable overhead.
- P3b-lite is currently a fixed reference subset selection, not a separate
  physics-validation pass in this closeout.
- `review_hold` samples are accepted for research coverage but must be reported
  separately from clean samples in training analysis.

## Training Handoff

Recommendation: proceed to formal training. Tracked registry/YAML entrances for
this dataset already exist:

- `V4P3_04`: boundary-distance replacement, train-minmax coordinates, B24.
- `V4P3_05`: legacy BC flags, sample-local isotropic coordinates, B24.
- `V4P3_06`: boundary-distance replacement plus sample-local isotropic
  coordinates, B24.

The standalone `configs/heat3d_v4/candidate1024_v0_e050_b32.yaml` remains a
short-horizon handoff reference, but long training should prefer the registry
entries above so CSV audit and result collection stay consistent.

Formal configs should keep:

- subset: `data/heat3d_v4_p3c_candidate1024_v0`
- split bridge:
  `configs/heat3d_v4/candidate1024_v0_test_as_valid_iid_split_map.json`
  until runner split naming is updated
- split semantics: train 768 and published test-as-`valid_iid` 256
- batch size: B24 registry entries are the current launch candidates; reduce to
  B8/B16 only for smoke or memory triage
- selection metric: `valid_base_mse`
- long-run diagnostics: enabled only after the first stable short run

Formal reporting must stratify metrics by:

- `qc_class`
- `DeltaT_bin`
- `q_family`
- `cooling_regime`
- `diag3_policy`
- P3b-lite reference subset membership

Closeout status: candidate1024_v0 is suitable for training handoff with the
split-bridge caveat above. It can enter long training through the tracked V4P3
registry entries after standard registry dry-run checks and explicit launch
approval. Do not treat any 1-epoch smoke metric as model quality evidence.
