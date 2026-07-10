# V4 P5 Clean-Nohard Dataset And Baseline

Read this file only for P5 dataset, split, hard-challenge, or baseline handoff.

## Dataset

- ID/path: `heat3d_v4_p5_clean_nohard_v0` /
  `data/heat3d_v4_p5_clean_nohard_v0`.
- Source: `heat3d_v4_p3c_candidate1024_v0`; no source sample was deleted,
  relabeled, or modified.
- Original non-hard roles are preserved: train `647`, valid_iid `116`,
  test_iid `116`.
- New solver-labeled `clean_keep` samples use generation seed `5301` and fill
  train/valid/test by `25/12/12`. Sixteen non-selected candidates were
  resampled. No replacement duplicates an original or another replacement.
- Replacement peak DeltaT spans `0.043-7.632 K` (median `1.380 K`); none enters
  the physical-hard threshold.
- All 49 replacements passed array preflight, `solver_status=solved`,
  source/side boundary checks, zero boundary-power checks, and the original
  geometry/BC schema.

The reproducible builder is
`scripts/build_heat3d_v4_p5_clean_nohard_dataset.py`. Generated arrays,
manifest, audit, and SHA256 manifest remain ignored under `data/`.

Published/synchronized locations:

- Hugging Face:
  `https://huggingface.co/datasets/133754144X/heat3d-thermal-simulation/tree/main/subsets/heat3d_v4_p5_clean_nohard_v0`
- wsl2/devbox:
  `~/myCodeGitOnly/heat3d-ic/data/heat3d_v4_p5_clean_nohard_v0`

All three locations contain 9660 files (`260013310` bytes). Root hashes:

- `manifest.json`: `248fd8c82eac352c9c224aa30800e26e3cc5f4b869262be5098f70d7acddf4cc`
- `audit_summary.json`: `f0e15b21579a0d2f274ef6abc946119db9b64e21e752053e16e52af081cb7797`
- `sha256_manifest.json`: `a3ca52aa9aa204bd8cbad4f1f6e012f6b5b142af403c319590dfd895e4e40d5c`

## Split Protocol

Tracked split:
`configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json`.

| role | count | meaning |
| --- | ---: | --- |
| `train` | 672 | non-hard clean training pool |
| `valid_iid` | 128 | non-hard clean validation |
| `test_iid` | 128 | non-hard clean test |
| `hard_train_holdout` | 121 | original train physical-hard samples; not trained |
| `hard_challenge_valid` | 12 | original valid physical-hard challenge |
| `hard_challenge_test` | 12 | original test physical-hard challenge |

- `clean_iid` reports `valid_iid` and `test_iid`.
- `hard_challenge` reports the two challenge splits separately; the train hard
  holdout remains available for later controlled studies.
- `all_iid` is a reporting union of each clean split and its corresponding
  original hard split. It does not change training membership.

The original candidate1024 split remains unchanged and is the historical
all-IID reference.

## Baseline And Closeout

`V4P5_01` remains the historical 200-epoch scratch handoff. It preserves the
V4P3_19 model/semantic feature family while preventing an old hard-data
continuation checkpoint from entering clean baseline training.

The frozen P5 clean baseline is now
`V4P5_02_clean_baseline_raw_B28_e600`, generated at
`configs/heat3d_v4/generated/V4P5_02_clean_baseline_raw_B28_e600.yaml`.

- It keeps raw coordinates, plain MSE, B28, 128-sample validation/prediction
  batches, and `prediction_split=valid_iid`.
- `672 / 28 = 24`, so every training batch has the same sample count.
- Its selected best checkpoint is epoch 405 under `valid_base_mse`; the final
  epoch-600 checkpoint is retained only as a trajectory control.
- `V4P5_03_clean_fourier_freq4_B_safe` is the completed frequency-4 Fourier
  negative ablation and is not a future default.

See `docs/v4_closeout.md` for the formal clean/hard/all-IID and final-probe
results.
