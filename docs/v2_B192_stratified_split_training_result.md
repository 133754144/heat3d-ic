# Heat3D v2 B192 stratified split training result

Scope: research-stage diagnostic result only. This run used the existing `medium1024_gapA_full1024_v2` samples with an external split map; no labels or arrays were regenerated.

## Setup

- Config: `configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_stratified_seed0.yaml`
- Split map: `configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json`
- Model: M1, latent64 / edge64 / processor_steps4 / mlp2
- Loss: base MSE only
- Optimizer: AdamW, lr `3e-4`, weight_decay `1e-4`, clip `1.0`
- Batch size: `192`
- Epochs: `50`
- Primary validation: `valid_iid`
- Stress validation: `valid_stress`, diagnostic only
- Updates: `4` per epoch, `200` total

## Split mode

The runner consumed `dataset.split_map_path` and reported:

| split | count |
|---|---:|
| train | 704 |
| valid_iid | 104 |
| valid_stress | 88 |
| test_id | 64 |
| test_ood_bc | 24 |
| test_ood_stack | 24 |
| test_ood_combined | 16 |

`valid_loss` is the backward-compatible alias for `valid_iid_loss` in this run. Best checkpoint selection used `valid_iid`.

## Baseline contrast

Old B192 base-MSE run used the original `sample_meta` split:

| run | primary valid | best_epoch | best_valid_loss | final_valid_loss | final/best |
|---|---|---:|---:|---:|---:|
| old split B192 base_mse | old `valid` | 1 | 0.651345 | 1.641265 | 2.52 |
| stratified B192 base_mse | `valid_iid` | 45 | 0.450394 | 0.470600 | 1.045 |

The old `valid` was dominated by stress/OOD-like low-power, high-top-h, diag3 and barrier/TIM samples. The stratified run separates IID validation from stress diagnostics.

## Results

| metric | initial | epoch 1 | epoch 25 | best | final |
|---|---:|---:|---:|---:|---:|
| train full loss | 1.153192 | NA | 0.473399 | NA | 0.452622 |
| valid_iid_loss | 1.295056 | 1.082032 | 0.502796 | 0.450394 @ e45 | 0.470600 |
| valid_stress_loss | 1.199217 | 1.111650 | 0.675358 | 0.632520 @ e45 | 0.676841 |
| valid_iid_raw_deltaT_mse | 0.002417 | 0.002019 | 0.000938 | 0.000840 @ e45 | 0.000878 |
| valid_stress_raw_deltaT_mse | 0.002238 | NA | NA | NA | 0.001263 |

## Interpretation

`valid_iid` no longer shows the old epoch-1 best followed by severe degradation. It improves with training and selects epoch 45. The final/best ratio is about `1.045`, not the old `2.52`.

Train loss and `valid_iid_loss` move in the same direction:

- train full loss: `1.153192 -> 0.452622`
- valid_iid_loss: `1.295056 -> 0.470600`

`valid_stress` is still harder than `valid_iid`, but it also improves:

- valid_stress_loss: `1.199217 -> 0.676841`

This confirms that M1 B192 base-MSE training has basic learning ability on an IID-like validation split. The previous B192 conclusion was confounded by using stress/OOD-like `valid` as the sole primary validation split.

## Next step

Keep `valid_iid` as primary selection and report `valid_stress` separately. The next implementation step is to extend read-only diagnostics to filter metrics by split-map split, so best/final field-shape and condition diagnostics can be reported for `valid_iid` and `valid_stress` separately.
