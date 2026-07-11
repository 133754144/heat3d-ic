# V5 Gate 4A Offline Learned Scale-Correction Closeout

## Scope

- `s_phys` is the frozen uncalibrated Gate 1 `z_collapsed_1d_operator`; the learned target is `delta_s = log(s_true / s_phys)`.
- V4 best/final shapes remain frozen. Corrected fields are reconstructed in raw temperature space and then projected only at prescribed Dirichlet nodes.
- This is offline scalar-model feasibility only: no RIGNO, formal loss/configuration, data, label, split, or shape-model update occurred.

## clean_only_zero_shot

| checkpoint | selected candidate | selection role | physics scale log-RMSE | selected scale log-RMSE | physics field RMSE K | selected field RMSE K | global adequate | latent stable gain |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| best | ridge_global_latent_l1 | valid_iid | 1.0305 | 0.3029 | 0.3184 | 0.1708 | True | False |
  - selected vs physics scale/field CI95: `[-0.8553, -0.5951]` / `[-0.1924, -0.1067]`.
  - selected vs uncorrected V4 scale/field CI95: `[0.0359, 0.1358]` / `[0.0191, 0.0668]`.
  - global+latent vs global-only scale/field CI95: `[-0.0823, 0.0338]` / `[-0.0693, -0.0018]`.
| final | ridge_global_latent_l1 | valid_iid | 1.0305 | 0.2999 | 0.3183 | 0.1691 | True | False |
  - selected vs physics scale/field CI95: `[-0.8487, -0.6018]` / `[-0.1912, -0.1087]`.
  - selected vs uncorrected V4 scale/field CI95: `[0.0335, 0.1425]` / `[0.0160, 0.0621]`.
  - global+latent vs global-only scale/field CI95: `[-0.0836, 0.0305]` / `[-0.0732, -0.0020]`.

- Best/final direction consistent: `True`; protocol passed: `True`.

## hard_adapted

| checkpoint | selected candidate | selection role | physics scale log-RMSE | selected scale log-RMSE | physics field RMSE K | selected field RMSE K | global adequate | latent stable gain |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| best | mlp_global | hard_challenge_valid | 2.6358 | 0.3090 | 7.0352 | 3.0349 | True | False |
  - selected vs physics scale/field CI95: `[-2.8192, -1.8015]` / `[-7.2115, -1.4430]`.
  - selected vs uncorrected V4 scale/field CI95: `[-1.4913, -0.4108]` / `[-5.7342, -0.1526]`.
  - global+latent vs global-only scale/field CI95: `[-0.1334, 0.3441]` / `[-0.5938, 1.3521]`.
| final | mlp_global | hard_challenge_valid | 2.6358 | 0.3090 | 7.0364 | 3.0799 | True | False |
  - selected vs physics scale/field CI95: `[-2.8340, -1.7787]` / `[-7.1528, -1.4667]`.
  - selected vs uncorrected V4 scale/field CI95: `[-1.5125, -0.3907]` / `[-5.9226, -0.1012]`.
  - global+latent vs global-only scale/field CI95: `[-0.0677, 0.2251]` / `[-0.4500, 1.4191]`.

- Best/final direction consistent: `True`; protocol passed: `True`.
- `best` clean guard: `{'role': 'valid_iid', 'scale_log_RMSE_ratio_to_physics': 0.3251701893609627, 'field_CV_RMSE_ratio_to_physics': 0.5918145639381477, 'maximum_allowed_ratio': 1.05, 'passed': True}`.
- `final` clean guard: `{'role': 'valid_iid', 'scale_log_RMSE_ratio_to_physics': 0.3251701893609627, 'field_CV_RMSE_ratio_to_physics': 0.5931862746972326, 'maximum_allowed_ratio': 1.05, 'passed': True}`.

## Interpretation And Integrity

- Global-physics adequacy means its selection-role family winner improves both scale log-RMSE and frozen-shape field CV-RMSE versus physics-only.
- A pooled-latent gain is called stable only when its paired-bootstrap CI95 upper bound is below zero for both metrics; a non-gain is a result, not an excuse to use test roles.
- `test_iid` and `hard_challenge_test` are descriptive output rows only; no selection, standardization, threshold, or model fitting uses them.
- Per-sample CSV: `1073` rows; SHA256 `414dd8e71098cbcae5733cd4e6354b2c8666f4f51fc95bc46858e145ea34805d`.
- Cross-role input/full/provenance duplicate groups: `0` / `0` / `0`.
- Overall Gate 4A feasibility pass: `True`.
- `--verify-summary` rebuilds outcomes from CSV only; `--verify-models` recomputes every trained scalar prediction from committed model parameters plus CSV input/latent columns.
