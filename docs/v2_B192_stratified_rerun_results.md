# Heat3D v2 B192 stratified rerun results

Scope: existing `medium1024_gapA_full1024_v2` labels only. All rows use `configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json`, with `valid_iid` as primary validation and `valid_stress` as diagnostic validation. These are research-stage diagnostic results, not formal benchmarks.

## Training Loss Summary

| run | loss | lr/schedule | wd | clip | best_epoch | best_valid_iid | final_valid_iid | final/best | final_valid_stress | wall-clock s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_3e-4` | `background_pseudo_negative` | `0.0003/constant` | 0.0001 | 1.0 | 46 | 0.7368 | 0.7539 | 1.023 | 0.9928 | 737.4 |
| `base_3e-4` | `mse` | `0.0003/constant` | 0.0001 | 1.0 | 45 | 0.4504 | 0.4706 | 1.045 | 0.6768 | 731.7 |
| `hotspot_3e-4` | `background_hotspot` | `0.0003/constant` | 0.0001 | 1.0 | 45 | 0.5001 | 0.5306 | 1.061 | 0.7802 | 731.9 |
| `base_1e-4` | `mse` | `0.0001/constant` | 0.0001 | 1.0 | 50 | 0.4995 | 0.4995 | 1.000 | 0.6654 | 729.7 |
| `base_3e-5` | `mse` | `3e-05/constant` | 0.0001 | 1.0 | 50 | 0.6168 | 0.6168 | 1.000 | 0.6956 | 731.7 |
| `full_1e-4` | `background_pseudo_negative` | `0.0001/constant` | 0.0001 | 1.0 | 50 | 0.8311 | 0.8311 | 1.000 | 1.0589 | 736.1 |
| `base_3e-4_wd1e-8` | `mse` | `0.0003/constant` | 1e-08 | 1.0 | 45 | 0.4505 | 0.4706 | 1.045 | 0.6765 | 730.2 |
| `base_3e-4_wd0` | `mse` | `0.0003/constant` | 0.0 | 1.0 | 45 | 0.4505 | 0.4706 | 1.045 | 0.6764 | 730.8 |
| `base_rapid_decay` | `mse` | `0.0003/rapid_decay` | 0.0 | 1.0 | 50 | 0.5919 | 0.5919 | 1.000 | 0.7224 | 732.5 |
| `base_warmup_cosine` | `mse` | `0.0003/warmup_cosine` | 0.0 | 1.0 | 50 | 0.4887 | 0.4887 | 1.000 | 0.6464 | 734.0 |
| `base_clip0.5` | `mse` | `0.0003/constant` | 0.0 | 0.5 | 45 | 0.4500 | 0.4690 | 1.042 | 0.6782 | 730.5 |
| `base_clip0.1` | `mse` | `0.0003/constant` | 0.0 | 0.1 | 45 | 0.4501 | 0.4683 | 1.040 | 0.6771 | 738.1 |

## Thermal-Field Diagnostics On Valid_IID

| run | checkpoint | field_variance_ratio | spatial_corr | amplitude_ratio | p95_abs_error | p99_abs_error | peak_error | top_k_overlap | hotspot_mae |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_3e-4` | `best` | 1.327 | 0.778 | 1.026 | 0.0499 | 0.0969 | 0.0819 | 0.667 | 0.0653 |
| `full_3e-4` | `final` | 2.002 | 0.783 | 0.973 | 0.0520 | 0.0968 | 0.0901 | 0.652 | 0.0662 |
| `base_3e-4` | `best` | 1.388 | 0.782 | 1.010 | 0.0494 | 0.0959 | 0.0844 | 0.669 | 0.0652 |
| `base_3e-4` | `final` | 2.080 | 0.783 | 0.949 | 0.0516 | 0.0966 | 0.0878 | 0.640 | 0.0658 |
| `hotspot_3e-4` | `best` | 1.419 | 0.786 | 1.005 | 0.0496 | 0.0957 | 0.0825 | 0.663 | 0.0646 |
| `hotspot_3e-4` | `final` | 2.599 | 0.784 | 1.026 | 0.0529 | 0.0954 | 0.0859 | 0.642 | 0.0648 |
| `base_1e-4` | `best` | 1.086 | 0.756 | 0.917 | 0.0495 | 0.1014 | 0.0942 | 0.665 | 0.0683 |
| `base_1e-4` | `final` | 1.086 | 0.756 | 0.917 | 0.0495 | 0.1014 | 0.0942 | 0.665 | 0.0683 |
| `base_3e-5` | `best` | 0.786 | 0.687 | 0.819 | 0.0548 | 0.1129 | 0.1230 | 0.635 | 0.0766 |
| `base_3e-5` | `final` | 0.786 | 0.687 | 0.819 | 0.0548 | 0.1129 | 0.1230 | 0.635 | 0.0766 |
| `full_1e-4` | `best` | 1.285 | 0.754 | 0.937 | 0.0499 | 0.1019 | 0.0954 | 0.663 | 0.0686 |
| `full_1e-4` | `final` | 1.285 | 0.754 | 0.937 | 0.0499 | 0.1019 | 0.0954 | 0.663 | 0.0686 |
| `base_3e-4_wd1e-8` | `best` | 1.388 | 0.782 | 1.010 | 0.0494 | 0.0959 | 0.0844 | 0.669 | 0.0652 |
| `base_3e-4_wd1e-8` | `final` | 2.077 | 0.783 | 0.948 | 0.0516 | 0.0966 | 0.0879 | 0.640 | 0.0658 |
| `base_3e-4_wd0` | `best` | 1.388 | 0.782 | 1.010 | 0.0494 | 0.0959 | 0.0844 | 0.667 | 0.0652 |
| `base_3e-4_wd0` | `final` | 2.076 | 0.783 | 0.949 | 0.0516 | 0.0966 | 0.0879 | 0.640 | 0.0658 |
| `base_rapid_decay` | `best` | 0.924 | 0.694 | 0.837 | 0.0537 | 0.1095 | 0.1171 | 0.638 | 0.0745 |
| `base_rapid_decay` | `final` | 0.924 | 0.694 | 0.837 | 0.0537 | 0.1095 | 0.1171 | 0.638 | 0.0745 |
| `base_warmup_cosine` | `best` | 0.981 | 0.763 | 0.927 | 0.0497 | 0.1001 | 0.0891 | 0.675 | 0.0682 |
| `base_warmup_cosine` | `final` | 0.981 | 0.763 | 0.927 | 0.0497 | 0.1001 | 0.0891 | 0.675 | 0.0682 |
| `base_clip0.5` | `best` | 1.382 | 0.784 | 1.000 | 0.0495 | 0.0960 | 0.0839 | 0.673 | 0.0651 |
| `base_clip0.5` | `final` | 2.043 | 0.785 | 0.989 | 0.0513 | 0.0956 | 0.0880 | 0.660 | 0.0655 |
| `base_clip0.1` | `best` | 1.385 | 0.783 | 1.001 | 0.0495 | 0.0960 | 0.0839 | 0.671 | 0.0651 |
| `base_clip0.1` | `final` | 2.036 | 0.785 | 0.995 | 0.0512 | 0.0954 | 0.0882 | 0.658 | 0.0655 |

## Thermal-Field Diagnostics On Valid_Stress

| run | checkpoint | field_variance_ratio | spatial_corr | amplitude_ratio | p95_abs_error | p99_abs_error | peak_error | top_k_overlap | hotspot_mae |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_3e-4` | `best` | 3.951 | 0.766 | 1.263 | 0.0640 | 0.1109 | 0.0978 | 0.632 | 0.0739 |
| `full_3e-4` | `final` | 6.174 | 0.772 | 1.275 | 0.0678 | 0.1086 | 0.1068 | 0.609 | 0.0709 |
| `base_3e-4` | `best` | 4.115 | 0.771 | 1.265 | 0.0643 | 0.1102 | 0.1007 | 0.639 | 0.0748 |
| `base_3e-4` | `final` | 6.285 | 0.772 | 1.232 | 0.0687 | 0.1090 | 0.1028 | 0.600 | 0.0717 |
| `hotspot_3e-4` | `best` | 4.270 | 0.777 | 1.264 | 0.0639 | 0.1104 | 0.0987 | 0.634 | 0.0749 |
| `hotspot_3e-4` | `final` | 7.732 | 0.772 | 1.328 | 0.0718 | 0.1113 | 0.1039 | 0.591 | 0.0719 |
| `base_1e-4` | `best` | 3.099 | 0.733 | 1.066 | 0.0656 | 0.1126 | 0.1043 | 0.605 | 0.0760 |
| `base_1e-4` | `final` | 3.099 | 0.733 | 1.066 | 0.0656 | 0.1126 | 0.1043 | 0.605 | 0.0760 |
| `base_3e-5` | `best` | 1.541 | 0.648 | 0.901 | 0.0658 | 0.1164 | 0.1219 | 0.568 | 0.0746 |
| `base_3e-5` | `final` | 1.541 | 0.648 | 0.901 | 0.0658 | 0.1164 | 0.1219 | 0.568 | 0.0746 |
| `full_1e-4` | `best` | 3.716 | 0.731 | 1.111 | 0.0670 | 0.1144 | 0.1068 | 0.607 | 0.0765 |
| `full_1e-4` | `final` | 3.716 | 0.731 | 1.111 | 0.0670 | 0.1144 | 0.1068 | 0.607 | 0.0765 |
| `base_3e-4_wd1e-8` | `best` | 4.116 | 0.771 | 1.265 | 0.0643 | 0.1102 | 0.1007 | 0.639 | 0.0748 |
| `base_3e-4_wd1e-8` | `final` | 6.277 | 0.772 | 1.232 | 0.0687 | 0.1090 | 0.1028 | 0.602 | 0.0716 |
| `base_3e-4_wd0` | `best` | 4.115 | 0.771 | 1.265 | 0.0643 | 0.1102 | 0.1007 | 0.639 | 0.0748 |
| `base_3e-4_wd0` | `final` | 6.274 | 0.772 | 1.232 | 0.0687 | 0.1090 | 0.1028 | 0.600 | 0.0716 |
| `base_rapid_decay` | `best` | 2.146 | 0.649 | 0.923 | 0.0674 | 0.1125 | 0.1234 | 0.570 | 0.0730 |
| `base_rapid_decay` | `final` | 2.146 | 0.649 | 0.923 | 0.0674 | 0.1125 | 0.1234 | 0.570 | 0.0730 |
| `base_warmup_cosine` | `best` | 2.914 | 0.747 | 1.089 | 0.0644 | 0.1118 | 0.0997 | 0.620 | 0.0769 |
| `base_warmup_cosine` | `final` | 2.914 | 0.747 | 1.089 | 0.0644 | 0.1118 | 0.0997 | 0.620 | 0.0769 |
| `base_clip0.5` | `best` | 4.107 | 0.773 | 1.248 | 0.0643 | 0.1102 | 0.0996 | 0.643 | 0.0748 |
| `base_clip0.5` | `final` | 6.308 | 0.775 | 1.285 | 0.0683 | 0.1086 | 0.1044 | 0.609 | 0.0723 |
| `base_clip0.1` | `best` | 4.115 | 0.772 | 1.250 | 0.0643 | 0.1102 | 0.0997 | 0.643 | 0.0748 |
| `base_clip0.1` | `final` | 6.292 | 0.776 | 1.296 | 0.0682 | 0.1083 | 0.1050 | 0.605 | 0.0723 |

## Conclusions

1. Stratified split 下最优 B192 参数组仍是 `base_mse` + `lr=3e-4` family。`clip=0.5` 的 best_valid_iid=0.4500，`clip=0.1` 的 final_valid_iid=0.4683；它们与 `base_3e-4` 的 0.4504/0.4706 只差约 0.1%-0.5%，应视为基本持平而不是稳定胜出。
2. Scalar loss 最优与 field-shape 最优不完全一致。`base_3e-4` family 给出最低 valid_iid loss；但 `hotspot_3e-4` 的 valid_iid best spatial correlation 最高（0.786）且 hotspot_mae 最低（0.0646），同时 scalar loss 明显更差（0.5001）。`warmup_cosine` 的 top_k_overlap 较高（0.675）但 scalar loss、correlation 和 error 指标不如 constant `3e-4`。
3. `base_mse` 比 full composite 和 hotspot loss 更适合当前 B192 stratified primary learning。Full composite 在 `3e-4` 下 best_valid_iid=0.7368，在 `1e-4` 下 0.8311；`background_hotspot` 比 full 好，但仍落后于 pure base MSE。
4. `valid_stress` 仍明显更难。最佳 base MSE variants 的 final_valid_stress_loss 约 0.63-0.68，高于 valid_iid 的 0.47 左右；stress split 的 field_variance_ratio 也显著过大，best checkpoint 通常约 4.1，final 约 6.3，说明 stress cases 上预测场空间起伏过冲更强。
5. 当前不建议马上转向 e100 作为主要动作。Several low-lr/schedule runs reach best_epoch=50 with final/best=1.0, but underfit relative to constant `3e-4`; constant `3e-4` already learns valid_iid well by e50. e100 should only be tested after selecting a metric target, likely for `base_1e-4` or `warmup_cosine`, not as a blanket rerun.
6. M1.5/M2 is a more defensible next capacity test than more B192 optimizer sweeps. The runner/split evidence now shows the model can learn IID validation; remaining failures are stress split gap and field-shape over-amplitude, not old-valid split artifacts.
7. Do not reintroduce the current hotspot/background composite loss as-is. If reintroduced, it should be a small staged or diagnostic-targeted term with valid_iid scalar loss and stress field-shape tracked separately, because current composite and hotspot variants hurt primary valid_iid loss.

Split-aware diagnostics also wrote representative slice metadata under each remote run directory at `output/heat3d_v2_runs/<run>/diagnostics/slices/<split>_<checkpoint>/`. These output artifacts are not tracked and should not be committed.
