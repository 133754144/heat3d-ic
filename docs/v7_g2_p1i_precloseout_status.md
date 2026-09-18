# V7-G2 P1i pre-closeout

状态：`G2_DEVELOPMENT_FREEZE_RECOMMENDATION`。

本轮没有训练、调参、checkpoint 重选，也没有访问新的 `test_iid`、sealed IID 或
DeepOHeat official100。

## Evidence closure

- `d9d961a` 的 checkpoint→P1i inference `FAIL_CLOSED` 保留，不被本轮结果覆盖。
- `2960ab5` 的 `PASS` 只表示冻结 historical direct240825 sidecar 的身份绑定与
  common-evaluator reproduction，不表示从 checkpoint 重新完成了 inference reproduction。
- 三份 sidecar 的 post-hoc provenance 见
  `docs/results/v7_g2_p23_g1_direct240825_posthoc_provenance_receipt.json`；原 G1 archive
  manifest 未被回写。

## Statistical closure

旧 paired bootstrap 对 RMSE/Hotspot RMSE 使用逐 case metric 平均，与主表 pooled
sufficient-statistics estimator 不一致，已标为 `SUPERSEDED_ESTIMAND_AUDIT_ONLY`。
corrected artifact 为
`docs/results/v7_g2_p1i_corrected_paired_bootstrap.json`：每次 case resample 后，按每个
seed 重新计算主 evaluator estimator，再对三个 seed estimate 取均值；point estimate 与
主表一致。training-seed dispersion 始终写作 **SD across training seeds**，不与 case-level
bootstrap 95% CI 混淆。

## Field-shape / mechanism diagnostics

严格复用了 `rigno/heat3d_v2_field_shape_diagnostics.py`：

- `Corr (%) = 100 × centered_spatial_correlation`；
- `Amp (%) = 100 × amplitude_ratio`，理想值为 100%；
- `top_k_overlap` 使用历史 `top_k=5` 定义；
- Corr/Amp/Hotspot 为 secondary/diagnostic，不改变 primary metric 或 checkpoint selection。

| model | Corr (%) mean ± seed SD | Amp (%) mean ± seed SD | |Amp−100| (%) | top-5 overlap (%) | case-mean hotspot RMSE K |
|---|---:|---:|---:|---:|---:|
| Heat3D | 98.1022 ± 0.2124 | 96.0984 ± 0.3790 | 3.9016 | 1.3542 | 4.6905 |
| Therm-FM | 97.4701 ± 0.1930 | 123.0210 ± 2.6190 | 23.0210 | 2.1875 | 3.0006 |

在预冻结的 high-power、high-HTC、strong-material-heterogeneity strata 中，两者 Corr
仍接近；Heat3D 普遍低估场幅度（约 94–96%），Therm-FM 普遍高估场幅度（约 104–117%）。
误差差异主要体现为幅度校准与热点定位，而不是明显的整体场形状相关性差异。该结论仅是
valid-only descriptive evidence，不是调参依据。

## Negative tests

`docs/results/v7_g2_p1i_negative_domain_gate.json` 通过：故意修改 sample ID、坐标顺序和
truth-row mapping 均得到 `EXPECTED_FAIL`，domain identity gate 保持 fail-closed。

## Recommendation

没有发现新的 implementation/scientific blocker。建议停止 G2 新训练与调参，冻结：
checkpoint、evaluator、metric contract、selection rules 和 claim set。P24 可进入
evaluation-only sealed preregistration review，但本轮没有 unlock sealed。

仍需向审稿/治理流程明确：G1 archive manifest 未枚举 direct sidecar；该问题已有
post-hoc provenance receipt，但不应改写成 checkpoint inference reproduction。
