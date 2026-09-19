# V7 G2 plotting audit

## Scope

本审计只验证已经存在的 prediction artifacts 和确定性绘图路径，不做新推理、
训练或模型选择。valid_iid figure 使用 case v6p1if1_0003，truth row=3；
sealed 未读，test_iid 没有可核验的固定 V7 G2 prediction archive，因此 test
figure 状态为 blocked 而不是伪造。

## Root cause

旧的通用绘图路径没有把 Therm-FM 的原生 z_x_y (57,65,65) layout、truth-row
binding 和 final-evaluator artifact 写成强制 contract，同时把 exact dense row
和 sparse-display assumptions 混用。这样即使 evaluator 数值正确，raw flatten、
错误的 transpose 或另一个历史 prediction archive 也会造成“图像与性能不一致”。

本次审计没有发现新的模型/evaluator 数值 bug；确认的是 provenance/layout
ambiguity risk。修复路径固定为：

1. full-field Heat3D U-v2 与 Therm-FM 只画 exact dense slices；
2. Therm-FM 明确 transpose (z,x,y) -> (x,y,z)，再按 canonical C-order
   flatten；
3. reference/prediction/error 使用同一 exact dense truth row 与 common color limits；
4. native1024 模型的 full-resolution 图只标为 interpolated diagnostic，同时
   输出 native-point scatter，不将其当作 benchmark-equivalent image。

## Checked bindings

valid fixture 的 binding 为：

- sample_id=v6p1if1_0003, split=valid_iid, truth_row=3；
- full truth archive SHA256 =
  49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb；
- Therm-FM prediction SHA256 =
  fa9d007fbaa76140f3de372ddf7d0fd14a7f5c7c5f5fcb219ec8a2504a882035；
- grid=65×65×57, 240825 nodes；slice is the dense truth-peak layer with 4225
  exact nodes.

The Therm-FM slice audit and full-case evaluator binding passed:
slice RMSE=10.869864 K, slice MAE=10.781683 K, while the same artifact's full-case
evaluator row is RMSE=10.491337 K and MAE=10.376319 K. The difference is expected
because the slice and full field have different domains; it is not evidence of a
layout bug. Heat3D uses the same artifact identity checks.

## Corrected artifacts

Scripts:

- scripts/audit_v7_g2_plotting.py
- scripts/plot_v7_g2_final_fixed.py

Valid-only figures (ignored generated artifacts, hashes are in
docs/results/v7_g2_final_plotting_audit.json):

- research_artifacts/v7_g2_final_figures/valid_fullfield_exact_dense.png
- research_artifacts/v7_g2_final_figures/valid_fullfield_exact_dense.pdf
- research_artifacts/v7_g2_final_figures/valid_native1024_interpolated_diagnostic.png
- research_artifacts/v7_g2_final_figures/valid_native1024_scatter.png

test_iid representative figures remain
BLOCKED_NO_AUTHORIZED_FIXED_PREDICTION_ARCHIVE; no test labels, predictions or
errors were read, and no claim depends on them.
