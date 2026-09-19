# V7 G2 plotting audit

## P19 final publication addendum (2026-09-19)

本 addendum 更新早期“test figure blocked”的过程状态；旧 receipt 保留作历史证据，
不覆盖。最终 publication figure route 已在读取任何新 test prediction 前由
`docs/results/v7_g2_publication_figure_selection_receipt.json` 冻结。选择仅使用
input-only physical metadata：valid 代表例为 `v6p1if1_0163`、`0467`、`0650`，
test visualization-only 代表例为 `0740`、`0571`、`0871`，z-layer 也由 receipt
固定。test 推理没有用于模型/检查点选择、指标、颜色范围或 claim。

最终 exact-dense figures 与 machine-checkable audit 位于：

- `research_artifacts/v7_g2_publication_figures/rendered/valid_iid_*_fullfield_exact_dense.png/pdf`
- `research_artifacts/v7_g2_publication_figures/rendered/test_iid_*_fullfield_exact_dense.png/pdf`
- `docs/results/v7_g2_publication_figure_audit.json`
- `docs/results/v7_g2_publication_test_visualization_receipt.json`

Heat3D U-v2 与 Therm-FM 均使用同一 `65×65×57` exact dense slice；色标为该 case
reference slice 的 min/max，误差面板使用 case-level pooled absolute-error 99th
percentile 且不低于 5 K。Therm-FM 的 `z_x_y` 仅在 adapter 中转置一次为 canonical
`x_y_z`；绘图脚本不再反归一化、不插值、不按模型单独缩放。每张图的 slice
RMSE/MAE/peak error 都从绘图前 slice 数组独立重算并写入 audit。

此处 test 图仍是 visualization-only；不构成测试 accuracy evidence，sealed 仍锁定。

## Scope

本审计只验证确定性 prediction artifacts 和绘图路径，不做训练或模型选择。早期
valid fixture `v6p1if1_0003` 与 test-blocked 状态保留在历史 receipt；最终
publication figures 使用上面的 input-only frozen selection。sealed 未读，test 只作
一次固定 visualization-only inference。

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

最终 test_iid representative figures 已完成，但 receipt 明确其
`visualization_only=true`、`used_for_model_selection=false`、`sealed_access=false`。
`v6p1if1_0571` 与 `v6p1if1_0871` 为冻结条件代表例；`v6p1if1_0003` 仍仅保留为
Therm-FM tail-error supplementary diagnostic，不是代表例。
