# Gate 6O selection contract

Gate 6O 仅使用 `train` 拟合 normalization/context/affine scale calibration，
仅使用 `valid_iid` 评估和选择；禁止访问 `test/hard/sealed`。

Stage 2 冻结 backbone、processor、decoder、bypass、Global FiLM 与 scale
attention，仅允许 `global_scale_hidden` 和 `global_scale_output` 更新。因此
e231/e543 的预注册选择指标为 valid shape CV-RMSE；若相等，依次比较
sample-first CV-relative RMSE 与 raw CV-weighted RMSE。

paired bootstrap、Q1–Q4、branch swap、固定 0.5 ensemble 和 train-only
affine scale calibration 均为诊断项，不参与初始化选择。

Stage 2 固定为 full graph、40 epochs、constant LR `1e-4`、重新初始化
optimizer state，并保存 point-global、sample-first 与 base-MSE best。
