# Gate 5 loss freeze

N0/N1 的 e10 calibration 仅用于损失尺度和梯度审计，不作为正式性能结果。四项初始权重 `1/1/1/1` 冻结不变，N0/N1 共用同一组权重。

- 两个 calibration 的四项 valid loss 中位数合并后为：shape `0.277`、log-scale `0.515`、relative `0.386`、raw absolute `0.217`。
- 单个 calibration 内最大/最小正 loss 中位数比最高为 `2.95`，低于预设的持续主导阈值 `10x`。
- N0 的 backbone/shape-decoder/scale-head 梯度中位数约为 `3.09/2.26/4.37`；N1 为 `6.88/2.44/24.80`。全部有限且非零，没有核心分支梯度过弱。
- N1 scale-head 梯度较大，但四项 loss 仍处于同一量级，且 optimizer 保留 `gradient_clip_norm=1.0`。没有证据支持为两个候选设置不同权重，也没有证据支持共同降权。

冻结配置：

- `configs/heat3d_v5/generated/V4P5_05_native_physics_only.yaml`
- `configs/heat3d_v5/generated/V4P5_06_native_pooled_latent.yaml`

正式 checkpoint selection 仍为最低 `valid_base_mse`；本轮未启动 e600。
