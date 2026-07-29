# V6-RandomBlock smoke16 v0 closeout

`smoke16_v0` 在 WSL2、commit `ab83433` 上完成 16 个 240825-node
FVM 求解。物理 gate 通过：最大能量守恒误差 `1.33e-10`、最大线性残差
`1.13e-10`、每个预登记 block 的最小 support 覆盖为 4；没有训练或模型
推理。

温升 gate 未通过。16 个样本的 peak ΔT 为 `48.637–274.999 K`，其中 7
个高功率样本超过 150 K，realized bins 为
`{0:2, 1:4, 2:0, 3:3, outside:7}`。全部结果、manifest 和 full-field
archive SHA 均保留；没有过滤、替换或重采，v0 不进入 pilot。

v1 只把所有 layout 共用的八项总功率表由
`[2,4,8,12,23.9,32,40,47.8] W` 改为
`[1.8,2.5,7,7,16.5,17,25,25] W`。该表依据两个 smoke groups 的
**逐 variant 聚合响应**设置到预登记 bin 中部；没有逐样本 Rth 反算，
没有改变 layout、k/q 分配规则、BC、mesh、support、seed 或 split。

- protocol SHA256: `45845240c21b0350fe50df785024b4a7027abd3dbcd434ab6132c78a60aa5530`
- manifest payload SHA256: `a32850d10775d346bd4764765335c91b088e78dfbb18001a24bb02675ce9ada2`
- full-field archive SHA256: `ea9ded9c7eb39fd9df1704c4eb8b7e623a184a7371704e892086d3f067806feb`
