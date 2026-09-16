# P22 Therm-FM real-target native-65 qualification

结论：`PASS_T4_REAL_TARGET_NATIVE65`。

在 devbox `PDEFormer`（PyTorch 2.9.0+cu128，RTX 5070）中，使用 Poseidon-T
`93adcbf10f75b45ac3bca3939cc3d3f239e1e663` 初始化 Therm-FM/scOT，输入严格为
`(741,65,65)`，输出为 `(57,65,65)`；未做外部 128×128 padding、resampling 或 sparse
interpolation。训练 fixture 是 frozen train IDs `v6p1if1_0000`、`v6p1if1_0002`，
valid fixture 是 `v6p1if1_0003`，target 为真实 `deltaT_K`，不是 synthetic target。

3 个 bounded optimizer steps 的归一化 p=2 loss 为 `1.19857 → 1.08732 → 0.91150`，
均为 finite 且下降。native forward 输出和 240,825-node denormalized evaluator 均 finite；
checkpoint save/reload 的 model 与 optimizer state 均通过严格 state equality。该 valid
数值只证明 pipeline qualification，不是论文 accuracy 证据。

| 项目 | 结果 |
|---|---:|
| 参数量 | receipt JSON 中记录 |
| native input / target | `[2,741,65,65]` / `[2,57,65,65]` |
| native output | `[1,57,65,65]` |
| peak allocated / reserved | `772,342,784` / `857,735,168` bytes |
| load time | `2.633 s` |
| forward (cold, warm, warm) | `1.494 / 0.040 / 0.019 s` |
| valid evaluator nodes | `240,825` |
| test/sealed | 未访问 |

完整 machine-readable receipt：`docs/v7_g2_p22_thermfm_real_target_qualification.json`。
