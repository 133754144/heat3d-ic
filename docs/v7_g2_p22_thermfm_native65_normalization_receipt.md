# P22 Therm-FM native-65 train-only normalization receipt

状态：`PASS_TRAIN_ONLY_STATS_NATIVE_65X65`。

在 devbox `PDEFormer` 环境中，脚本
`scripts/v7_g2_p22_thermfm_adapter.py` 仅读取 frozen P1i `train` 的 768 个
case-definition 输入与 `samples/deltaT_K` target，统计真实 `65×65` physical pixels；
valid、`test_iid`、sealed 均未参与拟合或 target 索引。共享网格先验证为 C-order
`65×65×57`，无外部 padding/resampling。

| artifact | SHA256 |
|---|---|
| normalization JSON | `76acbd1a09771f63de5f9ff400127d5d11871e7b702cfa12d70570b037644eec` |
| receipt JSON | `a05dd3adeb3a1002face0642a160c58f9a03d1cac1e06fbfa853c2003aa127e3` |
| split manifest | `87aaa84af4b203d0a8ba93ed33b3757d46adc3143dad970371daa0f893ea491f` |
| full-field archive | `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb` |

输入为 741 个 layer-major channels，输出为 57 个 layer channels；所有 mean/std
均为 train-only。该统计 artifact 是 transfer-track qualification 先决条件，不构成
accuracy 结果。
