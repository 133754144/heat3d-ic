# G2-E2 actual Heat3D exact-resume receipt

状态：`EXACT_RESUME_EQUIVALENCE_PASS`。这是 frozen actual Heat3D RIGNO 的 bounded B24、2-update execution gate，不是 publication accuracy 或长训练证据。

| 项目 | 结果 |
|---|---|
| continuous vs split+resume | PASS |
| final optimizer step | 2 vs 2 |
| parameters max abs difference | 0 |
| optimizer-state max abs difference | 0 |
| loss trajectory | bitwise equal |
| checkpoint reload parameters/state | max abs 0 / 0 |
| runner/config/data mismatch rejection | PASS |
| checkpoint write | atomic `g2_exact_resume_checkpoint_v1` |

运行使用 `g2_e2_epoch_train_batch_0001`、B24、seed 0；只读 train/valid preparation，未访问 P1i test/sealed 或 DeepOHeat official test。原始 JSON `/tmp/g2_e2_heat3d_real_resume.json` SHA256：`2a0fbb46e04c419ff177fb20ff49a7392a597ac2af69b4f665871c6139b64604`。checkpoint SHA256：`a9bc93458db97d53d935b3696783074f63110f5db56a16714ccc51a45d72dd34`。

该 gate 仅证明 bounded resume 机制和 provenance fail-closed；GINO/Transolver 尚未进行 model-level resume，因为 formal training 仍未释放。
