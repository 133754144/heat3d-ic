# V5 Gate 6C scratch loss ablation preflight

Gate 6C 隔离研究问题：Gate 6B 的 post-hoc warm-start reweighting 失败后，
相同 loss 权重从 random initialization 开始是否仍可能改善 N3。

两组配置都直接继承 `V4P5_07_native_pooled_latent_global_film`。除 run identity、
唯一输出路径和 loss 权重外，resolved dataset、split、N3 architecture、optimizer、
LR schedule、B28、seed、e600、`valid_base_mse` checkpoint selection 与 export
语义完全一致。

| candidate | config ID | shape / scale / relative / raw | initialization | status |
|---|---|---|---|---|
| Scratch-L1 | `V4P5_11_gate6c_scratch_l1_tail_balanced` | `1/1/0.5/1.5` | random | prepared, not started |
| Scratch-L2 | `V4P5_12_gate6c_scratch_l2_shape_balanced` | `1.5/0.5/0.5/1.5` | random | prepared, not started |

数据合同固定为 train-only optimization/normalization 和 valid_iid-only selection。
在候选选择前不得访问或评估 test/hard。本轮没有启动 e600 或 multi-seed。

手动命令仅供后续显式授权使用，本轮不执行：

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_11_gate6c_scratch_l1_tail_balanced.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_12_gate6c_scratch_l2_shape_balanced.yaml
```
