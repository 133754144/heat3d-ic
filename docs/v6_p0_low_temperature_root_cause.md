# V6-P0：V5 低温升根因与受控反事实

## 方法边界

只选择 train 中三个低温升样本，分别覆盖 strong/nominal/weak top-h 类别：`sample_0376`、`sample_0939`、`sample_0154`。审计脚本复放同一稳态线性 FVM，并一次只改变一个因素；没有调用训练、模型推理或生产 generator。

基线与已保存 temperature 的最大绝对复放误差分别为 `1.71e-13/2.84e-13/3.98e-13 K`。所有反事实的能量平衡误差量级不超过 `2e-11`。网格项是 nearest-neighbor field projection，不是 generator-native mesh-convergence，因此只作为敏感性提示。

## 结果

| sample / BC | baseline peak K | P W | bottom heat fraction | P x0.5 / x2 peak ratio | top h x0.5 / x2 | bottom Robin h=1000 | bottom adiabatic | Rcontact 5e-6 / 1e-5 m2K/W | larger source fixed P | refined 24x24x8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0376 / strong | 0.02281 | 0.1241 | 0.9863 | 0.500 / 2.000 | 1.006 / 0.988 | 17.41x | 23.37x | 1.188 / 1.341 | 1.603x | 0.934x |
| 0939 / nominal | 0.02733 | 0.1675 | 0.9949 | 0.500 / 2.000 | 1.001 / 0.997 | 35.35x | 72.89x | 1.027 / 1.045 | 0.877x | 1.032x |
| 0154 / weak | 0.03002 | 0.1120 | 0.9960 | 0.500 / 2.000 | 1.002 / 0.995 | 28.31x | 87.36x | 1.035 / 1.069 | 0.857x | 1.042x |

## 归因

1. **总功率是线性幅值开关。** 三例的 `P x0.5/x2` 都以机器精度给出 `DeltaT x0.5/x2`。V5 最低功率仅 0.0505 W，而全库中位功率 0.929 W；低功率是低温升的必要直接因素。
2. **固定 300 K 底面是主导散热短路。** 全库 bottom heat fraction 中位数 0.972，三个低温样本为 0.986-0.996。把底面改成 Robin 或 adiabatic 后 peak 上升 17-87 倍。该证据远强于 top-h 变化。
3. **top h 不是这些低温样本的根因。** h 加倍/减半仅改变 peak 约 0.1%-1.2%，因为热流主要没有走顶部。
4. **perfect contact 会进一步压低温升，但不是三例唯一主因。** 加入 `5e-6/1e-5 m2K/W` 中面接触热阻后，peak 上升约 2.7%-34%；幅度随材料/source 位置而变。
5. **固定功率下 source volume 的效应不是单调通则。** dilation 同时改变离底面距离、材料重叠和几何集中度，三例 peak 比为 0.857-1.603；不能把“体积增大”单独解释成温度必然下降。
6. **当前粗网格造成次级敏感性。** 投影到 24x24x8 后 peak 变化约 -6.6% 至 +4.2%，小于底面 BC 效应，但足以要求 V6 做真正的 mesh-convergence。
7. **不是 q clipping 或标定失败。** 1073 个样本 clipping=0，最大功率标定误差 `1.78e-15 W`。

因此，V5 低温升的主要机制是“低总功率 + 与全部样本共享的理想 300 K 底部热汇”；perfect contact 和粗垂向网格强化了这一趋势，top h 不是主导。V6 不应通过删除低温标签或提高 q 下限掩盖问题，而应先用有文献锚点的 package/Robin/contact 路径重建边界条件。

## 复现

```bash
python scripts/audit_heat3d_v5_physics_distribution.py \
  --dataset data/heat3d_v4_p5_clean_nohard_v0 \
  --split-map configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json \
  --counterfactual-sample-id sample_0376 \
  --counterfactual-sample-id sample_0939 \
  --counterfactual-sample-id sample_0154 \
  --output-json /tmp/v6_p0_counterfactual.json
```

有限 contact 与 bottom BC 仅存在于 audit-only operator；结果不得冒充 V5 原始样本或生产 solver 输出。
