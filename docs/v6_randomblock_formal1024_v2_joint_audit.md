# heat3d_v6_randomblock_formal1024_v2 joint-distribution audit

- status: `passed`
- samples/groups: 1024 / 128
- standardized physics-feature rank: 8/8
- unique P/top-h/bottom-h combinations: 7
- q max range: 2.08775e+09–2.55991e+10 W/m³
- CG iterations median/P95/max: 1182.0/1204.0/1221
- max energy/residual: 1.492e-10 / 1.199e-10
- support nodes/block min/P05/median: 2/6.0/19.0

Pearson correlation 的变量顺序见 JSON `feature_names`。该审计只描述
冻结数据生成的 k/q/BC 联合结构；不训练、不推理、不按温度过滤或替换样本。
