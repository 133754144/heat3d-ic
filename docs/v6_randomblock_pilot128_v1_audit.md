# heat3d_v6_randomblock_pilot128_v1 audit

- stage/status: `pilot128` / `failed_temperature_gate`
- samples/groups: 128 / 16
- peak ΔT: 43.651746–159.861564 K (median 89.392427 K)
- realized bins: `{'0': 32, '1': 32, '2': 31, '3': 28, 'outside': 5}`
- intended bins: `{'0': 32, '1': 32, '2': 32, '3': 32}`
- max energy error: 1.477e-10
- max linear residual: 1.192e-10
- k range: 20–600 W/(m·K)
- q range: 1.78941e+09–3.38834e+10 W/m³
- source flux range: 19.0721–525.78 W/cm²
- manifest SHA256: `5ef798b7da547a0a3a464d77593ed2ae77619b43a65bcf1de13dea78c7441a90`
- full-field archive SHA256: `303897e65e9ae6efef3d608dd06fec8275d2fa0c5cfdf8c02c7fb62cda70462a`

所有样本均保留；没有按温度过滤、替换或重采。该阶段没有训练或模型推理。
