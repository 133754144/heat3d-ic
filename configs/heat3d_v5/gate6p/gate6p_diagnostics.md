# Gate 6P read-only scale-path diagnostics

仅访问 `train` 与 `valid_iid`；未启动训练，未修改 checkpoint，`test/hard/sealed` 均未访问。

## Checkpoint transplant

| field | point-global % | sample-first % | raw CV K | shape CV | scale log |
|---|---:|---:|---:|---:|---:|
| e231 | 21.944499 | 20.606163 | 0.157325 | 0.148425 | 0.178125 |
| e543 | 22.727394 | 19.941327 | 0.163245 | 0.139848 | 0.175589 |
| v39_e24 | 22.444176 | 20.014313 | 0.161238 | 0.139849 | 0.173168 |
| e543_plus_e231_global_scale_mlp | 22.599871 | 19.825591 | 0.162216 | 0.139847 | 0.175315 |
| e543_plus_e231_mlp_scale_attention | 22.917721 | 19.864990 | 0.164518 | 0.139846 | 0.179652 |
| e543_plus_e231_complete_scale_head | 22.918201 | 19.865205 | 0.164521 | 0.139847 | 0.179655 |

## Frozen scale-feature readout CV

| feature set | width | log-scale RMSE | Q4 RMSE | bias |
|---|---:|---:|---:|---:|
| physics_24 | 24 | 0.478077 | 0.572415 | -0.002224 |
| pooled_latent_96 | 96 | 0.137743 | 0.169058 | -0.002743 |
| combined_120 | 120 | 0.131140 | 0.158644 | -0.002188 |
| physics_operator_no_readout | 0 | 1.215721 | 1.633992 | -0.959738 |

## Conclusion

- bottleneck: `objective`
- evidence: best transplant point-global=22.599871% (e543_plus_e231_global_scale_mlp); physics/combined Q4 readout RMSE=0.572415/0.158644; coverage Spearman=0.05095307917888562; e231 MLP transplant point/sample=22.599871/19.825591%; MLP+attention point=22.917721%
- next candidate: `e543 frozen backbone/shape/scale-attention plus e231 global-scale-MLP initialization; train only global scale MLP with a preregistered sample-first and Q4-balanced scale objective`

该候选仅为诊断结论，本轮没有生成或启动训练。
