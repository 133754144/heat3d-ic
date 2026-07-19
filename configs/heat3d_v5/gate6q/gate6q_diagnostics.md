# Gate 6Q oracle-scale and fixed-ridge diagnostics

仅访问 `train` 与 `valid_iid`；未训练、未生成 YAML、未访问 `test/hard/sealed`。

## Oracle-scale upper bounds

| field | point-global % | sample-first % | raw CV K | Q4 point-global % | top10 SSE fraction |
|---|---:|---:|---:|---:|---:|
| e231_oracle_scale | 16.585110 | 14.842554 | 0.117106 | 16.788673 | 0.591539 |
| e543_oracle_scale | 16.406409 | 13.984561 | 0.116602 | 16.711326 | 0.618290 |
| v39_e24_oracle_scale | 16.406058 | 13.984507 | 0.116600 | 16.710742 | 0.618269 |

## Train-fit fixed ridge, one-shot valid_iid

| readout | point-global % | sample-first % | raw CV K | Q4 point-global % |
|---|---:|---:|---:|---:|
| ridge_physics_24 | 42.929394 | 39.885344 | 0.312076 | 42.493047 |
| ridge_pooled_latent_96 | 24.438951 | 21.067362 | 0.175582 | 24.862788 |
| ridge_combined_120 | 23.965667 | 21.199391 | 0.172021 | 24.335776 |

## Conclusion

- scale-only theoretical <20%: `True`
- bottleneck: `representation`
- basis: best oracle=v39_e24_oracle_scale/16.406058%; best fixed ridge=ridge_combined_120/23.965667%; combined coverage-distance Spearman=0.06449711591283647
- unique route: `source-aware DeepSets scale pooling/readout over frozen e543 regional latents with explicit q/k weighting; preserve the e543 shape path and preregister valid-only selection`

该路线仅为只读诊断建议，本轮没有生成配置或启动训练。
