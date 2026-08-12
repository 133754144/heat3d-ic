# V6 P1i U5 direct timing freeze

Decision: `U_direct240825_on_same_output_pareto`. All three routes were measured sequentially in one Python/JAX session at commit `44c897aafa8fe7ce40cceeb714f14f5efd9f8ef6`; historical P5-R latency is excluded.

| route | PG % | raw K | source K | peak K | interface K | fresh median / p95 s |
|---|---:|---:|---:|---:|---:|---:|
| E16384_reconstruction | 2.702279 | 2.284809 | 3.872085 | 4.018083 | 0.385738 | 3.040339 / 3.650992 |
| E240825_direct | 3.067116 | 2.488403 | 4.734638 | 7.206492 | 0.474357 | 1.087986 / 1.125314 |
| U_direct240825 | 2.818822 | 2.318094 | 3.855983 | 3.621800 | 0.436341 | 0.658215 / 0.692390 |

U-direct uses `lean_output_query_v2`: the production span directly constructs only split-decoder inputs, graph and output native-physics tensors. The old full-group route is run only as an untimed deterministic CPU reference and is bitwise exact. No training or test/sealed access occurred.
