# V6 P1i U4 direct-240825 qualification

Decision: `GO_architecture_freeze_candidate`.

| route | PG % | raw K | source K | peak K | interface K | fresh median s | p95 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| native1024_reconstruction | 3.072016 | 2.930207 | 2.962902 | 2.675661 | 1.416680 | 1.672186 | 1.738907 |
| E16384_reconstruction | 2.702270 | 2.284806 | 3.872082 | 4.017769 | 0.385735 | 3.223886 | 3.624537 |
| E240825_direct | 3.067096 | 2.488376 | 4.734618 | 7.206823 | 0.474360 | 2.476497 | 2.586074 |
| U_direct240825 | 2.818849 | 2.318119 | 3.856053 | 3.621684 | 0.436341 | 0.620292 | 0.645742 |

## Same-output direct Pareto

U-direct minus E240825-direct: PG -0.248246 pp, raw -0.170257 K, fresh median -1.856205 s. U-direct dominates: `True`.

The historical U3 +0.1 pp comparison to E16384 is report-only in U4. Frozen historical artifacts were not modified. The paired replay is valid32-only and does not open test/sealed.
