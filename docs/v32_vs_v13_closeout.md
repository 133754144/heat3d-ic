# V32 versus frozen V13 closeout

Scope is valid_iid-only; test/hard/sealed were not accessed.

This is a `historical noncontemporaneous` comparison: V13 values come from the frozen historical report, while V32 was recomputed with the frozen V5 formulas.

| model/checkpoint | point-global % | sample-first % | raw CV RMSE K |
|---|---:|---:|---:|
| V13 historical | 23.700678 | 20.316459 | 0.167982 |
| V32 point-global e474 | 22.408387 | 21.034804 | 0.160067 |
| V32 final e600 | 22.627739 | 21.024261 | 0.161803 |

V32 e474 versus V13: point-global -1.292291 pp, sample-first +0.718345 pp, raw CV RMSE -0.007915 K.
V32 best→final: point-global +0.219352 pp, sample-first -0.010543 pp, raw CV RMSE +0.001736 K.

Decision: **V32 is not advanced**. Point-global and raw CV RMSE improved, but sample-first regressed and the <20% valid threshold was not met. No seed1/seed2 run is authorized by this closeout.
