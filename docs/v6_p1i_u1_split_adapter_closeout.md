# V6 P1i U1 split-adapter closeout

Identity gate: **PASS** on 32 frozen valid samples using the deterministic CPU backend; every registered intermediate and final tensor is bitwise equal.

The original frozen decoder actually produces 1024 pre-bypass nodes under the asymmetric graph. The split adapter separately applies the frozen pnode-local Encoder transform to `c_out`, so its pre-bypass tensor has N nodes without changing checkpoint parameters.

| N | original pre-bypass | split pre-bypass | PG (%) | raw (K) | source (K) | peak (K) | interface (K) | forward median (s) | ΔPG vs P5-R route (pp) | forward-only speedup |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | [1, 1, 1024, 1] | [1, 1, 8192, 1] | 2.754460 | 2.456087 | 3.738037 | 3.522444 | 0.481731 | 0.003515 | +0.014631 | 6.709x |
| 32768 | [1, 1, 1024, 1] | [1, 1, 32768, 1] | 2.748146 | 2.303823 | 3.814949 | 3.584269 | 0.432104 | 0.007038 | +0.075703 | 2.372x |

## Layer gates

- Frozen Encoder local output transform: **PASS**, 1024→1024 bitwise identity on 32/32 samples.
- Original decoder output contract: **BLOCKED for asymmetric N**, measured pre-bypass shape remains 1024 nodes.
- Split decoder contract: **PASS**, measured pre-bypass shapes are exactly 8192 and 32768 nodes.
- Native shape/scale reconstruction, finite forward, and unchanged checkpoint tree: **PASS**.

## Decision

**GO_checkpoint_preserving_asymmetric_query_feasible**.
Worth entering a separately preregistered 1024→240825 probe: **True**.
The measured 6.71× (8192) and 2.37× (32768) gains are forward-only comparisons, not matched production E2E speedups. P5-R still recommends E16384 for the current production route.
No training, checkpoint update, test/sealed access, 240825 U1 execution, or production-route replacement occurred.
