# V6/P1i fixed-geometry publication-runtime gate

Status: **FAIL-CLOSED before GPU null inference**.

The prior branch `research/v6-supplemental-fixed-geometry@171cd5e468fdae3ee599eac11bc1508097f7dd7e`
is retained unchanged as an `invalidated timing attempt`. Its timing, speedup,
setup and break-even values are excluded from this supplemental closeout and it
must not be merged.

## Frozen publication provenance

The authoritative runtime content is frozen at
`04dc85c6ec1b620f026ea546f28a045cd43bbc9c`; the later devbox replication at
`1fa83103fa01dff604c1f377fcc6cd61cdf2ec4d` contains byte-identical runner
files.

| Routes | Exact runner | Function | SHA256 |
|---|---|---|---|
| `E16384_reconstruction`, `E240825_direct_control` | `scripts/benchmark_heat3d_v6_p1i_final_e_service.py` | module-level `main` | `ef8087d3ffe19d4d3d044097baa14d5de39f029652cab8a4064105d53695f326` |
| `U_v2_16384_reconstruction`, `U_v2_direct240825` | `scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py` | module-level `main` | `ec916aa2e4bf37cbc2fb27d6a862f610b185e9b918781ae82b3c6c4e5fb6a834` |

The machine-readable gate binds the publication protocol and every shared
graph, support, reconstruction, model and runtime-helper file by Git-object
SHA256.

## Frozen input design

The intended train-only geometries remain `v6p1if1_0079`, `v6p1if1_0971`,
`v6p1if1_0393`, and `v6p1if1_0056`, selected only by static source-count strata
and sample-ID hash. `K-only` would keep q byte-exact. `K+Q-scale` would preserve
the q mask and normalized spatial distribution while applying positive alpha.
No temperature target is needed by this design.

## Null-gate blocker

The exact final publication runners cannot express the requested one-case,
two-workload null gate without changing their frozen runtime behavior:

- the E runner CLI permits only 4, 8, or 32 samples;
- neither E nor U runner exposes `known-topology/new-physics`;
- E `cache-hot` retrieves the complete cached payload by sample ID, while U
  `cache-hot` likewise reuses a complete prepared payload—both therefore reuse
  old k/q rather than updating dynamic physics;
- the relevant `prepare_host`/`service_one` and `prepare_one` functions are
  nested closures, not frozen public functions an input-only adapter can call;
- the older known-support runner is not the final publication runner and does
  not implement all four frozen routes.

Adding a graph-only dynamic-physics cache, extracting those closures, or
introducing a new timing state would directly violate the instruction not to
reimplement graph, packing, H2D, forward, reconstruction, JIT, cache, or timing.
Accordingly support/graph/map/model-input/prediction equivalence were not
fabricated from a self-replay. The null gate is recorded as
`FAIL_PRE_EXECUTION_CAPABILITY`; no smoke, GPU inference, formal sweep, test,
sealed IID, FVM, label generation, or training was run.
