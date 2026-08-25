# V6/P1i publication evidence consolidation

## Evidence roles

WSL2 Attempt 4 is the primary authoritative benchmark. Devbox is an independent, overclock-enabled hardware-state replication and does not replace or pool with WSL2. Both use the same frozen checkpoint, dataset/full-field hashes, three ordered sample permutations, protocol, collector, and CPU worker policy.

The nested historical `valid_iid_inputs_smoke_only` role string is stale legacy metadata. Authority comes from each immutable raw matrix's passed 30-process lifecycle records, formal attempted/completed flags, frozen exactness/seal, and collector `publication_results_generated=true` plus `publication_timing_freeze=GO`. The raw matrix correctly remains in its pre-collector state (`publication_results_generated=false`, `NO_GO_pending_collector`) and was not rewritten.

## Frozen provenance and environment

| Role | Measurement commit | CPU | GPU | Driver | JAX/jaxlib | CUDA runtime |
|---|---|---|---|---|---|---|
| WSL2 primary | `04dc85c6ec1b620f026ea546f28a045cd43bbc9c` | AMD Ryzen 7 9700X 8-Core Processor | NVIDIA GeForce RTX 5070 | N/A | 0.9.1/0.9.1 | N/A |
| devbox replication | `1fa83103fa01dff604c1f377fcc6cd61cdf2ec4d` | N/A | N/A (`CudaDevice(id=0)` only) | N/A | N/A | N/A |

Devbox clock state is user-designated as overclock-enabled, but raw clock values are not recorded. Missing environment fields are intentionally `N/A`; they are not inferred from WSL2.

Frozen identity: checkpoint `51567afe…b90e` (epoch 559), dataset manifest `f19987c…8514`, full-field archive `49023a…3cb`, protocol `325dd8…90b`, collector code `455680…0e4`. Cross-host ordered sample IDs and E/U CPU worker policy are exact.

Three historical high-N implementation fingerprints differ from the current runtime binding:

| Component | Historical SHA | Current runtime SHA | Interpretation |
|---|---|---|---|
| `adapter_and_selector` | `db2cc1f59a61…` | `196a13e823b1…` | exact-safe implementation evolution; current runtime binding and seal govern Attempt 4 |
| `graph_builder` | `fce189e90aa3…` | `4d40e0f851e5…` | exact-safe implementation evolution; current runtime binding and seal govern Attempt 4 |
| `reconstruction` | `8ffaa7680d14…` | `45d8e8ea8d06…` | exact-safe implementation evolution; current runtime binding and seal govern Attempt 4 |

The old binding governs only its historical artifacts. Current authority is the immutable raw matrix, runtime binding, frozen golden hashes, padding/exactness evidence, current seal, and collector output; no frozen raw artifact was edited.

## Primary strategy table (WSL2)

| Route | PG (%) | raw (K) | source (K) | peak (K) | interface (K) | Fresh med/p95 (s) | Resident (s) | Q2 (sample/s) | Fresh/Q2 speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E16384_reconstruction | 2.7023 | 2.2848 | 3.8721 | 4.0178 | 0.3857 | 0.8832/0.9830 | 0.00644 | 1.7263 | 2.001×/2.005× |
| U_v2_16384_reconstruction | 2.6909 | 2.2797 | 3.8332 | 3.6075 | 0.3964 | 0.8662/0.9115 | 0.00405 | 1.7595 | 1.976×/2.053× |
| U_v2_direct240825 | 2.8188 | 2.3181 | 3.8561 | 3.6217 | 0.4363 | 1.3091/1.5273 | 0.05923 | 1.1476 | 1.292×/1.340× |
| E240825_direct_control | 3.0671 | 2.4884 | 4.7346 | 7.2068 | 0.4744 | 1.3102/1.3755 | 0.09460 | 1.3109 | 1.322×/1.531× |
| FVM240825_reference | N/A | N/A | N/A | N/A | N/A | 1.7007/1.9608 | 1.62679 | 0.8570 | 1.000×/1.000× |

FVM accuracy is `Reference/N.A.`: it supplies the reference solution and is not assigned surrogate error. Cold/cache-hot/resident/Q2/B16-to-B32/RAM/VRAM three-lifecycle median and min–max values are retained in `v6_p1i_master_strategy_table.csv`; the table above is the compact primary view.

## Cross-machine conclusion

- `E16384_reconstruction`: devbox/WSL2 Fresh ratio `0.942`; Q2-throughput ratio `1.046`. WSL2 remains primary.
- `U_v2_16384_reconstruction`: devbox/WSL2 Fresh ratio `0.936`; Q2-throughput ratio `1.062`. WSL2 remains primary.
- `U_v2_direct240825`: devbox/WSL2 Fresh ratio `0.895`; Q2-throughput ratio `1.156`. WSL2 remains primary.
- `E240825_direct_control`: devbox/WSL2 Fresh ratio `0.907`; Q2-throughput ratio `1.145`. WSL2 remains primary.
- `FVM240825_reference`: devbox/WSL2 Fresh ratio `0.843`; Q2-throughput ratio `1.105`. WSL2 remains primary.

The two 16k reconstruction routes span Fresh speedup `1.793–2.001×` and Q2 speedup `1.924–2.053×` across the two separately reported machines.

## Stage evidence

- wsl2 `E16384_reconstruction`: packing 0.3155s (35.6%), input_support 0.3053s (34.3%), reconstruction_map 0.1554s (17.4%).
- wsl2 `U_v2_16384_reconstruction`: input_support 0.3391s (39.1%), packing 0.2749s (31.6%), reconstruction_map 0.1508s (17.4%).
- wsl2 `U_v2_direct240825`: graph 0.8373s (62.6%), packing 0.3444s (25.6%), input_support 0.0862s (6.6%).
- wsl2 `E240825_direct_control`: graph 0.6189s (47.8%), packing 0.5100s (39.2%), nn_reconstruction 0.0948s (7.3%).
- devbox `E16384_reconstruction`: input_support 0.3047s (36.4%), packing 0.2982s (35.7%), reconstruction_map 0.1450s (17.2%).
- devbox `U_v2_16384_reconstruction`: input_support 0.3348s (41.4%), packing 0.2459s (30.2%), reconstruction_map 0.1429s (17.6%).
- devbox `U_v2_direct240825`: graph 0.7492s (63.1%), packing 0.2928s (25.0%), input_support 0.0803s (6.8%).
- devbox `E240825_direct_control`: graph 0.5721s (48.9%), packing 0.4494s (37.8%), nn_reconstruction 0.0871s (7.4%).

The dominant measured stages are preprocessing (support/input, graph, packing, and reconstruction-map), while NN/reconstruction is smaller; this supports the bounded claim `preprocessing-bound`, not a universal hardware claim.

## Accuracy-only addition

U-v2 16384 valid32 was absent from tracked aggregate accuracy evidence, so one `model_seed0` frozen-route accuracy-only evaluation was run on `devbox`. No timing result from that execution is used. PG `2.690885%`, raw `2.279715 K`, source `3.833230 K`, peak `3.607479 K`, interface `0.396382 K`.

## Frozen claims

- 16k + reconstruction is the main accuracy-latency Pareto family in this valid32 evidence.
- E/U 16384 are in the same E2E performance class on both machines; neither machine is pooled as extra seeds.
- U-v2 direct has better valid32 accuracy than E-direct at approximately equal WSL2 direct-route Fresh latency.
- The evidence is complete for publication-table construction on valid32. Generalization beyond this scope still requires the separately governed test/sealed confirmation; FVM retains reference physics fidelity.

Final: `publication evidence completeness = GO`.

## P6-A confirmatory amendment

The earlier valid32 closeout statement that test/sealed confirmation remained pending is superseded only for the corrected `test_iid` holdout. After route and checkpoint freeze, `E16384_reconstruction` with `model_seed0` was evaluated once on all 128 ordered test samples: full-field PG `2.992001%`, sample-first `2.948519%`, raw CV RMSE `2.389097 K`, source `3.940479 K`, peak `5.726285 K`, and interface `0.355507 K`. The result is descriptive confirmation and was not used for selection, tuning, or threshold revision. `sealed IID` remains ungenerated and unopened.

## Final peak error-tail amendment

Offline analysis of the frozen valid32 and confirmatory test128 artifacts uses
the preregistered 180 K temperature-rise scale, not a split-specific maximum.
The formal dataset's observed maximum peak ΔT is 173.098431 K. Peak RMSE is
4.018012 K (2.232229% of 180 K) on valid32 and 5.726285 K (3.181269% of
180 K) on test128. Test sample-wise peak relative error is 3.993478% median,
9.133358% p90, 10.437191% p95 and 15.987477% maximum. The largest ten test
samples contribute 52.30% of total test peak-error SSE and 95.45% of the excess
SSE relative to the valid32 mean-SSE reference. The confirmatory increase is
therefore classified as `primarily_tail_driven_with_modest_broad_shift`.
This is an applicability-boundary claim only; it cannot alter the frozen model,
route, thresholds or selection. Full evidence is in
`docs/v6_p1i_error_tail_closeout.md`; sealed IID remains unopened.
