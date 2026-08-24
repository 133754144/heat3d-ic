# V6/P1i P6-A publication evidence tables

All neural accuracy rows use `model_seed0`. WSL2 Attempt 4 is the primary performance result; devbox is a separate overclock-enabled hardware-state replication and is never pooled as additional model seeds.

Paired speedup definition: within each lifecycle seed and identical ordered sample IDs compute FVM/neural paired workload ratio first; report median and min-max over the three lifecycle seeds; never pool 96 samples across machines.

## Main Table

| Route | valid PG/sample-first (%) | valid raw/source/peak/interface (K) | test PG/sample-first (%) | test raw/source/peak/interface (K) | Fresh med/p95 (s) | Q2 (sample/s) | Fresh/Q2 paired speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| E16384_reconstruction | 2.7023/2.7385 | 2.2848/3.8721/4.0178/0.3857 | 2.9920/2.9485 | 2.3891/3.9405/5.7263/0.3555 | 0.8832/0.9830 | 1.7263 | 2.001×/2.005× |
| U_v2_16384_reconstruction | 2.6909/2.7496 | 2.2797/3.8332/3.6075/0.3964 | N/A/N/A | N/A/N/A/N/A/N/A | 0.8662/0.9115 | 1.7595 | 1.976×/2.053× |
| U_v2_direct240825 | 2.8188/2.8003 | 2.3181/3.8561/3.6217/0.4363 | N/A/N/A | N/A/N/A/N/A/N/A | 1.3091/1.5273 | 1.1476 | 1.292×/1.340× |
| E240825_direct_control | 3.0671/3.0394 | 2.4884/4.7346/7.2068/0.4744 | N/A/N/A | N/A/N/A/N/A/N/A | 1.3102/1.3755 | 1.3109 | 1.322×/1.531× |
| FVM240825_reference | N/A/N/A | N/A/N/A/N/A/N/A | N/A/N/A | N/A/N/A/N/A/N/A | 1.7007/1.9608 | 0.8570 | 1.000×/1.000× |

FVM is the reference solution; surrogate-error cells are N/A. The E16384 test row is a one-time corrected confirmatory holdout result obtained after route freeze and was not used for selection. Relative to frozen valid32, test PG is +0.2897 percentage points, raw CV RMSE is +0.1043 K, source RMSE is +0.0684 K, peak RMSE is +1.7085 K, and interface RMSE is -0.0302 K; the peak tail increase is retained rather than hidden.

## Supplementary lifecycle table

The complete 10-row machine/route lifecycle table is frozen in `v6_p1i_p6a_supplementary_lifecycle_table.csv`; it retains cold, fresh, cache-hot, resident, Q2, B16-to-B32, RAM/VRAM, and three-lifecycle median/min/max fields.

## Replication table

The 5-route WSL2-versus-devbox table is frozen in `v6_p1i_p6a_replication_table.csv`. Devbox does not replace the WSL2 primary benchmark.

## Stage decomposition table

The 56-row decomposition is frozen in `v6_p1i_p6a_stage_decomposition_table.csv`. WSL2 E16384 NN/reconstruction median=0.007738753 s and p95=0.085637189 s; the observed upper tail is retained without exclusion or replacement. No tail row was deleted or winsorized.

## Claim/evidence mapping

| ID | Claim | Status | Boundary |
|---|---|---|---|
| C1 | 16k plus reconstruction is the primary valid32 accuracy-latency Pareto family | supported_with_valid32_scope | not a universal topology or dataset claim |
| C2 | E16384 and U-v2 16384 are in the same end-to-end performance class | supported_on_two_reported_hosts | hosts are reported separately and not pooled |
| C3 | U-v2 direct improves valid32 direct-output accuracy at approximately equal WSL2 fresh latency versus E-direct | supported_on_valid32 | diagnostic direct strategies; no test comparison |
| C4 | paired neural/FVM speedup is reproducible across WSL2 primary and devbox replication | supported_with_hardware_state_caveat | within each lifecycle seed and identical ordered sample IDs compute FVM/neural paired workload ratio first; report median and min-max over the three lifecycle seeds; never pool 96 samples across machines |
| C5 | the measured neural service is preprocessing-bound | supported_for_frozen_valid32_workload | WSL2 E16384 NN/reconstruction median=0.007738753 s and p95=0.085637189 s; the observed upper tail is retained without exclusion or replacement |
| C6 | frozen E16384 has one-time quantified accuracy on the corrected confirmatory test_iid holdout | confirmatory_descriptive_only | test opened once after route freeze and never used for selection or tuning; test peak RMSE is higher than valid32 |
| C7 | sealed IID remains an unopened future confirmation boundary | not_evaluated | labels not generated or opened |

`sealed IID` remains unopened because its labels have not been generated. Publication evidence completeness: **GO**.
