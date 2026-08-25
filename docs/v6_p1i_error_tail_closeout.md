# V6/P1i peak error-tail closeout

## Scope and frozen evidence

This is an offline analysis of already-frozen artifacts. It did not train a
model, run inference, modify a checkpoint/route/reconstruction/threshold, or
open sealed IID. The two model-result inputs are the existing
`model_seed0` E16384 reconstruction result on frozen `valid32` and the already
opened 128-sample corrected confirmatory `test_iid` result. Formal truth and
physics covariates come from the frozen 1,024-row formal1024_v1 QC table.

Peak error is the absolute error between predicted and true maximum temperature
rise for one sample. The primary normalized quantity is
`peak RMSE / 180 K`; 180 K is the preregistered outer temperature-rise scale,
not a split-specific observed maximum. The formal dataset's actual observed
maximum peak temperature rise is **173.098431 K** (`v6p1if1_0639`), independently
matched by the frozen distribution audit.

## Primary tail metrics

| Population | n | Peak RMSE (K) | Peak RMSE / 180 K | Sample-wise peak relative error median / p90 / p95 / max | Peak SSE top-5 / top-10 |
|---|---:|---:|---:|---:|---:|
| frozen valid32 | 32 | 4.018012 | 2.232229% | 3.391837% / 7.010589% / 7.744964% / 11.281645% | 47.1429% / 76.3781% |
| confirmatory test128 | 128 | 5.726285 | 3.181269% | 3.993478% / 9.133358% / 10.437191% / 15.987477% | 33.6628% / 52.3026% |

The valid tail table uses the existing order-20260814 per-sample replay because
the compact publication aggregate has no per-sample rows. Its 4.018012 K peak
RMSE differs only at numerical-replay scale from the published 4.0178 K
aggregate; the maximum per-sample peak-error drift across the three already
stored orders is 0.005112 K. No replay or inference was run for this closeout.

The test peak RMSE is 1.4252 times the valid32 value, whereas the median
absolute peak error rises only 1.1548 times. Removing the ten largest test peak
errors gives 4.118927 K, close to the untrimmed valid32 value of 4.018012 K.
Using the valid32 mean peak-error SSE as a fixed descriptive reference, the
largest five test samples account for 62.52% and the largest ten for 95.45% of
the test excess SSE. Therefore the increase is **primarily driven by a small
error tail, with a modest broad distribution shift also present**. This is a
descriptive comparison, not a population-significance test; valid32 and
test128 have different sample counts.

## Physical relationships

Absolute peak error increases descriptively with physical severity, power and
true peak temperature, while relative peak error has little monotonic
dependence on those variables:

| Population | Variable | Absolute-error Pearson / Spearman | Relative-error Pearson / Spearman |
|---|---|---:|---:|
| valid32 | true peak ΔT | 0.259 / 0.307 | -0.222 / -0.102 |
| valid32 | total power | 0.393 / 0.357 | 0.035 / 0.115 |
| valid32 | continuous severity | 0.274 / 0.231 | -0.063 / -0.056 |
| test128 | true peak ΔT | 0.496 / 0.408 | 0.066 / -0.024 |
| test128 | total power | 0.466 / 0.336 | 0.123 / -0.004 |
| test128 | continuous severity | 0.372 / 0.347 | 0.083 / 0.011 |

The test absolute peak-error RMSE rises from 2.575 K in the formal-dataset
true-ΔT Q1 to 8.371 K in Q4, from 3.152 K in power Q1 to 8.958 K in Q4, and
from 2.596 K in severity Q1 to 7.415 K in Q4. Yet the median sample-wise
relative error remains approximately 3.1–5.2% across these bins. The tail is
therefore concentrated in high-energy/high-temperature cases mainly in
absolute units, rather than showing a comparably strong rise after each
sample's true peak is used as denominator. All formal-dataset quartile cuts and
per-bin counts are preserved in the machine-readable result.

## Interpretation boundary

This result adds an applicability boundary: E16384 remains reliable in overall
full-field confirmatory metrics, but peak prediction has a heavier absolute
error tail at high power/high temperature. It does not change the canonical
checkpoint, inference strategy, model selection, thresholds, or any
hyperparameter. The result may be reported in a limitations/error-analysis
section only. Sealed IID remains ungenerated and unopened.

Machine-readable evidence:

- `configs/heat3d_v6_p1i/v6_p1i_error_tail_closeout.json`
- `configs/heat3d_v6_p1i/v6_p1i_error_tail_samples.csv`
