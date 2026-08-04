# V6 P1i controlled cross-resolution closeout

This is a frozen valid-only diagnostic. Test/sealed remained closed; no training, tuning, or checkpoint mutation occurred. Direct-N results are compound-OOD diagnostics and are not model-selection evidence.

## Source-aware nested ladder with training-scale regional mesh

| N | support PG % | sample-first % | full PG % | oracle PG % | Nr | inactive p2r regional |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 14.3520 ± 3.2068 | 9.7259 ± 2.1258 | 14.0779 ± 3.2858 | 3.2059 ± 0.0101 | 255.0 | 46.8 |
| 1024 | 15.1279 ± 1.4684 | 10.1117 ± 0.4712 | 15.4837 ± 1.4623 | 2.9465 ± 0.0134 | 256.0 | 0.0 |
| 2048 | 23.0441 ± 1.7949 | 16.5117 ± 1.4496 | 23.3745 ± 1.7598 | 2.6096 ± 0.0125 | 256.0 | 0.0 |
| 4096 | 31.9351 ± 1.1072 | 25.2219 ± 0.7441 | 32.1779 ± 1.1091 | 2.3133 ± 0.0034 | 256.0 | 0.0 |
| 8192 | 46.0543 ± 1.6472 | 37.3515 ± 0.9656 | 46.2708 ± 1.6323 | 2.0792 ± 0.0019 | 256.0 | 0.0 |
| 16384 | 69.1684 ± 1.2855 | 57.1148 ± 0.6212 | 69.3475 ± 1.2745 | 1.8045 ± 0.0020 | 256.0 | 0.0 |

## A-D factor diagnostic

| N | A source/fixed | B source/growing | C structured/fixed | D structured/growing | B-A | C-A | D-C |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 13.6759 | 13.6759 | 7301.5645 | 7301.5165 | +0.0000 | +7287.8886 | -0.0480 |
| 4096 | 32.3614 | 33.5666 | 426.6367 | 403.6025 | +1.2052 | +394.2753 | -23.0342 |
| 16384 | 68.4452 | 70.8744 | 41.5146 | 42.9975 | +2.4292 | -26.9306 | +1.4829 |
| 65536 | 97.1742 | 102.2708 | 4.7030 | 4.7151 | +5.0966 | -92.4711 | +0.0121 |

## Attribution

The largest fixed-Nr support-distribution contrast is C-A=+7287.8886 percentage points at N=1024. The largest source-aware regional-scale contrast is B-A=+5.0966 points at N=65536.

Interpretation uses the full A-D pattern together with Global Context, QK, physical-scale, predicted-scale, graph-size, and oracle-floor drift. It does not attribute the existing structured direct-N curve to resolution alone.

The source-aware ladder is conservative and label-independent, but it is not checkpoint-IID: the checkpoint was trained on the frozen sparse P1i support, whereas this audit redistributes full-field control volume, q, and conductivity moments onto nested supports. Therefore even N=1024 is a support-measure/discretization diagnostic, not a replay of the training support.

Oracle full-field error decreases monotonically from 3.2059% at N=512 to 1.8045% at N=16384 while model error increases after N=1024. Sampling resolution is therefore not the primary failure. The sign-changing C-A contrast also shows that structured support and resolution cannot be interpreted independently in Direct-N mode.

## Graph-scale and feature drift

| N | mean p2r in-degree | mean p2r edges | mean r2r edges | Global Context z L2 drift | QK L2 drift | abs d log(s_phys) | abs d predicted log-scale |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 3.971 | 1012.6 | 4110.2 | 4.747 | 0.728 | 0.001655 | 0.054 |
| 1024 | 9.775 | 2502.3 | 4141.1 | 0.000 | 0.000 | 0.000000 | 0.000 |
| 2048 | 27.469 | 7031.9 | 4141.3 | 3.376 | 0.363 | 0.001074 | 0.064 |
| 4096 | 63.454 | 16244.1 | 4152.5 | 5.150 | 0.289 | 0.001447 | 0.133 |
| 8192 | 149.585 | 38293.7 | 4137.3 | 6.891 | 0.302 | 0.001460 | 0.223 |
| 16384 | 307.966 | 78839.4 | 4138.4 | 8.836 | 0.234 | 0.001527 | 0.354 |

With Nr fixed near 256, mean p2r regional in-degree rises from 9.775 at N=1024 to 307.966 at N=16384 while r2r edge count remains near 4.1k. Global Context z drift and predicted log-scale drift increase with N even though physical log-scale drift stays near 0.0015. The supported attribution is therefore: support-distribution/measure shift plus p2r graph-scale and context/scale-response drift are primary; changing Nr is secondary in the source-aware A-B contrast.

At N=512, simplex-centroid refinement reaches Nr=255 but leaves 46.8 regional nodes inactive on average in p2r/r2p. Every physical node remains covered and the r2r graph remains one connected component; this is a reported sub-resolution boundary rather than a hidden pass condition.

## Reproducibility and governance

- Fixed subset: 32 valid_iid samples.
- Discretization seeds: [0, 1, 2, 3].
- Checkpoint SHA256: `51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e`.
- Dataset manifest SHA256: `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514`.
- Full-field SHA256: `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb`.
- test accessed: false; sealed accessed: false; training/tuning: false.
