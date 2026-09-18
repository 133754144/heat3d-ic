# V7 G2 P1i Field-Shape / Mechanism Diagnostics

Valid-only, read-only diagnostics using the historical V4/V5 field-shape definitions.

| model | Corr (%) mean ± seed SD | Amp (%) mean ± seed SD | |Amp-100| (%) | top-5 overlap (%) | hotspot RMSE K |
|---|---:|---:|---:|---:|---:|
| Heat3D | 98.1022 ± 0.2124 | 96.0984 ± 0.3790 | 3.9016 | 1.3542 | 4.6905 |
| Therm-FM | 97.4701 ± 0.1930 | 123.0210 ± 2.6190 | 23.0210 | 2.1875 | 3.0006 |

## Frozen high-condition mechanism strata

| stratum | model | cases/seed | RMSE K | MAE K | Corr (%) | Amp (%) | hotspot RMSE K | peak abs K | top-5 overlap (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_bottom_h_W_m2K | Heat3D seed0 | 43 | 3.1124 | 2.8738 | 99.2119 | 94.5047 | 5.4294 | 4.6575 | 0.0000 |
| high_material_conductivity_heterogeneity | Heat3D seed0 | 43 | 2.6315 | 2.4012 | 98.5847 | 95.5363 | 5.2043 | 4.2065 | 1.8605 |
| high_top_h_W_m2K | Heat3D seed0 | 43 | 2.2457 | 1.8955 | 97.2064 | 94.9726 | 5.1543 | 4.4047 | 0.4651 |
| high_total_power_W | Heat3D seed0 | 43 | 2.8135 | 2.4278 | 97.5993 | 95.9530 | 6.2095 | 5.3423 | 0.4651 |
| high_bottom_h_W_m2K | Heat3D seed1 | 43 | 3.4026 | 3.1628 | 99.1402 | 94.5353 | 6.2015 | 4.9750 | 0.9302 |
| high_material_conductivity_heterogeneity | Heat3D seed1 | 43 | 2.8076 | 2.5502 | 98.2683 | 96.0287 | 5.6737 | 4.2317 | 2.7907 |
| high_top_h_W_m2K | Heat3D seed1 | 43 | 2.5452 | 2.1757 | 96.6562 | 94.2342 | 5.9502 | 4.7167 | 0.4651 |
| high_total_power_W | Heat3D seed1 | 43 | 3.2096 | 2.8182 | 97.3253 | 95.0613 | 7.1949 | 5.6623 | 1.3953 |
| high_bottom_h_W_m2K | Heat3D seed2 | 43 | 3.3824 | 3.1554 | 99.2740 | 94.3220 | 5.9340 | 5.0279 | 0.4651 |
| high_material_conductivity_heterogeneity | Heat3D seed2 | 43 | 2.8311 | 2.6084 | 98.5638 | 95.4486 | 5.4116 | 4.3646 | 0.9302 |
| high_top_h_W_m2K | Heat3D seed2 | 43 | 2.3868 | 2.0485 | 97.2788 | 94.4910 | 5.5005 | 4.8691 | 0.9302 |
| high_total_power_W | Heat3D seed2 | 43 | 3.0031 | 2.6391 | 97.6797 | 94.8506 | 6.5921 | 5.8196 | 1.8605 |
| high_bottom_h_W_m2K | Therm-FM seed0 | 43 | 1.5958 | 1.3269 | 99.2533 | 110.2499 | 2.8652 | 3.3381 | 2.7907 |
| high_material_conductivity_heterogeneity | Therm-FM seed0 | 43 | 1.6116 | 1.3548 | 98.4752 | 115.4484 | 2.9122 | 3.5725 | 1.3953 |
| high_top_h_W_m2K | Therm-FM seed0 | 43 | 1.4993 | 1.1977 | 97.3892 | 110.1917 | 3.5034 | 3.8039 | 4.1860 |
| high_total_power_W | Therm-FM seed0 | 43 | 1.9783 | 1.6580 | 98.4255 | 103.7254 | 4.3683 | 4.3840 | 4.1860 |
| high_bottom_h_W_m2K | Therm-FM seed1 | 43 | 1.6001 | 1.3322 | 99.2512 | 113.6671 | 2.7786 | 3.6389 | 0.9302 |
| high_material_conductivity_heterogeneity | Therm-FM seed1 | 43 | 1.5818 | 1.3288 | 98.4144 | 118.2955 | 2.8639 | 3.1484 | 0.4651 |
| high_top_h_W_m2K | Therm-FM seed1 | 43 | 1.3997 | 1.0943 | 97.5657 | 113.1051 | 3.5270 | 3.9685 | 0.9302 |
| high_total_power_W | Therm-FM seed1 | 43 | 1.6821 | 1.3589 | 98.5498 | 104.9027 | 4.0320 | 4.0605 | 0.9302 |
| high_bottom_h_W_m2K | Therm-FM seed2 | 43 | 2.0455 | 1.7852 | 99.1023 | 110.8685 | 3.0059 | 3.2890 | 3.2558 |
| high_material_conductivity_heterogeneity | Therm-FM seed2 | 43 | 1.6770 | 1.3991 | 98.5251 | 115.9958 | 2.7093 | 3.2451 | 2.7907 |
| high_top_h_W_m2K | Therm-FM seed2 | 43 | 1.7080 | 1.4259 | 97.6450 | 108.0528 | 3.4035 | 3.5810 | 1.8605 |
| high_total_power_W | Therm-FM seed2 | 43 | 2.2645 | 1.9702 | 98.5246 | 103.2155 | 3.9886 | 3.9417 | 3.2558 |

Corr/Amp/top-k are secondary diagnostics; primary metric and checkpoint selection are unchanged.

No test_iid, sealed IID, or DeepOHeat official100 was read.
