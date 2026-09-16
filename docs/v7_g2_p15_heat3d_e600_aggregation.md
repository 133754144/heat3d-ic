# V7 G2-P15：Heat3D e600 valid-only convergence

仅使用 train 768 / valid_iid 128；dense U-v2 只在预注册的 epoch 200/400/600 评估。

| seed | best-native epoch | best-native [%] | best-common epoch | best-common U-v2 [%] | final-native [%] | wall h |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 574 | 0.686510 | 600 | 0.703715 | 0.701800 | 0.731 |
| 1 | 593 | 0.696821 | 600 | 0.718607 | 0.699823 | 0.744 |
| 2 | 581 | 0.673527 | 600 | 0.707343 | 0.685947 | 0.752 |

## Across-seed summary

native best = 0.685619 ± 0.011673% (sample SD); 
scheduled U-v2 best = 0.709888 ± 0.007765%; 
final native = 0.695857 ± 0.008638%.

Dense metrics are descriptive and do not alter native checkpoint selection. No test/sealed/official100 data were opened.
