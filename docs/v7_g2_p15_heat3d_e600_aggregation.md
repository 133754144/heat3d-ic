# V7 G2-P15：Heat3D e600 valid-only convergence

仅使用 train 768 / valid_iid 128；dense U-v2 只在预注册的 epoch 200/400/600 评估。
P19 将 U-v2 统一称为 `direct-query dense inference`；本页的 e200/IDW 数值仅为历史诊断，
不进入正式比较表或主结论。

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

## Conservative convergence interpretation

`NO_CLEAR_BOUNDARY_RIGHT_CENSORING_STABLE_PLATEAU_UNPROVEN`: 预注册的 native 与
direct-query dense 观测没有显示明确的 boundary right-censoring，但不足以证明稳定 plateau；
不授权自行延长训练。

## Scheduled dense views (mean ± sample SD)

| epoch | native-1024 [%] | IDW historical diagnostic [%] | U-v2 direct-query dense [%] |
|---:|---:|---:|---:|
| 200 | 0.830212 ± 0.025093 | 1.481163 ± 0.073425 | 0.868863 ± 0.034575 |
| 400 | 0.741490 ± 0.017958 | 1.434201 ± 0.037833 | 0.770950 ± 0.034560 |
| 600 | 0.695839 ± 0.008660 | 1.381139 ± 0.018880 | 0.709888 ± 0.007765 |
