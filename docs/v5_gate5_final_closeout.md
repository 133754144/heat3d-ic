# V5 Gate 5 final closeout

统一 evaluator commit：`639872abcb0f7afd3b6c2d319a7d395bde75c9a4`。best 为最低 `valid_base_mse`；final 为 epoch 600。

| Model | best epoch | valid point/sample % | test point/sample % | valid/test raw K | final valid/test point % | <20% |
|---|---:|---:|---:|---:|---:|---|
| B0 | 271 | 27.066/27.065 | 27.354/26.562 | 0.1925/0.2363 | 28.164/27.721 | fail |
| N0 | 120 | 31.488/28.311 | 34.082/29.671 | 0.2248/0.3024 | 33.597/35.979 | fail |
| N1 | 261 | 29.965/25.863 | 28.779/24.171 | 0.2172/0.2544 | 30.819/28.372 | fail |
| N3 | 402 | 24.076/20.659 | 24.511/21.588 | 0.1719/0.2178 | 24.384/24.389 | fail |

全部角色、全部指标、best/final checkpoint SHA 与三类 commit 见 JSON。hard roles 仅作冻结后的描述性报告。

## Hard report-only

| Model | Role | best point/sample % | best raw K | final point/sample % | final raw K |
|---|---|---:|---:|---:|---:|
| B0 | hard_train_holdout | 74.473/55.118 | 6.1917 | 74.364/55.069 | 6.1848 |
| B0 | hard_challenge_valid | 85.349/56.163 | 8.9696 | 85.560/56.530 | 8.9906 |
| B0 | hard_challenge_test | 68.327/60.974 | 4.8322 | 68.164/60.820 | 4.8240 |
| N0 | hard_train_holdout | 46.467/37.161 | 3.8376 | 43.417/37.138 | 3.5648 |
| N0 | hard_challenge_valid | 46.440/34.357 | 4.7873 | 45.049/33.637 | 4.6613 |
| N0 | hard_challenge_test | 42.927/42.393 | 3.0109 | 40.818/39.665 | 2.8375 |
| N1 | hard_train_holdout | 44.772/36.902 | 3.6909 | 44.527/35.749 | 3.6738 |
| N1 | hard_challenge_valid | 52.306/34.682 | 5.4855 | 52.312/33.053 | 5.4896 |
| N1 | hard_challenge_test | 44.373/42.547 | 3.0782 | 42.305/40.736 | 2.9301 |
| N3 | hard_train_holdout | 49.163/34.712 | 4.0413 | 49.339/35.253 | 4.0540 |
| N3 | hard_challenge_valid | 51.417/35.940 | 5.3841 | 50.619/36.206 | 5.2986 |
| N3 | hard_challenge_test | 42.893/41.108 | 2.9777 | 43.304/41.442 | 3.0082 |

## N3 improvement (reference minus N3; positive is better)

| Reference | checkpoint | population | point pp | sample-first pp | raw K |
|---|---|---|---:|---:|---:|
| B0 | best | clean | 2.9163 | 5.6895 | 0.0196 |
| B0 | best | hard | 28.2249 | 20.1650 | 2.5301 |
| B0 | final | clean | 3.5553 | 2.7049 | 0.0246 |
| B0 | final | hard | 28.2753 | 19.8391 | 2.5462 |
| N1 | best | clean | 5.0785 | 3.8938 | 0.0409 |
| N1 | best | hard | -0.6747 | 0.7901 | -0.0495 |
| N1 | final | clean | 5.2087 | 3.3703 | 0.0429 |
| N1 | final | hard | -1.3723 | -1.1209 | -0.0891 |

最终候选结论：N3 的 MSE-best clean point-global 均值最低；未通过冻结的 valid/test 均 <20% 可信门槛。
