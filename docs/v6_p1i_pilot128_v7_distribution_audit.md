# V6-P1i pilot128_v3 qualification

Status: **failed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 125/128 (97.656%).
- 20--180 K violations: 0 [].
- 12-bin counts: `[4, 12, 14, 21, 13, 18, 10, 13, 5, 9, 5, 1]`.
- Core q05--q95 maximum gap: 4.305957 K.
- Full maximum gap: 8.918813 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.750275.
- top h versus Reff Spearman: -0.975113.
- top h versus peak partial Spearman controlling power: -0.898794.
- severity versus peak Spearman: 0.864248.
- mean source area versus peak Spearman: -0.089344.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 4.
- Maximum energy error / residual: 6.887e-11 / 1.165e-10.
- Failed gates: `['temperature_bin_ratio']`.

formal1024_v1 remains forbidden unless every gate passes.
