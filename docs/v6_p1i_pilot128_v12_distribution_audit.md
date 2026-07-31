# V6-P1i pilot128_v3 qualification

Status: **failed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 121/128 (94.531%).
- 20--180 K violations: 0 [].
- 12-bin counts: `[5, 7, 5, 11, 9, 16, 13, 19, 12, 13, 8, 3]`.
- Core q05--q95 maximum gap: 7.437408 K.
- Full maximum gap: 7.437408 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.683317.
- top h versus Reff Spearman: -0.981339.
- top h versus peak partial Spearman controlling power: -0.884281.
- severity versus peak Spearman: 0.861879.
- mean source area versus peak Spearman: 0.031296.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 4.
- Maximum energy error / residual: 8.389e-11 / 1.148e-10.
- Failed gates: `['temperature_bin_ratio']`.

formal1024_v1 remains forbidden unless every gate passes.
