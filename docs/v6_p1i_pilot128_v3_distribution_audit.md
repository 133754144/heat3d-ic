# V6-P1i pilot128_v3 qualification

Status: **failed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 121/128 (94.531%).
- 20--180 K violations: 1 ['v6p1i3_0036'].
- 12-bin counts: `[6, 12, 17, 8, 18, 16, 6, 10, 11, 10, 4, 3]`.
- Core q05--q95 maximum gap: 9.389183 K.
- Full maximum gap: 27.706766 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.780706.
- top h versus Reff Spearman: -0.974003.
- top h versus peak partial Spearman controlling power: -0.922722.
- severity versus peak Spearman: 0.864414.
- mean source area versus peak Spearman: -0.017339.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 4.
- Maximum energy error / residual: 1.015e-10 / 1.154e-10.
- Failed gates: `['temperature_outer_safety', 'temperature_bin_ratio', 'temperature_core_gap', 'temperature_full_gap']`.

formal1024_v1 remains forbidden unless every gate passes.
