# V6-P1i pilot128_v3 qualification

Status: **failed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 124/128 (96.875%).
- 20--180 K violations: 0 [].
- 12-bin counts: `[6, 15, 10, 14, 13, 13, 14, 16, 9, 6, 6, 2]`.
- Core q05--q95 maximum gap: 3.056808 K.
- Full maximum gap: 15.575083 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.757193.
- top h versus Reff Spearman: -0.978163.
- top h versus peak partial Spearman controlling power: -0.912141.
- severity versus peak Spearman: 0.887252.
- mean source area versus peak Spearman: 0.000773.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 5.
- Maximum energy error / residual: 1.338e-10 / 1.180e-10.
- Failed gates: `['temperature_bin_ratio', 'temperature_full_gap']`.

formal1024_v1 remains forbidden unless every gate passes.
