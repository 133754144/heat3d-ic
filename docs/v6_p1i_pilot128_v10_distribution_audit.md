# V6-P1i pilot128_v3 qualification

Status: **failed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 124/128 (96.875%).
- 20--180 K violations: 0 [].
- 12-bin counts: `[8, 11, 13, 13, 12, 18, 12, 9, 14, 5, 7, 2]`.
- Core q05--q95 maximum gap: 4.244401 K.
- Full maximum gap: 11.571004 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.750206.
- top h versus Reff Spearman: -0.979846.
- top h versus peak partial Spearman controlling power: -0.896014.
- severity versus peak Spearman: 0.889163.
- mean source area versus peak Spearman: -0.022924.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 5.
- Maximum energy error / residual: 1.129e-10 / 1.153e-10.
- Failed gates: `['temperature_bin_ratio']`.

formal1024_v1 remains forbidden unless every gate passes.
