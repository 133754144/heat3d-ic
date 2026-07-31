# V6-P1i pilot128_v3 qualification

Status: **failed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 127/128 (99.219%).
- 20--180 K violations: 0 [].
- 12-bin counts: `[5, 13, 12, 16, 19, 13, 13, 15, 8, 3, 6, 4]`.
- Core q05--q95 maximum gap: 4.798824 K.
- Full maximum gap: 9.048760 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.727963.
- top h versus Reff Spearman: -0.976979.
- top h versus peak partial Spearman controlling power: -0.926712.
- severity versus peak Spearman: 0.876591.
- mean source area versus peak Spearman: -0.048589.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 4.
- Maximum energy error / residual: 9.065e-11 / 1.153e-10.
- Failed gates: `['temperature_bin_ratio']`.

formal1024_v1 remains forbidden unless every gate passes.
