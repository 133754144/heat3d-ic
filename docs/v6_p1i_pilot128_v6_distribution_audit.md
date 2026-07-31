# V6-P1i pilot128_v3 qualification

Status: **failed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 127/128 (99.219%).
- 20--180 K violations: 0 [].
- 12-bin counts: `[7, 10, 14, 17, 15, 16, 20, 5, 8, 5, 4, 6]`.
- Core q05--q95 maximum gap: 8.776500 K.
- Full maximum gap: 8.776500 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.757782.
- top h versus Reff Spearman: -0.975623.
- top h versus peak partial Spearman controlling power: -0.905623.
- severity versus peak Spearman: 0.849072.
- mean source area versus peak Spearman: -0.058534.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 5.
- Maximum energy error / residual: 1.042e-10 / 1.181e-10.
- Failed gates: `['temperature_core_gap']`.

formal1024_v1 remains forbidden unless every gate passes.
