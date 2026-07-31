# V6-P1i pilot128_v3 qualification

Status: **failed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 119/128 (92.969%).
- 20--180 K violations: 1 ['v6p1i9_0104'].
- 12-bin counts: `[3, 10, 10, 13, 11, 12, 15, 12, 13, 9, 4, 7]`.
- Core q05--q95 maximum gap: 4.493413 K.
- Full maximum gap: 21.906074 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.709766.
- top h versus Reff Spearman: -0.980057.
- top h versus peak partial Spearman controlling power: -0.905346.
- severity versus peak Spearman: 0.854142.
- mean source area versus peak Spearman: 0.104966.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 4.
- Maximum energy error / residual: 7.492e-11 / 1.148e-10.
- Failed gates: `['temperature_outer_safety', 'temperature_full_gap']`.

formal1024_v1 remains forbidden unless every gate passes.
