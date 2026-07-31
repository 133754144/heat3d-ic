# V6-P1i pilot128_v3 qualification

Status: **passed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 120/128 (93.750%).
- 20--180 K violations: 0 [].
- 12-bin counts: `[8, 11, 8, 7, 20, 12, 14, 6, 14, 4, 10, 6]`.
- Core q05--q95 maximum gap: 5.772472 K.
- Full maximum gap: 6.075526 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.796781.
- top h versus Reff Spearman: -0.979457.
- top h versus peak partial Spearman controlling power: -0.913437.
- severity versus peak Spearman: 0.675935.
- mean source area versus peak Spearman: -0.067221.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 4.
- Maximum energy error / residual: 8.670e-11 / 1.121e-10.
- Failed gates: `[]`.

formal1024_v1 remains forbidden unless every gate passes.
