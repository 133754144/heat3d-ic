# V6-P1i pilot128_v3 qualification

Status: **passed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 123/128 (96.094%).
- 20--180 K violations: 0 [].
- 12-bin counts: `[5, 12, 11, 8, 12, 17, 13, 7, 13, 9, 10, 6]`.
- Core q05--q95 maximum gap: 4.073047 K.
- Full maximum gap: 7.355959 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.777931.
- top h versus Reff Spearman: -0.974713.
- top h versus peak partial Spearman controlling power: -0.878695.
- severity versus peak Spearman: 0.708461.
- mean source area versus peak Spearman: -0.021299.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 5.
- Maximum energy error / residual: 8.798e-11 / 1.216e-10.
- Failed gates: `[]`.

formal1024_v1 remains forbidden unless every gate passes.
