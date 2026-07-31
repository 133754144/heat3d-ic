# V6-P1i pilot128_v3 qualification

Status: **failed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 119/128 (92.969%).
- 20--180 K violations: 0 [].
- 12-bin counts: `[6, 11, 9, 9, 16, 16, 13, 15, 9, 6, 6, 3]`.
- Core q05--q95 maximum gap: 7.361131 K.
- Full maximum gap: 7.583461 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.756054.
- top h versus Reff Spearman: -0.972498.
- top h versus peak partial Spearman controlling power: -0.896720.
- severity versus peak Spearman: 0.861049.
- mean source area versus peak Spearman: -0.018747.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 4.
- Maximum energy error / residual: 1.630e-10 / 1.163e-10.
- Failed gates: `['temperature_bin_ratio']`.

formal1024_v1 remains forbidden unless every gate passes.
