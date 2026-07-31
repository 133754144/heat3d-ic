# V6-P1i pilot128_v3 qualification

Status: **failed**.

The frozen 128 cases were audited without filtering, replacement, training, or model inference. A failed pilot is retained as a negative qualification result and cannot authorize formal1024_v1.

## Temperature coverage

- 30--150 K: 125/128 (97.656%).
- 20--180 K violations: 0 [].
- 12-bin counts: `[3, 11, 18, 25, 16, 17, 13, 13, 4, 3, 0, 2]`.
- Core q05--q95 maximum gap: 4.996387 K.
- Full maximum gap: 13.204443 K.
- KDE modes: 1.

## Physical response

- power versus peak Spearman: 0.604777.
- top h versus Reff Spearman: -0.978982.
- top h versus peak partial Spearman controlling power: -0.931019.
- severity versus peak Spearman: 0.738509.
- mean source area versus peak Spearman: 0.027199.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 4.
- Maximum energy error / residual: 8.428e-11 / 1.113e-10.
- Failed gates: `['temperature_no_empty_bins', 'temperature_bin_ratio']`.

formal1024_v1 remains forbidden unless every gate passes.
