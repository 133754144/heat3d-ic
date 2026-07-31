# V6-P1i formal1024_v1 qualification

Status: **passed**.

The frozen 1024 cases were audited without filtering, replacement, training, or model inference. The formal dataset qualifies only when every preregistered gate passes.

## Temperature coverage

- 30--150 K: 993/1024 (96.973%).
- 20--180 K violations: 0 [].
- 12-bin counts: `[55, 85, 86, 62, 78, 138, 103, 96, 85, 81, 69, 55]`.
- Core q05--q95 maximum gap: 1.138694 K.
- Full maximum gap: 4.076699 K.
- KDE modes: 2.

## Physical response

- power versus peak Spearman: 0.776859.
- top h versus Reff Spearman: -0.978985.
- top h versus peak partial Spearman controlling power: -0.896181.
- severity versus peak Spearman: 0.700835.
- mean source area versus peak Spearman: -0.023199.

## Physics and decision

- Minimum source control volumes: 200.
- Minimum projected local-region support: 4.
- Maximum energy error / residual: 1.314e-10 / 1.225e-10.
- Failed gates: `[]`.

formal1024_v1 is qualified only if the status above is `passed`.
