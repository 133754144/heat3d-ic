# V6 model closeout and anchored high-resolution inference

Status: **passed**. Evaluation is local CPU, `valid_iid` only.

| nodes | pooling | point-global mean±std % | sample-first mean±std % | raw mean±std K | shape mean | scale-log mean |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | joint_pooling | 0.9308±0.0442 | 0.7540±0.0172 | 0.3982±0.0189 | 0.00746 | 0.00134 |
| 1024 | anchor_derived_scale_pooling | 0.9308±0.0442 | 0.7540±0.0172 | 0.3982±0.0189 | 0.00746 | 0.00134 |
| 2048 | joint_pooling | 8.9581±0.4743 | 8.6524±0.4890 | 3.6069±0.1910 | 0.02311 | 0.08687 |
| 2048 | anchor_derived_scale_pooling | 7.6550±0.1517 | 7.1428±0.1233 | 3.0822±0.0611 | 0.02311 | 0.07363 |
| 4096 | joint_pooling | 9.6147±1.0168 | 9.4741±1.0177 | 3.8886±0.4112 | 0.02859 | 0.09146 |
| 4096 | anchor_derived_scale_pooling | 7.3304±0.1616 | 6.9086±0.0998 | 2.9647±0.0654 | 0.02859 | 0.06803 |
| 8192 | joint_pooling | 10.2740±1.0463 | 10.2110±1.0240 | 4.1573±0.4234 | 0.03149 | 0.09706 |
| 8192 | anchor_derived_scale_pooling | 7.4062±0.1682 | 7.0162±0.1158 | 2.9969±0.0681 | 0.03149 | 0.06750 |

## Volume-support attribution

- Canonical anchors + canonical context: 0.9025% point-global.
- Volume-only + volume context: 144.9835%.
- Volume-only + frozen anchor context: 10.0557%.
- RMSE-gap attribution: context drift 93.65%; remaining local support/graph gap 6.35%.
- Normalized-SSE-gap attribution: context drift 99.52%; remaining local support/graph gap 0.48%.
- 24D context z-score L2 drift: mean 6.091, P95 7.620; per-feature raw/z-score audit is frozen in the JSON.
- Source-aware anchors cover every source box (minimum 4 nodes); volume-only support has 40 zero-covered source boxes and only 88/128 samples with all sources covered.
- Both supports retain finite p2r/r2r/r2p connectivity with zero zero-degree nodes; the residual error is therefore tied to sparse local source representation, not graph disconnection.

## Decision

- High-resolution workflow recognized: **true**.
- Lowest-error high-resolution setting: 4096 nodes with anchor-derived scale/pooling, point-global 7.3304±0.1616%.
- The canonical 1024 source-aware support remains the lowest-error evaluation domain.
- Scope is frozen to the P1h source-aware support family. Test/hard were not accessed.
- Added query nodes still enter the node-aligned encoder/processor path; this is not a pure decoder-only zero-shot query path.
