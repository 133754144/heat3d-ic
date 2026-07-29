# V6 final performance preregistration

This contract is frozen before any `test_iid` temperature label is opened.

- Canonical model: `V6_03_V5best_P1h`, seed0, point-global checkpoint e111.
- Primary test resolutions are fixed to 4096 (default), 8192 (full field),
  and 16384 (high resolution).
- 32768 remains experimental and is excluded from the primary test table.
- The workflow is frozen to 1024 conditioning anchors, anchor-derived context
  and scale, source-aware query nodes, anchor-scale reconstruction, and the
  existing layer/interface-aware full-field reconstruction.
- Test results cannot change any model, checkpoint, resolution, graph,
  sampling, context, pooling, or reconstruction decision.
- `hard` remains sealed.

The Git commit containing this document and
`v6_final_performance_preregistration.json` is the auditable freeze point.
