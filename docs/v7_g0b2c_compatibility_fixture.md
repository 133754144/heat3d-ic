# V7-G0b-2c compatibility fixture boundary

状态：`V7 Refactor Compatibility Fixture`，不是 V6 historical evidence。

本轮恢复了 V6/P1i E/U 的 route contract。devbox 已找到历史 route receipts、
input plans、padding record 和 prediction arrays，但未找到可供 binary reconciliation
的 identity-level support、graph-cache 或 reconstruction-map artifacts；WSL2
在本轮不可访问。因此允许建立的临时 fixture 只从 `valid_iid` sample 的
label-independent geometry、material/source/BC metadata 和已冻结的 route contract
在内存或 `/tmp` 中构造，用于验证 V7 refactor 的行为等价边界。它不读取
temperature/deltaT labels，不调用 solver，不生成数据，不访问 `test_iid` 或 sealed
labels，也不写入 `data/`、`output/`、`checkpoints/` 或 `logs/`。

## Required state

```json
{
  "temporary_compatibility_fixture_due_to_wsl2_unavailable": true,
  "historical_artifact_reconciliation": "pending_missing_identity_artifacts",
  "wsl2_mirror_reconciliation": "pending",
  "fixture_label": "V7 Refactor Compatibility Fixture",
  "historical_evidence_replacement": false,
  "publication_headline_eligible": false,
  "g3_eligible": false,
  "final_test_eligible": false
}
```

The temporary fixture must never be silently promoted to a V6 artifact or used for
a publication headline, G3, final test, or a formal latency claim. Any receipt that
uses it must carry the fixture label and the pending reconciliation state.

## Deferred reconciliation checklist

When WSL2 and its historical artifact mount are available again:

1. Verify the historical checkpoint, dataset manifest, full-field SHA-256 and every
   `valid32` sample identity against
   `configs/heat3d_v6_p1i/v7_g0b2c_legacy_freeze_manifest.json`.
2. Reconcile historical E/U conditioning support, query coordinates, support order,
   raw graph metadata, route-specific `edge_targets`, padded metadata, graph hashes,
   model-visible tensors, context/scale tensors, anchor-scale application and
   reconstruction-map hashes against the temporary fixture.
3. Compare E16384, E32768 compatibility samples and U-v2 16384 direct-query samples
   on CPU before any GPU or timing work. Preserve any mismatch as an explicit audit
   finding; do not tune tolerance or rewrite the legacy contract to make it pass.
4. Record checkpoint/data/artifact SHA-256 values and the exact reconciliation command
   in a new receipt. Do not overwrite this pending receipt or the V6 frozen evidence.
5. Clear `pending` only after all required historical-vs-temporary comparisons are
   evidenced and reviewed; until then, keep every temporary result ineligible for
   publication, G3 and final-test claims.

The generation boundary for this phase is the read-only V7 compatibility audit
entrypoint. It may materialize arrays in memory or under `/tmp`; it must not persist
large NPZ/cache files in the repository or in frozen artifact directories.
