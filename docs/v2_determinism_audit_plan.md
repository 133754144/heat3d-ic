# Heat3D v2 Determinism Audit Plan

- Why audit: M1 B192 e300 did not reproduce the old M1 B192 e200 trajectory by epoch 200, despite matching core config fields.
- e200/e300 inconsistency summary: epochs/output_dir/run_name differ as intended; `train_metrics_schedule=half_and_final` also changes full-train metric epochs from 100/200 to 150/300.
- Runner semantic-change audit: no evidence that the stdout commit changed train step, loss, optimizer, batch shuffle, seed, best selection, or prediction export; it added display fields and extra metrics reporting only.
- Batch shuffle judgment: train groups are fixed at startup, epoch order uses `np.random.default_rng(seed + epoch)`, so the first 200 epoch permutations should match between e200 and e300.
- Added hash fields: loss summaries now record group counts, short sample-id hashes, per-epoch train batch order hashes, deterministic audit flag, and git commit.
- e5 configs prepared: two B192 deterministic smoke configs differ only in description, run name, and output directory.
- 5epoch smoke: run only if local checks pass and the remote output directories are absent, to compare hash fields and loss history.
