# V7 G1 P1i publication-training dependency audit

The registered publication-training graph is now library-level:

`run_heat3d_v7_formal_p1i_training.py → rigno.heat3d_training.p1i → V7FormalTrainer → RIGNO/Heat3DGraphBuilder`

It does not import `scripts/`, `*_smoke.py`, `check_*`, `*_development.py`, mutate `sys.path`, call cross-script private APIs, or monkey-patch module state. The V1–V6 chain remains available only as a read-only historical/compatibility oracle.

The frozen P1i path prepares train and `valid_iid` once, constructs B24/B32 graph and feature payloads once per run, attaches the input-only 24-D context and q/k regional features explicitly, and uses the existing native shape–scale loss, AdamW, clipping, warmup-cosine schedule, and branch gradient scope. The current P1i contract disables epoch-wise regrouping, so the legacy regroup-and-rebuild path is not reachable.

Static audit details and machine-readable categories are in [v7_g1_p1i_publication_dependency_audit.json](v7_g1_p1i_publication_dependency_audit.json). The completed devbox Full P1i rehearsal is recorded in [v7_g1_p1i_rehearsal_receipt.json](v7_g1_p1i_rehearsal_receipt.json); its timing/resource values are readiness observations, not publication evidence.

The real one-epoch rehearsal used the frozen P1i population (768 train, 128 `valid_iid`) and the registered B24 contract (32 training batches). It executed on `cuda:0` at commit `c5cbb06aea97a276eff65c09e4db0abfdf11c4a5`. Preparation measured 322.442 s total, including 311.315 s graph preparation and 2.312 s feature preprocessing. The run performed 32 train steps, 32 cached executable entries, one validation pass (24.284 s), and one checkpoint/state round-trip. Parameter and optimizer reload were exact (`max_abs=0`); the resumed GPU validation scalar differed by `1.1473894119262695e-05`, retained as a repeatability observation only.

The readiness implementation removes the legacy pre-scan/rebuild path from the production graph, reuses input-derived context rows within one preparation session, caches the explicit V7 step executable per fixed batch signature, and uses one explicit checkpoint writer. The measured run reports 896 feature-transform calls, 896 metadata builds, 36 graph builds, and 32 unique batch signatures; the per-signature compile count is an identified remaining cost for future, separately qualified optimization. No support, graph, padding, model, loss, normalization, batching contract, or scientific evidence was changed.

Deferred technical debt: the historical wrappers still contain duplicated metadata scans, legacy metric/checkpoint wrappers, and script-private library access; they are intentionally not edited in this preflight. Future caching/JIT/batching changes must qualify against the stable V7 reference implementation.
