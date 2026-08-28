# V7 G1 P1i publication-training dependency audit

The registered publication-training graph is now library-level:

`run_heat3d_v7_formal_p1i_training.py → rigno.heat3d_training.p1i → V7FormalTrainer → RIGNO/Heat3DGraphBuilder`

It does not import `scripts/`, `*_smoke.py`, `check_*`, `*_development.py`, mutate `sys.path`, call cross-script private APIs, or monkey-patch module state. The V1–V6 chain remains available only as a read-only historical/compatibility oracle.

The frozen P1i path prepares train and `valid_iid` once, constructs B24/B32 graph and feature payloads once per run, attaches the input-only 24-D context and q/k regional features explicitly, and uses the existing native shape–scale loss, AdamW, clipping, warmup-cosine schedule, and branch gradient scope. The current P1i contract disables epoch-wise regrouping, so the legacy regroup-and-rebuild path is not reachable.

Static audit details and machine-readable categories are in [v7_g1_p1i_publication_dependency_audit.json](v7_g1_p1i_publication_dependency_audit.json). Actual GPU profile counts and measured impact are recorded only after the devbox rehearsal; no timing value in this audit is publication evidence.

Deferred technical debt: the historical wrappers still contain duplicated metadata scans, legacy metric/checkpoint wrappers, and script-private library access; they are intentionally not edited in this preflight. Future caching/JIT/batching changes must qualify against the stable V7 reference implementation.
