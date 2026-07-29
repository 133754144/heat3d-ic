# V6 core integration

This branch starts from `main` at `159d349` and imports a strict allowlist from
the frozen V6 evidence archive at `d7f72f1`. It does not merge the research
branch.

Included:

- V6 dual-Robin shared-support loader and 24-dimensional Global Context;
- canonical `V6_03_V5best_P1h` configuration and P1h manifest binding;
- Anchor-derived source-aware high-resolution inference;
- sparse KD-tree edge-list graph construction and versioned graph cache;
- layer/interface-aware full-field reconstruction;
- compact lifecycle, performance, holdout/hard, governance, and phase-index
  evidence;
- deterministic integration and production preflight checkers.

Excluded:

- datasets, checkpoints, predictions, outputs, logs, remote paths;
- generated research YAMLs, registries, per-sample diagnostics, failed-run
  payloads, and historical V6_01/V6_02/V6_04 training configs;
- any wholesale merge of `research/v6-p1h-shared-support`.

The canonical checkpoint remains an external frozen artifact identified by
SHA256. The tracked canonical config uses random initialization and
`training_started=false`; this integration performs no training.

Governance language is frozen as follows:

- hard: preregistered IID stress subgroup within the already-opened corrected
  confirmatory holdout;
- 16384: highest IID-average full-field accuracy mode;
- FVM: legal structured-FVM mesh sensitivity.

The machine-readable allowlist and exclusions are in
`configs/heat3d_v6/v6_core_integration_manifest.json`.

Clean-checkout validation was completed at
`a2055d84ccdd4f16e9a96264c4e9e831da40eb4d`: six V5 checkers, the V6 core
checker, the real-P1h production preflight, configuration dry-run, Python
compilation, and JSON/YAML/CSV parsing all passed. No training or test/hard
access occurred.
