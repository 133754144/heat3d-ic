# V5 core integration boundary

The reusable V5 surface consists of runner option resolution, true-RMS
metrics, native shape--scale output/losses, train-only global/scale context,
regional pooling/DeepSets, the corresponding RIGNO paths, and synthetic
regression tests.

The sole migrated V5 training profile is the canonical V42 contract at
`configs/heat3d_v5/V4P5_42_canonical.yaml`. It is selected only when the
tracked config entry point receives no explicit `--config`; explicit YAML and
direct runner CLI behavior retain their legacy resolution paths. Runtime output
locations remain runner concerns and are not encoded in the canonical profile.

V5 generated experiment YAML, registries and result payloads, Gate-specific
evaluators/diagnostics/closeout scripts, warm-start-only runners, checkpoints,
predictions, datasets, and output/log artifacts are intentionally excluded.
The durable file boundary is described in
`docs/v5_core_integration_manifest.json`.
