# V5 core integration boundary

`integration/v5-core` starts from `main@11e9d2f` and carries only the reusable
V5 implementation surface: runner option resolution, true-RMS metrics,
native shape--scale output/losses, train-only global/scale context, regional
pooling/DeepSets, the corresponding RIGNO paths, and synthetic regression
tests.

It deliberately excludes V5 generated YAML, registries and result payloads,
Gate-specific evaluators/diagnostics/closeout scripts, warm-start-only runners,
and all untracked training artifacts. The branch is prepared for review and is
not merged by this closeout.

The migration is described machine-readably in
`docs/v5_core_integration_manifest.json`.
