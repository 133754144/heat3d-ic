# V4 Branching Policy

This document defines the V4 branch and worktree boundaries.

## Branch Roles

- `research/v4`: research process control, phase gates, policy updates, and
  merge decisions.
- `research/v4-yaml-registry`: YAML generation, config validation, dry runs, and
  remote-training launch handoff. It must not start training unless the current
  user request explicitly approves a launch.
- `research/v4-results-data`: run-result summaries, checkpoint/evaluation
  reviews, metric tables, and result documentation.
- `research/v4-model-lab`: model, runner, loader, loss, graph-policy, and
  training-code experiments.

## Write Scope

`research/v4` may update:

- V4 control-plane docs, branching policy, stage gates, decision logs, and
  closeout notes.
- It must not add experiment YAML, start training, edit model code, summarize
  fresh results, or absorb implementation, launch, or result-mining work unless
  that work first passed through the appropriate V4 branch.

`research/v4-yaml-registry` may update:

- tracked config registry files, YAML files, dry-run notes, launch readiness
  checklists, and command documentation.
- It must not edit model/runtime code, result-analysis scripts, datasets,
  checkpoints, logs, or result artifacts.
- It must not start training by default. A training launch requires explicit
  current-turn approval and must follow the workflow launch rules.

`research/v4-results-data` may update:

- result summaries, evaluation reports, comparison tables, figure references,
  and documentation derived from existing run outputs.
- It must not edit training runners, model code, config-generation logic,
  YAML registry files, datasets, checkpoints, logs, or prediction artifacts.

`research/v4-model-lab` may update:

- model code, runner code, loader code, losses, metrics implementation, graph
  policy code, and focused code-level tests.
- It must not modify data protocols, published dataset layout, generated
  outputs, checkpoints, logs, YAML registry policy, or result summaries unless
  explicitly approved.

## Merge Gates

Before merging `research/v4-yaml-registry` back to `research/v4`:

- Confirm changed files are limited to tracked config/YAML and launch-readiness
  documentation.
- Run config-load or dry-run checks for every changed YAML when applicable.
- Confirm output directories, log paths, and run IDs do not collide with
  existing artifacts.
- Confirm no training artifacts, checkpoints, logs, predictions, or datasets are
  staged.

Before merging `research/v4-results-data` back to `research/v4`:

- Confirm all reported metrics cite existing artifact paths or committed result
  documents.
- State whether each result is diagnostic, benchmark-candidate, or formal
  benchmark.
- Confirm no training runner, model code, config generator, dataset,
  checkpoint, log, or prediction artifact is staged.

Before merging `research/v4-model-lab` back to `research/v4`:

- Confirm the code change has a narrow research question and a documented
  expected behavioral effect.
- Run focused tests, import checks, or dry-run checks that match the changed
  code path.
- Confirm no dataset protocol change, generated artifact, checkpoint, log, or
  result summary is bundled into the merge.

Before merging any V4 branch back to `research/v4`:

- Run `WF-STATUS`.
- Confirm the branch is clean after commit.
- Push the branch so remote worktrees and servers can fetch the same state.
- End the task with `WF-REPORT`.
