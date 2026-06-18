# Codex Heat3D Workflow

This document is the fixed rule set for Codex work in the Heat3D-IC repository.
`AGENTS.md` is only the short entry point and must not grow into a runbook.

## V4 Branch

- V4 development happens on `research/v4` unless the user explicitly directs a
  different branch.
- V4 starts from the defaults recorded in `docs/v4_starting_defaults.md`.
- The first V4 control-plane change is constraint-only: it may create or update
  `AGENTS.md` and this file, but must not start experiments or generate
  artifacts.

## Default Guardrails

- Do not modify `data/`, `output/`, `checkpoints/`, or `logs/` unless the user
  explicitly requests that path-level scope.
- Do not launch training or long-running evaluation from a control-plane task.
- Do not add a config generator unless the user requests generator
  implementation work.
- Do not add result-analysis scripts unless the user requests analysis tooling.
- Do not treat smoke checks, dry runs, training, evaluation, or artifact export
  as implied by a documentation-only request.

## Change Scope

- Keep documentation-only and code-changing work separated.
- When asked for constraints, plans, or workflow rules, update Markdown docs
  only unless the user gives a broader implementation request.
- Preserve existing research outputs and generated artifacts; use Git status and
  diff checks before finishing.
- Prefer narrow, reviewable commits that name the research phase and the
  control-plane purpose.

## Experiment Rules

- Training, evaluation, prediction export, diagnostics, and result mining all
  require explicit user approval in the current turn.
- Any future experiment config must state its baseline, changed variables,
  expected outputs, and no-write zones before execution.
- Generated configs and scripts should be introduced as separate implementation
  changes, not bundled into rule-setting commits.
