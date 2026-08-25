# V6/P1i main integration plan

## Audit result

The comparison `origin/main...research/v6-p1i-training`, using current main
`332ef3f463d91442632c3ebddd4f7549c7895b8d`, contains 1,277 changed paths:
878 under `configs`, 202 under `scripts`, 186 under `docs`, 9 under `rigno`,
and 2 under `tests`. It includes extensive raw/failure/smoke payloads. A direct branch
merge is therefore **NO-GO**: it would import research history and large
diagnostic evidence into reusable main.

Main integration is **GO only as a separate allowlist-based integration task**.
The exact proposed paths are machine-readable in
`configs/heat3d_v6_p1i/v6_p1i_main_integration_manifest.json`. The integration
branch is created from that exact main base and copies only manifest paths.

## Allowed content

The allowlist contains only:

1. stable RIGNO/V6 runtime modules and the minimum runner/evaluator entrypoints;
2. canonical P1h and P1i configs, immutable dataset/full-field manifests and
   the frozen high-N binding;
3. the V6 phase index and lifecycle/governance summaries;
4. compact publication, replication, stage, Pareto and peak-tail tables;
5. final closeout/handoff documents; and
6. core/preflight/publication/final-closeout checkers.

The P1i full test per-sample payload and formal 1,024-row QC table remain on the
research evidence branch/HF archive. Main receives compact aggregate/tail
tables, not the larger source evidence.

## Explicit exclusions

Exclude all datasets, checkpoints, predictions, outputs, logs, raw experiment
directories, failed benchmark attempts, smoke payloads, large per-sample
diagnostics and figures. Failed attempts remain immutable evidence on the
research branch; exclusion from main is not deletion or concealment.

## Integration procedure

1. Create a clean integration branch from the then-current `origin/main`.
2. Copy only the exact manifest allowlist; fail if any copied path matches a
   denylist pattern.
3. Resolve conflicts path-by-path. Do not merge the research branch or import
   directory globs.
4. Run V5 regression checks, V6 core/preflight/final-closeout checkers, config
   dry-run, Python compile, JSON/YAML/CSV parsing and `git diff --check`.
5. Inspect the staged path list against the manifest and require a clean
   checkout replay before opening a PR.

Readiness is therefore `GO_in_clean_checkout_validation` for strict allowlist
integration and `NO_GO` for a whole-history branch merge.

The existing V6 core integration checker was also invoked in this research
worktree. It failed its intentional clean-integration assertion because this
research checkout contains `data`/`output` paths. That is not a V6 scientific
or closeout-checker failure; it is evidence that the allowlist must be replayed
and revalidated in a separate clean integration checkout before a main PR.
