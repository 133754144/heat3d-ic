# V7 G1 → `main` merge readiness

**Conclusion: `MERGE_BLOCKED`**

本审计只检查 merge readiness，不执行 merge，不改变 G1 science、archive、
checkpoint、support、U-route、metric/statistics，也不访问 `test_iid`/sealed。
G1 science blocker 为 `none`；当前阻塞来自 repository hygiene 和 engineering
边界，而不是 science validity。

## Ref and history snapshot

审计基于 publication package commit `8c16ee19813e826ce4df50581e59358cb6619d9e`
（后续只增加本 readiness 文档的 publication-only commit，不改变下述分类）：

| Ref | SHA |
| --- | --- |
| `origin/main` | `9cb6b374cf2b6f9dde8e5f06da078dee169d9b43` |
| merge base | `9cb6b374cf2b6f9dde8e5f06da078dee169d9b43` |
| `research/v7` at audit | `8c16ee19813e826ce4df50581e59358cb6619d9e` |

`origin/main` is an ancestor of `research/v7`: ahead `113`, behind `0` at the
snapshot above. The branch range is the full V7 G0/G1 development history, not
just the final publication package. A three-way `git merge-tree` check found no
content conflict; absence of a conflict is not sufficient for readiness here.

The range contains 148 changed paths, approximately 177,652 added lines and 37
deleted lines. The largest contributors are:

- `docs/`: approximately 142,338 changed lines, including the 118,694-line
  native H2 closeout receipt and 6,503-line archive manifest;
- `scripts/`: approximately 9,440 lines of audit, training, inference, and
  closeout tooling;
- `rigno/`: approximately 7,283 lines of new runtime/training implementation,
  plus the core `rigno/models/rigno.py` modification.

## File and artifact audit

The current committed range contains no `data/`, `research_artifacts/`, logs,
cache directories, checkpoint files, or prediction files. The only path matched
by a broad “checkpoint” name scan is the source module
`rigno/heat3d_runtime/checkpoint.py`, not a checkpoint artifact. No tracked file
exceeds 50 MB. The `.gitignore` protects `data/`, generated outputs/caches, and
`research_artifacts/v7_g1_formal_archive/`.

The new publication figures are small, deliberate PDF/SVG/review-PNG outputs;
they are not model predictions or raw archive data. Their input hashes and
selection provenance are in `docs/v7_g1_figures/v7_g1_figure_provenance.json`.

Absolute paths occur only in documentation/provenance or historical evidence
code (for example the off-worktree archive locator and the figure source
record). No absolute local path is used as an import/runtime dependency in the
new `rigno/heat3d_runtime`, `rigno/heat3d_training`, or V7 config modules. The
historical finalizer contains devbox paths as evidence-generation metadata and
is not a suitable mainline runtime dependency.

No changed path matches `G2`, `research/v7-g2`, or `g2-baseline`. The G2
worktrees and refs were not modified.

## Content suitability for `main`

| Area | Assessment | Mainline disposition |
| --- | --- | --- |
| `rigno/heat3d_runtime/` | Candidate stable runtime/library code, but added as a large new API surface and coupled to V7 route contracts. | Separate API/behavior review before promotion. |
| `rigno/heat3d_training/` and `rigno/models/rigno.py` | Training and core-model changes are science-adjacent and can affect reusable behavior. | Do not bring through an uncurated research merge; review as a focused implementation change. |
| `configs/heat3d_v7/` | Frozen protocol, support, metric, and route contracts are valuable provenance. | Keep the frozen research record; promote only configs intentionally supported by the mainline interface. |
| `scripts/` and `tools/` | Includes historical G0/Gate-A replay, training, inference, and closeout utilities, not all reusable production tooling. | Split stable utilities from historical-only research scripts. |
| `tests/` and CI | Useful V7 regression/control-plane coverage, but its dependency surface follows the new runtime. | Review with the focused runtime/library change. |
| G1 docs/receipts/manifests | Required scientific audit trail, including retained superseded fail-closed history. | Preserve on `research/v7`/archive; do not make the 118k-line evidence receipt a default mainline dependency. |
| publication figures/summary/close doc | Publication-facing outputs and concise entry points. | Suitable as a reviewed documentation subset, independent of science evidence merge. |

The branch history also contains earlier V7 G0 compatibility work and iterative
Gate-A/H2 recovery implementations. Those are useful historical provenance but
are not all current reusable library surface. Directly merging the whole range
would therefore import development-only code and a very large evidence log in
one operation, even though Git reports no conflict.

## Historical-status and documentation checks

The authoritative G1 entry point is [`v7_g1_close.md`](v7_g1_close.md). It
explicitly places old `FAIL_CLOSED`/`BLOCKED` H2 attempts, V6 `3074`, and the
mixed-domain table in superseded historical/audit status. The historical receipts
remain unchanged as required. `test_iid` is described as future final-model
evaluation and is not a G1 blocker. The H2 naming is unified as one hypothesis
group with H2a/H2b contrasts. Publication-facing H4 wording uses
`physics-scale-only (learned residual scale correction removed)`; the exact
science-sealed receipt is not rewritten.

## Required hygiene before a future merge

This report does not perform the cleanup. A future mainline integration should
first create a curated change set that:

1. separates stable runtime/library and focused tests from G0/G1
   experiment-control scripts;
2. keeps the science archive, large manifests, historical receipts, and
   publication-only evidence in the research/documentation boundary;
3. reviews the `rigno/models/rigno.py` behavior/API change independently; and
4. re-runs import, syntax, static, and regression checks on the curated set.

After that review, a no-FF merge may be considered for the curated branch. No
merge command was run in this audit.

## Readiness classification

| Category | Status | Reason |
| --- | --- | --- |
| Science blocker | `none` | G1 is science-frozen and evidence-sealed. |
| Conflict blocker | `none observed` | Three-way merge-tree check is clean. |
| Engineering blocker | `present` | New runtime/training/core-model surface requires focused review. |
| Repository hygiene blocker | `present` | 113-commit/148-path research history and oversized evidence docs should not be merged wholesale. |
| Documentation blocker | `none for G1 authority` | `v7_g1_close.md` is the current entry point; historical fail-closed records are explicitly retained as history. |
| G2 blocker | `none` | No G2 path/ref was modified. |

**Final readiness: `MERGE_BLOCKED` until the repository is curated into a
mainline-appropriate change set.** This is not a request to alter the sealed
G1 science or delete historical evidence.
