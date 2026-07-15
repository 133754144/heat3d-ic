# Heat3D Remote Server Connection Workflow

This document covers only SSH connection, remote repository alignment, and
remote Python-environment checks for Heat3D servers.

## Quick Contract

- Read this section before using a Heat3D server, then read
  `WF-REMOTE-CONNECT` for the connection procedure.
- Use the SSH aliases `devbox` or `wsl2`; do not substitute an unverified host.
- The repository remote must use SSH:
  `git@github.com:133754144/heat3d-ic.git`.
- Confirm the remote branch, commit, and worktree status before relying on a
  server checkout.
- Activate the `rigno` conda environment before running project Python
  commands.

## WF-REMOTE-CONNECT: SSH, Repository, And Environment Check

Connect to the target server:

```bash
ssh devbox
# Or: ssh wsl2
conda activate rigno
```

If `conda` is not available in the SSH session, initialize it and activate the
environment before running any other project command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
```

On the server, verify the checkout before updating it. Do not pull when
`git status --short` reports changes that you do not intend to keep.

```bash
cd ~/myCodeGitOnly/heat3d-ic
git remote get-url origin
git branch --show-current
git rev-parse --short HEAD
git status --short
```

For the V5 checkout, after confirming that the worktree is clean, align it with
GitHub:

```bash
git fetch origin
git switch research/v5
git pull --ff-only origin research/v5
git branch --show-current
git rev-parse --short HEAD
git status --short
```

Verify the active Heat3D Python environment:

```bash
python --version
```

For a non-interactive SSH command, initialize conda in the same remote shell:

```bash
ssh devbox 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate rigno && cd ~/myCodeGitOnly/heat3d-ic && python --version'
```

## Connection Troubleshooting

- `ssh devbox` or `ssh wsl2` fails: verify the corresponding host alias, SSH
  key, and network access in `~/.ssh/config`; do not guess another server.
- `conda: command not found`: source `~/miniconda3/etc/profile.d/conda.sh`
  before `conda activate rigno`.
- Project imports or Python packages are missing: verify that `rigno` is active
  and use `python` from that environment rather than assuming `python3` exists.
- The server shows an unexpected branch or commit: inspect `git status --short`
  first, then use the clean-worktree update sequence above.

## WF-EVAL: Checkpoint Evaluation And Diagnostics

Use this after a completed run or when comparing checkpoints.

Required outputs when available:

- `loss_summary.json`
- `run_config.json`
- `params_best.pkl` and `params_final.pkl`
- `best_predictions.npz` and `predictions.npz`
- post-training diagnostics
- final-probe metrics and figures

Metric rules:

- Report normalized validation metrics separately from raw DeltaT metrics.
- For scalar splits, include `valid_base_mse`, raw DeltaT MSE/RMSE, stress
  metrics, best epoch, final epoch, and checkpoint availability.
- For final probe, do not rank only by absolute RMSE. Include
  `relRMSE_DeltaT`, `Tmax_error`, `q_region_RMSE` or `strong_q_RMSE` when
  available, and probe family.
- State whether the evaluation is diagnostic, benchmark-candidate, or formal
  benchmark. V5 results with incomplete frozen metric payloads remain partial.

### Frozen evaluator replay

When an audit names an exact evaluator commit, treat the evaluator source at
that commit as immutable. Use the following sequence before a full replay:

1. Create a detached temporary worktree at the named commit and verify both
   `git rev-parse HEAD` and the SHA256 of the evaluator source.
2. Activate `rigno` in the same remote shell. Do not assume that a
   non-interactive SSH shell has initialized conda.
3. Copy only the required read-only run inputs (`run_config.json`,
   `loss_summary.json`, and the requested checkpoints) into the exact relative
   run directory inside the temporary worktree. Verify copied checkpoint
   SHA256 values against the originals.
4. Do not symlink a top-level `output/` directory. A pre-existing directory can
   turn that command into `output/output`, and evaluators that call
   `Path.resolve()` can reject a run reached through a symlink outside the
   temporary repository root.
5. If the stored temporary `run_config.json` contains an absolute output path,
   rewrite only that temporary copy to the contract-relative run directory.
   Never edit the original run directory or checkpoint.
6. If a later config ID is absent from the old evaluator allowlist, use a
   compatibility adapter that changes only config/run provenance bindings and
   the allowlist. Record the adapter path and SHA256; it must not change metric
   formulas, split handling, normalization, inference, or aggregation.
7. Run a binding preflight before full inference: config ID, run directory,
   checkpoint kind/epoch/hash, 1024 nodes, split hashes, train-only
   normalization/context fitting, and expected roles must all pass.
8. Use the frozen evaluator JSON, not a closeout summary or registry aggregate,
   as the source for per-sample paired analysis. If the historical artifact has
   no per-sample payload, produce the minimum role-only replay required by the
   analysis and record that scope explicitly.
9. Write replay outputs outside the original run directory, then compare every
   numeric metric leaf against the existing collector. If they disagree and
   the task names the old evaluator as authoritative, retain the frozen replay
   and record the disagreement instead of averaging or silently replacing it.

Common failures encountered during Gate 6D were: a nested `output/output`
symlink, repository-root rejection after symlink resolution, an absolute
`output_dir` retained in the copied config, and attempting to read per-sample
rows from aggregate-only closeout/registry data. The sequence above prevents
all four. Prefer checked repository scripts over brittle multiline `python -c`
commands or assuming `jq` is installed on a server.

## WF-SYNC: Remote Result Synchronization

Use this when copying ignored output artifacts between devbox and WSL2.
V5 `output_dir` artifacts should stay on servers, not on the Codex host.

Rules:

- Synchronize ignored artifacts only; do not commit them.
- Prefer non-overwriting sync when copying between remote machines.
- Record source machine, destination machine, source path, destination path,
  file counts, checkpoint counts, and whether existing files were skipped.
- Do not treat copied artifacts as new tracked data unless the user explicitly
  asks for packaging or upload work.

## WF-REPORT: Standard Final Report

Every non-trivial Heat3D task should end with:

- current path, branch, HEAD, and `git status --short`
- files changed and whether they were committed or pushed
- checks run and their result
- whether training/evaluation/data generation happened
- remote machine status if SSH was used
- artifact paths for any generated output
- tracked-file confirmation for `AGENTS.md`
- ignored-artifact confirmation for `data/`, `output/`, checkpoints, logs, and
  predictions
- next recommended task, scoped to one research question

Do not claim benchmark, publication-ready, production-ready, or solved OOD
behavior unless the current task explicitly produced that level of evidence.
