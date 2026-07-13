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
