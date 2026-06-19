# Codex Heat3D Workflow

This document is the fixed rule set for Codex work in the Heat3D-IC repository.
`AGENTS.md` is only the short entry point and must not grow into a runbook.

## Quick Contract

- Read this section first, then read only the `WF-*` section(s) relevant to the
  current task.
- For V4 work, expected branch is `research/v4` unless the user names another
  branch.
- For branch creation, merge, or cross-branch write-scope questions, read
  `docs/v4_branching_policy.md`.
- `AGENTS.md` is a tracked short entry file. It may be staged or committed when
  the user asks for constraint or control-plane files.
- File-changing tasks should be committed and pushed to GitHub when complete
  unless the user explicitly says not to; servers pull latest changes from
  GitHub.
- Do not modify `data/`, `output/`, `checkpoints/`, or `logs/` unless the user
  explicitly requests those paths.
- Do not start training, evaluation, data generation, config-generator work, or
  result-analysis tooling unless explicitly requested.
- End non-trivial Heat3D tasks with `WF-REPORT`.

## V4 Branch

- V4 development happens on `research/v4` unless the user explicitly directs a
  different branch.
- V4 starts from the defaults recorded in `docs/v4_starting_defaults.md`.
- Control-plane documentation tasks are constraint-only unless the user
  explicitly requests implementation work.

## Guardrails

- Documentation-only tasks may only update Markdown.
- Training, evaluation, data generation, config-generator work, result-analysis
  tooling, and artifact export require explicit user approval in the current
  turn.
- Do not modify `data/`, `output/`, `checkpoints/`, or `logs/` unless those
  paths are explicitly in scope.
- Future experiment configs must state baseline, changed variables, expected
  outputs, and no-write zones.

## Repeated Prompt Blocks To Retire

The following workflows were repeatedly spelled out during v2/v3. Future user
prompts should reference the workflow name instead of restating the full block.
Codex must expand the referenced workflow, adapt paths to the current branch,
and report any mismatch before changing files or launching remote work.

| Workflow | Replaces repeated prompt content |
| --- | --- |
| `WF-STATUS` | branch, HEAD, status, tracked-entry, and no-artifact checks |
| `WF-REMOTE-ENV` | devbox / WSL2 repo, conda, Python, GPU, disk, and process checks |
| `WF-SHELL-SAFETY` | shell quoting, heredoc, pipe, redirection, and remote snippet safety |
| `WF-CONFIG` | config/YAML preparation, dry-run command building, and output-collision checks |
| `WF-LAUNCH` | tmux training launch and monitor commands |
| `WF-EVAL` | checkpoint evaluation, diagnostics, final-probe inference, and metric table rules |
| `WF-SYNC` | WSL2/devbox result synchronization without overwriting existing artifacts |
| `WF-REPORT` | standard final answer format |

## V4 Skills

- `heat3d-v4-yaml-registry`: prepare registry/YAML, regenerate CSV/YAML, dry-run,
  and checker; no launch.
- `heat3d-v4-remote-run` (`skills/heat3d-v4-remote-run`): after explicit launch
  approval, get `config_id`, push local changes, remote pull, tmux launch, and
  return log monitor commands.
- `heat3d-v4-result-collector` (`skills/heat3d-v4-result-collector`): read
  `loss_summary.json` and related payloads, then update only CSV `result_*`
  columns.

## WF-STATUS: Local Git And Artifact Hygiene

Run this at the start and end of any non-trivial Heat3D task.

Required checks:

```bash
pwd
git branch --show-current
git rev-parse --short HEAD
git status --short
git ls-files AGENTS.md docs/codex_heat3d_workflow.md
git status --short --ignored data output checkpoint checkpoints logs log | head -n 80
```

Rules:

- `AGENTS.md` is tracked. If a local exclude rule still lists it, use
  `git add -f AGENTS.md` only when the user explicitly asks to stage or commit
  the entry rules.
- Do not commit `data/`, `output/`, checkpoints, predictions, logs, tmux logs,
  `__pycache__/`, `*.pyc`, or `.DS_Store`.
- If tracked changes exist outside the requested scope, stop and report them
  before editing.
- If only ignored artifacts exist, keep working but mention that they were not
  committed.
- For V4 work, expected branch is `research/v4` unless the user names another
  branch.

## WF-REMOTE-ENV: Remote Preflight

Use this before any remote training, evaluation, diagnostics, or result mining.

Required remote checks:

```bash
ssh devbox
cd ~/myCodeGitOnly/heat3d-ic
git branch --show-current
git rev-parse --short HEAD
git status --short
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
python --version
python -c "import jax; print(jax.devices())"
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not in PATH"
df -h .
pgrep -af "python|train_heat3d|heat3d_v" || true
tmux ls || true
```

For WSL2, replace `ssh devbox` with `ssh wsl2`.

Rules:

- Always activate `rigno` after entering the SSH environment.
- In non-interactive SSH commands, source `conda.sh` before `conda activate
  rigno` if `conda` is not already available.
- Do not use `set -u` around conda activation.
- `nvidia-smi` is optional on WSL-backed remotes; CUDA usability is checked by
  `jax.devices()` after activating `rigno`.
- Do not start a new run if an active Heat3D training process is already using
  the target machine unless the user explicitly approves concurrent training.
- Confirm branch and HEAD before trusting remote output.

## WF-SHELL-SAFETY: Quoting And Command Robustness

Use this before writing or running any multiline SSH, heredoc, `python - <<`,
pipe, `tee`, or redirection command. V3 repeatedly lost time to small shell
errors, especially when remote Python snippets were wrapped in shell quotes.

Rules:

- Prefer tracked scripts or existing CLI entrypoints over long inline SSH
  snippets. If a snippet is repeated twice, turn it into a script or a workflow
  entry instead of pasting it again.
- Do not place `python - <<'PY' ... PY` inside an outer single-quoted SSH
  command. The inner `'PY'` can terminate or corrupt the outer quote.
- If a remote one-off Python snippet is unavoidable, avoid single quotes inside
  the snippet when the outer shell command uses single quotes. Use double-quoted
  Python dictionary keys such as `row["metric"]`, not `row['metric']`.
- Do not put quoted dictionary lookups directly inside f-string braces in
  remote snippets, for example `f"{row['metric']}"`. Extract first, then print:
  `metric = row["metric"]`; `print(f"{metric}")`.
- When a remote JSON inspection command fails because of quoting, rerun a small
  schema read first: print top-level keys, object types, and one representative
  item before doing the full summary.
- For complex remote Python, prefer one of these safer patterns:
  - run a tracked script already committed to the branch;
  - run `python -c` with a short expression only;
  - write a temporary script under an ignored remote `output/` or `/tmp/`
    location and execute it there, then report that it was temporary.
- Quote local paths containing spaces, Chinese characters, or `+`, for example:
  `'/Users/xuyihua/Desktop/学习相关/myCode/3D IC Heat'`.
- After `conda activate rigno` on remote machines, use `python`; do not assume
  `python3` exists in the activated environment.
- For long-running download or scan snippets, use unbuffered output or
  `flush=True` so progress is visible and a stalled command is distinguishable
  from buffered stdout.
- Avoid mixing local redirection with SSH output unless local artifact creation
  is the goal. Prefer remote-side output files under ignored `output/` or print
  compact summaries to stdout.
- For cross-machine sync pipelines, state the source and destination first, use
  non-overwriting behavior when required, and verify counts after the copy.
- If a pipeline, redirection, or tar-over-ssh command fails because of local
  sandboxing or shell interpretation, do not rewrite the data flow blindly.
  Report the failure mode and rerun the same intended operation with the proper
  permission path or a simpler remote-only command.
- After any command that previously failed due to quoting, rerun a minimal
  structure print first, then run the full extraction. Do not trust partial
  output from a failed quote context.

Known V3 failure patterns that this workflow prevents:

- remote heredoc truncated by shell single quotes;
- Python dict keys like `row['x']` breaking an outer single-quoted SSH command;
- f-string expressions with quoted dictionary keys being corrupted by the
  surrounding shell quote context;
- local `>` / `|` / `tee` around SSH producing sandbox failures or unexpected
  local files;
- macOS tar metadata creating `._*` files on Linux and corrupting file-count
  checks;
- assuming remote `python3` exists before activating `rigno`.

## WF-CONFIG: Config Preparation

Use this when preparing a new YAML or V4 registry entry.

Required fields for every experiment:

- baseline run or checkpoint
- one changed variable group only
- dataset and split map
- graph policy and repair policy
- optimizer, schedule, epochs, seeds, and batch plan
- output directory and log path
- checkpoint selection metric
- prediction, diagnostics, and final-probe export policy
- expected comparison set
- no-write zones

Rules:

- Prefer a tracked registry entry over a one-off YAML name.
- Do not hand-copy a prior YAML without stating the exact semantic difference.
- Run config load and command dry-run before launching.
- Test that the output directory does not already exist.
- Long runs must save `params_best.pkl`, `params_final.pkl`, best/final
  predictions, `loss_summary.json`, and `run_config.json` unless explicitly
  declared as smoke runs.
- Smoke configs must explicitly disable prediction export, post-training
  diagnostics, and final-probe inference unless the user asks for them.

## WF-LAUNCH: Remote Training Launch And Monitoring

Use this only after explicit user approval to start training.

Prefer `$heat3d-v4-remote-run` and the tracked helper:

```bash
python3 -B scripts/heat3d_v4_remote_run.py --host devbox check
python3 -B scripts/heat3d_v4_remote_run.py --host devbox launch --config-id <config_id>
python3 -B scripts/heat3d_v4_remote_run.py --host devbox monitor --config-id <config_id>
```

Rules:

- Do not start training from Mac local unless the user explicitly asks.
- Commit and push local config/script changes before launch; the helper performs
  remote `git fetch`, checkout, and `git pull --ff-only`.
- Prefer one active long run per machine unless the user approves otherwise.
- If a run fails, capture the failing log tail and stop; do not silently launch
  a replacement run.

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
- Keep P10 caveated until localized top contact and side asymmetry are actually
  represented in the generator/schema.
- State whether the evaluation is diagnostic, benchmark-candidate, or formal
  benchmark. V4 defaults are still diagnostic unless explicitly promoted.

## WF-SYNC: Remote Result Synchronization

Use this when moving ignored output artifacts between WSL2, devbox, and local
review folders.

Rules:

- Synchronize ignored artifacts only; do not commit them.
- Prefer non-overwriting sync when copying between remote machines.
- Record source machine, destination machine, source path, destination path,
  file counts, checkpoint counts, and whether existing files were skipped.
- After macOS-to-Linux tar copies, remove AppleDouble `._*` metadata files if
  they appear.
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

## Short Prompt Templates

These short prompts are preferred over long repeated instructions.

```text
按 WF-STATUS 检查 research/v4 工作树，只读汇报。
```

```text
按 WF-CONFIG 为 registry 中的 <run_id> 生成 YAML 并 dry-run，不启动训练。
```

```text
按 WF-REMOTE-ENV 和 WF-LAUNCH 在 devbox 启动 <run_id>，训练前检查 output collision。
```

```text
按 WF-EVAL 汇总 <run_id> 的 best/final 指标和 final-probe 结果，不训练。
```

```text
按 WF-SYNC 把 wsl2 的 <run_id> 输出同步到 devbox，已有文件不覆盖。
```

```text
按 WF-REPORT 收尾，确认 AGENTS.md 是否已提交，并确认没有提交 data/output/checkpoint/log。
```
