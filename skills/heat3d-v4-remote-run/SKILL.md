---
name: heat3d-v4-remote-run
description: "Use when launching a registered Heat3D V4 YAML config on a remote machine: get the local config ID, ensure the branch is committed and pushed, have the remote pull GitHub, start the run in tmux through the V4 remote helper, and report live log monitor commands. Requires explicit user approval to train or use SSH."
---

# Heat3D V4 Remote Run

## Contract

Read `AGENTS.md`, workflow Quick Contract, `WF-REMOTE-ENV`, `WF-CONFIG`,
`WF-LAUNCH`, and `WF-REPORT`. Do not train or SSH unless the current user turn
explicitly approves it.

## Workflow

1. Identify the `config_id` from the user request or
   `configs/heat3d_v4/run_registry.csv`.
2. Verify locally:

```bash
python3 -B scripts/prepare_heat3d_v4_run.py --config-id <config_id> --write-csv-mirror --write-yaml --dry-run
python3 -B scripts/check_heat3d_v4_registry.py
git status --short
```

3. Commit and push local file changes before launch; remotes pull from GitHub.
4. Run remote preflight with the helper after `rigno` is active. The preflight
   checks repository/environment state only; it does not probe accelerator
   availability.

```bash
python3 -B scripts/heat3d_v4_remote_run.py --host devbox check
```

5. Launch with tmux through the tracked helper:

```bash
python3 -B scripts/heat3d_v4_remote_run.py --host devbox launch --config-id <config_id>
```

6. Return monitor commands every time:

```bash
python3 -B scripts/heat3d_v4_remote_run.py --host devbox monitor --config-id <config_id>
```

7. For devbox/WSL2 artifact mirroring, print the server-to-server sync command:

```bash
python3 -B scripts/heat3d_v4_remote_run.py --host devbox sync-command --config-id <config_id> --target-host wsl2
```

## Rules

- The helper performs remote `git fetch`, checkout, and `git pull --ff-only`.
- `sync-command` syncs the complete `output_dir` between servers only; it must
  not copy outputs to the Codex host.
- Default branch is `research/v4`; change it only when the user names another
  branch.
- Default remote is `devbox`, default conda env is `rigno`, default session is
  `v4_<config_id>`.
- If a run fails, report the log tail and stop; do not silently relaunch.
