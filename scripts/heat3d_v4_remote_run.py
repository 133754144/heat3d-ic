#!/usr/bin/env python3
"""Reusable remote check, launch, monitor, and sync helpers for Heat3D V4."""

from __future__ import annotations

import argparse
import posixpath
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_heat3d_v4_registry import (  # noqa: E402
    DEFAULT_REGISTRY,
    check_registry,
    load_registry,
    registry_rows,
)


DEFAULT_HOST = "devbox"
DEFAULT_BRANCH = "research/v4"
DEFAULT_REMOTE_REPO = "~/myCodeGitOnly/heat3d-ic"
DEFAULT_CONDA_SH = "~/miniconda3/etc/profile.d/conda.sh"
DEFAULT_CONDA_ENV = "rigno"


def main() -> int:
    args = _parse_args()
    if args.command == "check":
        return _run_or_print(args, _remote_check_script(args))
    if args.command == "launch":
        row = _row_for_config(args.config_id, args.registry)
        return _run_or_print(args, _remote_launch_script(args, row))
    if args.command == "monitor":
        row = _row_for_config(args.config_id, args.registry)
        print(_monitor_commands(args, row))
        return 0
    if args.command == "sync-command":
        row = _row_for_config(args.config_id, args.registry)
        print(_sync_command(args, row))
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--conda-sh", default=DEFAULT_CONDA_SH)
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Run WF-REMOTE-ENV checks.")

    launch = subparsers.add_parser("launch", help="Launch a config in tmux.")
    launch.add_argument("--config-id", required=True)
    launch.add_argument("--session")
    launch.add_argument("--allow-existing-output", action="store_true")

    monitor = subparsers.add_parser("monitor", help="Print monitor commands.")
    monitor.add_argument("--config-id", required=True)

    sync = subparsers.add_parser(
        "sync-command",
        help="Print server-to-server rsync command for a run output_dir.",
    )
    sync.add_argument("--config-id", required=True)
    sync.add_argument(
        "--target-host",
        default="wsl2",
        help="Destination server. Data stays on remote servers, not local host.",
    )
    sync.add_argument(
        "--target-repo",
        default=None,
        help="Destination repo path. Defaults to --remote-repo.",
    )
    return parser.parse_args()


def _row_for_config(config_id: str, registry_path: str) -> dict[str, str]:
    registry = load_registry(_repo_path(registry_path))
    rows = registry_rows(registry)
    check_registry(_repo_path(registry_path), emit_warnings=False)
    for row in rows:
        if row["config_id"] == config_id:
            return row
    raise SystemExit(f"missing config_id: {config_id}")


def _remote_check_script(args: argparse.Namespace) -> str:
    repo = _remote_path(args.remote_repo)
    conda_sh = _remote_path(args.conda_sh)
    env = _q(args.conda_env)
    return "\n".join(
        [
            "set -e",
            f"cd {repo}",
            "git branch --show-current",
            "git rev-parse --short HEAD",
            "git status --short",
            f"source {conda_sh}",
            f"conda activate {env}",
            "python --version",
            "df -h .",
            "pgrep -af \"python|train_heat3d|heat3d_v\" || true",
            "tmux ls || true",
        ]
    )


def _remote_launch_script(args: argparse.Namespace, row: dict[str, str]) -> str:
    repo = _remote_path(args.remote_repo)
    conda_sh = _remote_path(args.conda_sh)
    env = _q(args.conda_env)
    branch = _q(args.branch)
    config = _q(row["generated_yaml"])
    out_dir = _q(row["output_dir"])
    log_path = _q(row["log_path"])
    session = _q(args.session or _tmux_session(row["config_id"]))
    out_check = ":" if args.allow_existing_output else f"test ! -e {out_dir}"
    inner = " && ".join(
        [
            f"cd {repo}",
            f"source {conda_sh}",
            f"conda activate {env}",
            "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
            f"python -u scripts/run_heat3d_v4_config.py --config {config}",
        ]
    )
    tmux_command = f"{inner} > {row['log_path']} 2>&1"
    return "\n".join(
        [
            "set -e",
            f"cd {repo}",
            "git fetch origin",
            f"git checkout {branch}",
            f"git pull --ff-only origin {branch}",
            f"source {conda_sh}",
            f"conda activate {env}",
            f"test -f {config}",
            out_check,
            f"mkdir -p \"$(dirname {log_path})\"",
            f"tmux new-session -d -s {session} {_q(tmux_command)}",
            f"echo session={shlex.quote(args.session or _tmux_session(row['config_id']))}",
            f"echo log={log_path}",
            f"echo monitor='tail -f {row['log_path']}'",
        ]
    )


def _monitor_commands(args: argparse.Namespace, row: dict[str, str]) -> str:
    log_path = row["log_path"]
    session = args.command and _tmux_session(row["config_id"])
    return "\n".join(
        [
            f"ssh {args.host}",
            f"cd {args.remote_repo}",
            f"tail -f {log_path}",
            (
                "grep -E "
                "'epoch|valid_base|stress_base|best|checkpoint|final probe|"
                f"diagnostics|OOM|RESOURCE_EXHAUSTED' {log_path} | tail -n 100"
            ),
            f"tmux attach -t {session}",
        ]
    )


def _sync_command(args: argparse.Namespace, row: dict[str, str]) -> str:
    output_dir = row["output_dir"].strip("/")
    parent_dir = posixpath.dirname(output_dir)
    target_repo = args.target_repo or args.remote_repo
    mkdir_target_parent = shlex.join(
        [
            "ssh",
            args.target_host,
            "bash",
            "-lc",
            f"mkdir -p {_remote_path(posixpath.join(target_repo, parent_dir))}",
        ]
    )
    rsync_target = f"{args.target_host}:{target_repo.rstrip('/')}/{output_dir}/"
    remote_script = "\n".join(
        [
            "set -e",
            f"cd {_remote_path(args.remote_repo)}",
            f"test -d {_q(output_dir)}",
            mkdir_target_parent,
            " ".join(
                [
                    "rsync",
                    "-av",
                    "--ignore-existing",
                    f"{_q(output_dir)}/",
                    _q(rsync_target),
                ]
            ),
        ]
    )
    return "\n".join(
        [
            "# Server-to-server sync only; does not copy output_dir to this Codex host.",
            _ssh_command_text(args.host, remote_script),
        ]
    )


def _run_or_print(args: argparse.Namespace, script: str) -> int:
    if args.print_only:
        print(_ssh_command_text(args.host, script))
        return 0
    return subprocess.call(["ssh", args.host, "bash", "-lc", shlex.quote(script)])


def _ssh_command_text(host: str, script: str) -> str:
    return shlex.join(["ssh", host, "bash", "-lc", script])


def _tmux_session(config_id: str) -> str:
    session = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"v4_{config_id}")
    return session[:80]


def _repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _q(value: Any) -> str:
    return shlex.quote(str(value))


def _remote_path(value: str) -> str:
    if value.startswith("~/") or value.startswith("$HOME/"):
        return value
    return shlex.quote(value)


if __name__ == "__main__":
    raise SystemExit(main())
