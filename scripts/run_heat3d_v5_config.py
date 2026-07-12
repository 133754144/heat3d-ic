#!/usr/bin/env python3
"""Resolve a tracked V5 ablation YAML into a guarded runnable command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v5_config import V5ConfigError, build_v5_runner_plan  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Tracked V5 YAML plan path.")
    parser.add_argument("--variant", required=True, help="Named V5 ablation_matrix entry.")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute only when the selected V5 config explicitly sets training_allowed: true.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.dry_run and args.execute:
        raise SystemExit("--dry-run and --execute are mutually exclusive")
    config = Path(args.config)
    if not config.is_absolute():
        config = REPO_ROOT / config
    try:
        plan = build_v5_runner_plan(
            config,
            variant=args.variant,
            python_executable=args.python_executable,
        )
    except V5ConfigError as exc:
        print(f"V5 config runner error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": plan["schema_version"],
                "variant": plan["variant"],
                "training_allowed": plan["training_allowed"],
                "guardrails": plan["guardrails"],
                "command": plan["command"],
                "command_shell": shlex.join(plan["command"]),
                "mode": "execute" if args.execute else "dry_run",
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not args.execute:
        return 0
    if not plan["training_allowed"]:
        print(
            "V5 config runner error: selected plan is prepare-only and forbids training",
            file=sys.stderr,
        )
        return 2
    return subprocess.call(plan["command"])


if __name__ == "__main__":
    raise SystemExit(main())
