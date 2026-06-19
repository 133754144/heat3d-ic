#!/usr/bin/env python3
"""Run one tracked Heat3D V4 YAML config through the v2 training runner."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from rigno.heat3d_v2_config import load_v2_config, validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


def main() -> int:
    args = _parse_args()
    config_path = _repo_path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"config not found: {args.config}")
    command = build_training_command(
        _load_config(config_path),
        python_executable=args.python_executable,
    )
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Tracked YAML config path.")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if isinstance(payload, dict) and payload.get("schema_version") == (
        "heat3d_v4_inherited_config_v0"
    ):
        resolved = resolve_inherited_yaml(payload, config_path)
        validate_v2_config(resolved, config_path=config_path)
        return resolved
    return load_v2_config(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
