"""Smoke-check compact Heat3D v2 group-build progress output."""

from __future__ import annotations

import io
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


def main() -> int:
    _check_progress_detail_modes()
    _check_basic_non_tty_is_compact()
    _check_edge_totals()
    print("Heat3D v2 group-build progress smoke passed.")
    return 0


def _check_progress_detail_modes() -> None:
    expected = {
        "off": "off",
        "quiet": "off",
        "basic": "basic",
        "verbose": "full",
        "full": "full",
    }
    for value, mode in expected.items():
        actual = runner._progress_detail_mode(value)
        if actual != mode:
            raise AssertionError(f"progress_detail={value!r}: expected {mode!r}, got {actual!r}")


def _check_basic_non_tty_is_compact() -> None:
    stream = io.StringIO()
    bar = runner._ProgressBar(
        True,
        "[startup] group build all",
        200,
        min_interval_s=999.0,
        stream=stream,
    )
    for index in range(1, 201):
        bar.update(index)
    bar.close(current=200)
    output = stream.getvalue()
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) >= 50:
        raise AssertionError(f"basic progress should be compact for 200 groups, got {len(lines)} lines")
    if "arrays+graph start" in output or "arrays+graph built" in output:
        raise AssertionError("basic progress must not emit per-group start/built lines")
    if "completed groups=200" not in output:
        raise AssertionError("basic progress must emit a completed summary")


def _check_edge_totals() -> None:
    for total in (0, 1, 192):
        stream = io.StringIO()
        bar = runner._ProgressBar(
            True,
            f"[startup] group build smoke{total}",
            total,
            min_interval_s=999.0,
            stream=stream,
        )
        for index in range(1, total + 1):
            bar.update(index)
        bar.close(current=total)
        output = stream.getvalue()
        if f"completed groups={total}" not in output:
            raise AssertionError(f"missing completed summary for total={total}")


if __name__ == "__main__":
    raise SystemExit(main())
