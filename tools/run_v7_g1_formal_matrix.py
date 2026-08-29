#!/usr/bin/env python3
"""Run the frozen V7 G1 matrix one registered seed at a time.

This orchestration layer owns process control and status receipts only.  Each
numerical run is delegated to the manifest-bound V7 P1i entrypoint; no model,
graph, support, loss, batching or reconstruction logic is duplicated here.
The output root must be outside the repository so a formal run cannot write a
frozen artifact path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "heat3d_v7" / "v7_g1_formal_launch_manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def _run_command(manifest_path: Path, run: dict[str, Any], output_dir: Path, *, dry_run: bool) -> list[str]:
    config = ROOT / "configs" / "heat3d_v7" / "v7_g1_full_p1i.json"
    subset = ROOT / "data" / "heat3d_v6_p1i_continuous_physics1024_v1"
    dataset_manifest = ROOT / "configs" / "heat3d_v6_p1i" / "v6_p1i_formal1024_v1_manifest.json"
    command = [
        sys.executable,
        "-m",
        "scripts.run_heat3d_v7_formal_p1i_training",
        "--experiment-id",
        str(run["experiment_id"]),
        "--config",
        str(config),
        "--subset",
        str(subset),
        "--manifest",
        str(dataset_manifest),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "200",
        "--seed",
        str(run["seed"]),
        "--launch-manifest",
        str(manifest_path),
        "--jit-cache",
    ]
    command.append("--formal")
    if dry_run:
        command.append("--dry-run")
    return command


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != "heat3d_v7_g1_formal_launch_manifest_v1":
        raise ValueError("formal launch manifest schema drifted")
    if manifest.get("status") != "frozen_launch_manifest":
        raise ValueError("formal launch manifest is not frozen")
    matrix = manifest.get("matrix", {})
    if matrix.get("formal_execution_started") is not False:
        raise ValueError("formal launch manifest is already marked started")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 21:
        raise ValueError("formal launch manifest must contain exactly 21 runs")
    run_ids = [str(row.get("run_id")) for row in runs]
    if len(set(run_ids)) != 21 or any(not value or value == "None" for value in run_ids):
        raise ValueError("formal run IDs must be unique")
    if manifest.get("test_iid_access") is not False or manifest.get("sealed_access") is not False:
        raise ValueError("formal launch manifest opened forbidden splits")
    return runs


def _receipt_complete(output_dir: Path) -> bool:
    receipt_path = output_dir / "v7_g1_formal_receipt.json"
    if not receipt_path.exists():
        return False
    try:
        receipt = _load(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        receipt.get("status") == "COMPLETE"
        and receipt.get("g1_formal") is True
        and receipt.get("publication_evidence") is True
        and receipt.get("scientific_evidence_eligible") is True
    )


def main() -> int:
    args = _args()
    manifest_path = args.manifest.resolve()
    manifest = _load(manifest_path)
    runs = _validate_manifest(manifest)
    output_root = Path(str(manifest["output_root"])).resolve()
    status_path = (args.status_path or output_root / "matrix_status.json").resolve()
    status = _load(status_path) if status_path.exists() else {
        "schema_version": "heat3d_v7_g1_formal_matrix_status_v1",
        "manifest": str(manifest_path),
        "manifest_code_sha": manifest.get("g1_formal_code_sha"),
        "status": "NOT_STARTED",
        "created_at": _now(),
        "runs": {},
    }
    status["status"] = "DRY_RUN" if args.dry_run_only else "RUNNING"
    status["updated_at"] = _now()
    _write(status_path, status)

    for run in runs:
        run_id = str(run["run_id"])
        output_dir = output_root / run_id
        row = status.setdefault("runs", {}).setdefault(run_id, {})
        row.update(
            {
                "experiment_id": run["experiment_id"],
                "variant": run["variant"],
                "seed": int(run["seed"]),
                "output_dir": str(output_dir),
                "progress_path": str(output_dir / "v7_g1_progress.json"),
            }
        )
        if args.dry_run_only:
            result = subprocess.run(
                _run_command(manifest_path, run, output_dir, dry_run=True),
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            row.update(
                {
                    "status": "DRY_RUN_PASS" if result.returncode == 0 else "DRY_RUN_FAIL",
                    "returncode": result.returncode,
                    "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-2000:],
                }
            )
            _write(status_path, status)
            if result.returncode != 0:
                status["status"] = "DRY_RUN_FAIL"
                _write(status_path, status)
                return result.returncode
            continue

        if not args.no_resume and _receipt_complete(output_dir):
            row.update({"status": "COMPLETE", "resumed": True, "finished_at": _now()})
            _write(status_path, status)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "formal_stdout.log"
        stderr_path = output_dir / "formal_stderr.log"
        started = _now()
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                _run_command(manifest_path, run, output_dir, dry_run=False),
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            row.update(
                {
                    "status": "RUNNING",
                    "pid": process.pid,
                    "started_at": started,
                    "resumed": False,
                }
            )
            _write(status_path, status)
            while process.poll() is None:
                time.sleep(max(1.0, float(args.poll_seconds)))
                progress_path = output_dir / "v7_g1_progress.json"
                if progress_path.exists():
                    try:
                        progress = _load(progress_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        progress = {}
                    for key in ("status", "epoch", "epochs", "best_epoch", "best_selection_metric", "latest_selection_metric"):
                        if key in progress:
                            row[key] = progress[key]
                row["updated_at"] = _now()
                _write(status_path, status)
            returncode = int(process.returncode or 0)
        row.update(
            {
                "status": "COMPLETE" if returncode == 0 and _receipt_complete(output_dir) else "FAILED",
                "returncode": returncode,
                "finished_at": _now(),
            }
        )
        _write(status_path, status)

    failures = [row for row in status.get("runs", {}).values() if row.get("status") == "FAILED"]
    status["status"] = "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES"
    status["finished_at"] = _now()
    status["failed_run_count"] = len(failures)
    _write(status_path, status)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
