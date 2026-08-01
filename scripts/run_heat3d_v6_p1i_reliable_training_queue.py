#!/usr/bin/env python3
"""Launch one or a serial P1i seed queue with valid-only post-run gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/heat3d_v6_p1i"
FULL_FIELDS = ROOT / "data/heat3d_v6_p1i_continuous_physics1024_v1_full_fields/full_fields.h5"
DATASET = ROOT / "data/heat3d_v6_p1i_continuous_physics1024_v1"
MANIFEST = CONFIG_DIR / "v6_p1i_formal1024_v1_manifest.json"
RUNS = {
    "seed0": ("V6_06_V5best_P1i_seed0_reliable_B24.yaml", "V6_06_V5best_P1i_seed0_reliable_B24"),
    "seed1": ("V6_07_V5best_P1i_seed1_reliable_B24.yaml", "V6_07_V5best_P1i_seed1_reliable_B24"),
    "seed2": ("V6_08_V5best_P1i_seed2_reliable_B24.yaml", "V6_08_V5best_P1i_seed2_reliable_B24"),
}


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _run_seed(label: str) -> None:
    config_name, run_name = RUNS[label]
    output = ROOT / "output/heat3d_v6_p1i_runs" / run_name
    subprocess.run(
        [sys.executable, "scripts/run_heat3d_v4_config.py", "--config", str(CONFIG_DIR / config_name)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/check_heat3d_v6_p1i_training_closeout.py", "--output-dir", str(output)],
        cwd=ROOT,
        check=True,
    )
    required = (
        "params_best.pkl", "params_final.pkl", "params_latest.pkl",
        "best_predictions.npz", "predictions.npz", "run_config.json",
        "loss_summary.json", "environment.json", "resolved_config_pretraining.yaml",
        "resolved_command.txt", "pretraining_provenance.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"{run_name}: missing formal artifacts: {missing}")
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_heat3d_v6_p1i_valid_full_field.py",
            "--dataset-root", str(DATASET),
            "--manifest", str(MANIFEST),
            "--full-fields", str(FULL_FIELDS),
            "--predictions", f"point_global_best={output / 'best_predictions.npz'}",
            "--predictions", f"final={output / 'predictions.npz'}",
            "--output-json", str(output / "valid_full_field.json"),
            "--output-csv", str(output / "valid_full_field.csv"),
            "--output-md", str(output / "valid_full_field.md"),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", nargs="+", choices=tuple(RUNS))
    parser.add_argument("--status-json", type=Path, required=True)
    args = parser.parse_args()
    status = {
        "schema_version": "heat3d_v6_p1i_reliable_queue_v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "seeds": args.seeds,
        "completed": [],
        "current": None,
        "status": "running",
        "accessed_roles": ["train", "valid_iid"],
        "test_accessed": False,
        "sealed_accessed": False,
    }
    _write_status(args.status_json, status)
    try:
        for label in args.seeds:
            status["current"] = label
            _write_status(args.status_json, status)
            _run_seed(label)
            status["completed"].append(label)
            status["current"] = None
            _write_status(args.status_json, status)
    except Exception as error:
        status["status"] = "failed_stopped_before_next_seed"
        status["error"] = f"{type(error).__name__}: {error}"
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_status(args.status_json, status)
        raise
    status["status"] = "completed"
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_status(args.status_json, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
