"""Read-only review helper for Heat3D v2 M1 lower-lr ablation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNS = (
    ("lr=1e-3", "m1_batch_e50_seed0"),
    ("lr=3e-4", "m1_batch_e50_lr3e4_seed0"),
    ("lr=1e-4", "m1_batch_e50_lr1e4_seed0"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read existing Heat3D v2 M1 lower-lr output JSON files and print a "
            "compact comparison. This script never trains or writes output."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/heat3d_v2_runs"),
        help="Directory containing m1_batch_e50_* run output folders.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report missing output directories/files without failing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for label, run_name in RUNS:
        run_dir = args.root / run_name
        summary_path = run_dir / "loss_summary.json"
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        summary = _read_json(summary_path)
        epoch_history = summary.get("epoch_history") or []
        if not epoch_history:
            raise ValueError(f"{summary_path}: missing epoch_history")
        best_epoch = int(summary["best_epoch"])
        best_row = epoch_history[best_epoch - 1]
        recomputed = min(epoch_history, key=lambda row: float(row["valid_loss"]))
        rows.append(
            {
                "label": label,
                "run_name": run_name,
                "lr": float(summary["lr"]),
                "best_epoch": best_epoch,
                "recomputed_best_epoch": int(recomputed["epoch"]),
                "selection_ok": int(recomputed["epoch"]) == best_epoch,
                "best_valid_loss": float(summary["best_valid_loss"]),
                "best_valid_raw_deltaT_mse": float(summary["best_valid_raw_deltaT_mse"]),
                "best_hotspot_mae": float(best_row["valid_hotspot_raw_mae"]),
                "best_bg_bias": float(best_row["valid_bg_signed_bias"]),
                "best_pn_over_ratio": float(best_row["valid_pn_over_ratio"]),
                "final_valid_loss": float(summary["final_valid_loss"]),
                "final_best_ratio": float(summary["final_valid_loss"])
                / float(summary["best_valid_loss"]),
                "status_ok": bool(summary.get("status_ok")),
            }
        )

    if missing and not args.allow_missing:
        for path in missing:
            print(f"missing: {path}")
        return 1

    if rows:
        print(
            "lr | best_epoch | best_valid_loss | best_valid_raw_deltaT_mse | "
            "best_hotspot_mae | final_valid_loss | final/best | selection_ok | status"
        )
        for row in rows:
            print(
                f"{row['lr']:.1e} | {row['best_epoch']} | "
                f"{row['best_valid_loss']:.8e} | "
                f"{row['best_valid_raw_deltaT_mse']:.8e} | "
                f"{row['best_hotspot_mae']:.8e} | "
                f"{row['final_valid_loss']:.8e} | "
                f"{row['final_best_ratio']:.3f} | "
                f"{row['selection_ok']} | {row['status_ok']}"
            )

    if missing:
        print(f"missing_count: {len(missing)}")
    print("Heat3D v2 lower-lr results review passed.")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{path}: failed to read JSON") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
