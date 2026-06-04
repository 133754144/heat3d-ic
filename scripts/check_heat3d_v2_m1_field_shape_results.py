"""Read-only compact review of Heat3D v2 M1 field-shape diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNS = (
    ("m1_batch_e50_seed0", "1e-3"),
    ("m1_batch_e50_lr3e4_seed0", "3e-4"),
    ("m1_batch_e50_lr1e4_seed0", "1e-4"),
)
FIELDS = (
    "field_variance_ratio",
    "centered_spatial_correlation",
    "amplitude_ratio",
    "peak_abs_error",
    "top_k_overlap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read existing Heat3D v2 M1 field-shape diagnostics JSON files "
            "and print compact markdown tables. This script never trains or writes output."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/heat3d_v2_runs"),
        help="Directory containing M1 run output folders.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Print warnings for missing files without failing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing: list[Path] = []
    basics: list[dict[str, Any]] = []
    field_rows: dict[str, list[dict[str, Any]]] = {"best": [], "final": []}
    for run_name, lr in RUNS:
        run_dir = args.root / run_name
        summary_path = run_dir / "loss_summary.json"
        if not summary_path.exists():
            missing.append(summary_path)
            continue
        summary = _read_json(summary_path)
        best_epoch = int(summary["best_epoch"])
        best_row = summary["epoch_history"][best_epoch - 1]
        basics.append(
            {
                "run_name": run_name,
                "lr": lr,
                "best_epoch": best_epoch,
                "best_valid_loss": float(summary["best_valid_loss"]),
                "best_valid_raw_deltaT_mse": float(summary["best_valid_raw_deltaT_mse"]),
                "best_valid_hotspot_mae": float(best_row["valid_hotspot_raw_mae"]),
                "final_valid_loss": float(summary["final_valid_loss"]),
                "final_degradation": float(summary["final_valid_loss"])
                - float(summary["best_valid_loss"]),
            }
        )
        for label in ("best", "final"):
            diag_path = run_dir / f"field_shape_diagnostics_{label}.json"
            if not diag_path.exists():
                missing.append(diag_path)
                continue
            diag = _read_json(diag_path)
            overall = diag.get("overall")
            if not isinstance(overall, dict):
                raise ValueError(f"{diag_path}: missing overall metrics object")
            field_rows[label].append(
                {
                    "run_name": run_name,
                    "lr": lr,
                    "best_epoch": best_epoch,
                    **{field: float(overall[field]) for field in FIELDS},
                }
            )

    for path in missing:
        print(f"warning: missing {path}")
    if missing and not args.allow_missing:
        return 1

    if basics:
        _print_basic_table(basics)
    for label in ("best", "final"):
        if field_rows[label]:
            _print_field_table(label, field_rows[label])
    print("Heat3D v2 M1 field-shape results review passed.")
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


def _print_basic_table(rows: list[dict[str, Any]]) -> None:
    print("basic:")
    print("| run_name | lr | best_epoch | best_valid_loss | best_raw_deltaT_mse | best_hotspot_mae | final_valid_loss | final_degradation |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['run_name']} | {row['lr']} | {row['best_epoch']} | "
            f"{row['best_valid_loss']:.6e} | {row['best_valid_raw_deltaT_mse']:.6e} | "
            f"{row['best_valid_hotspot_mae']:.6e} | {row['final_valid_loss']:.6e} | "
            f"{row['final_degradation']:.6e} |"
        )


def _print_field_table(label: str, rows: list[dict[str, Any]]) -> None:
    print(f"{label}:")
    print("| run_name | lr | best_epoch | field_variance_ratio | centered_spatial_correlation | amplitude_ratio | peak_abs_error | top_k_overlap |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['run_name']} | {row['lr']} | {row['best_epoch']} | "
            f"{row['field_variance_ratio']:.6e} | "
            f"{row['centered_spatial_correlation']:.6e} | "
            f"{row['amplitude_ratio']:.6e} | "
            f"{row['peak_abs_error']:.6e} | "
            f"{row['top_k_overlap']:.6e} |"
        )


if __name__ == "__main__":
    raise SystemExit(main())
