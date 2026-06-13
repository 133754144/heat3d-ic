#!/usr/bin/env python3
"""Read-only Heat3D v3 loss trajectory audit from loss_summary.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MILESTONE_EPOCHS = (20, 50, 100, 200, 400, 600, 800, 1000, 1200)


def _parse_run(token: str) -> tuple[str, Path]:
    if "=" not in token:
        path = Path(token)
        return path.name, path
    label, path = token.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty run label in {token!r}")
    return label, Path(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _epoch_value(summary: dict[str, Any], key: str, epoch: int) -> float | None:
    values = summary.get(key)
    index = epoch - 1
    if not isinstance(values, list) or index < 0 or index >= len(values):
        return None
    return _to_float(values[index])


def _fmt(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.6g}"


def _late_regression(summary: dict[str, Any]) -> dict[str, Any]:
    best = _to_float(summary.get("best_valid_iid_loss"))
    final = _to_float(summary.get("final_valid_iid_loss"))
    best_epoch = summary.get("best_epoch")
    final_epoch = summary.get("final_epoch")
    if best is None or final is None:
        return {
            "available": False,
            "reason": "missing best/final valid_iid loss",
        }
    return {
        "available": True,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "best_valid_iid_loss": best,
        "final_valid_iid_loss": final,
        "absolute_regression": final - best,
        "relative_regression": (final - best) / abs(best) if best != 0 else None,
        "final_best_ratio": summary.get("final_best_ratio"),
    }


def _summarize_run(label: str, run_dir: Path) -> dict[str, Any]:
    path = run_dir / "loss_summary.json"
    if not path.is_file():
        return {
            "label": label,
            "run_dir": str(run_dir),
            "status": "missing_loss_summary",
        }
    summary = _read_json(path)
    rows = []
    for epoch in MILESTONE_EPOCHS:
        rows.append(
            {
                "epoch": epoch,
                "lr": _epoch_value(summary, "epoch_lrs", epoch),
                "valid_iid_loss": _epoch_value(summary, "valid_iid_losses", epoch),
                "valid_stress_loss": _epoch_value(summary, "valid_stress_losses", epoch),
            }
        )
    return {
        "label": label,
        "run_dir": str(run_dir),
        "status": "complete",
        "lr_schedule": summary.get("lr_schedule"),
        "final_epoch": summary.get("final_epoch"),
        "best_epoch": summary.get("best_epoch"),
        "final_best_ratio": summary.get("final_best_ratio"),
        "best_valid_iid_loss": summary.get("best_valid_iid_loss"),
        "final_valid_iid_loss": summary.get("final_valid_iid_loss"),
        "best_valid_stress_loss": summary.get("best_valid_stress_loss"),
        "final_valid_stress_loss": summary.get("final_valid_stress_loss"),
        "late_regression": _late_regression(summary),
        "milestones": rows,
    }


def build_payload(runs: list[tuple[str, Path]]) -> dict[str, Any]:
    results = [_summarize_run(label, path) for label, path in runs]
    return {
        "diagnostic_scope": "read-only checkpoint trajectory audit from loss_summary.json",
        "milestone_epochs": list(MILESTONE_EPOCHS),
        "runs": results,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Heat3D v3 Checkpoint Trajectory Audit",
        "",
        "Read-only summary from existing `loss_summary.json` files.",
        "",
        "## Summary",
        "",
        "| run | status | schedule | best_epoch | final_epoch | best iid | final iid | best stress | final stress | final/best | late regression |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in payload["runs"]:
        late = run.get("late_regression", {})
        lines.append(
            "| {label} | {status} | {schedule} | {best_epoch} | {final_epoch} | {best_iid} | {final_iid} | {best_stress} | {final_stress} | {ratio} | {late} |".format(
                label=run["label"],
                status=run["status"],
                schedule=run.get("lr_schedule", "-"),
                best_epoch=run.get("best_epoch", "-"),
                final_epoch=run.get("final_epoch", "-"),
                best_iid=_fmt(run.get("best_valid_iid_loss")),
                final_iid=_fmt(run.get("final_valid_iid_loss")),
                best_stress=_fmt(run.get("best_valid_stress_loss")),
                final_stress=_fmt(run.get("final_valid_stress_loss")),
                ratio=_fmt(run.get("final_best_ratio")),
                late=_fmt(late.get("relative_regression")),
            )
        )
    lines.extend(
        [
            "",
            "## Milestones",
            "",
            "| run | epoch | lr | valid_iid | valid_stress |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in payload["runs"]:
        for row in run.get("milestones", []):
            lines.append(
                f"| {run['label']} | {row['epoch']} | {_fmt(row.get('lr'))} | "
                f"{_fmt(row.get('valid_iid_loss'))} | {_fmt(row.get('valid_stress_loss'))} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Run entry as LABEL=RUN_DIR.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload([_parse_run(token) for token in args.run])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    for run in payload["runs"]:
        print(
            "{label}: status={status} best_epoch={best_epoch} final_best={ratio}".format(
                label=run["label"],
                status=run["status"],
                best_epoch=run.get("best_epoch", "-"),
                ratio=_fmt(run.get("final_best_ratio")),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
