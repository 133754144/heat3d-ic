"""Audit completed Heat3D v3 B88 seed-stability loss curves.

This script is intentionally offline: it only reads existing run
``loss_summary.json`` files and writes compact JSON/CSV/Markdown summaries.
It does not import JAX, build graphs, or start training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SELECTED_EPOCHS = (1, 2, 5, 10, 20, 50, 100, 200, 300, 350, 400)

DEFAULT_RUNS: tuple[tuple[str, str], ...] = (
    (
        "nearest_seed0",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_e400_model_seed0_batchbuild0_batchorder0_graphseed0",
    ),
    (
        "nearest_seed1",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_e400_model_seed1_batchbuild0_batchorder0_graphseed0",
    ),
    (
        "nearest_seed3_A1",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_A1_e400_model_seed3_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup10_minlr1e-6_wd1e-4",
    ),
    (
        "nearest_seed4_A2",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_A2_e400_model_seed4_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup10_minlr1e-6_wd1e-4",
    ),
    (
        "nearest_seed5_A3",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_A3_e400_model_seed5_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup10_minlr1e-6_wd1e-4",
    ),
    (
        "nearest_seed6_A4",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_A4_e400_model_seed6_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup10_minlr1e-6_wd1e-4",
    ),
    (
        "nearest_seed7_A5",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_A5_e400_model_seed7_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup10_minlr1e-6_wd1e-4",
    ),
    (
        "discrete_seed0",
        "latent96_s6_mlp2_B88_sample_shuffle_discrete_radius_e400_model_seed0_batchbuild0_batchorder0_graphseed0",
    ),
    (
        "discrete_seed1",
        "latent96_s6_mlp2_B88_sample_shuffle_discrete_radius_e400_model_seed1_batchbuild0_batchorder0_graphseed0",
    ),
    (
        "discrete_seed6_B4",
        "latent96_s6_mlp2_B88_sample_shuffle_discrete_radius_B4_e400_model_seed6_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup10_minlr1e-6_wd1e-4",
    ),
    (
        "C1_seed1_warmup50",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_C1_e400_model_seed1_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup50_minlr1e-6_wd1e-4",
    ),
    (
        "C2_seed1_warmup100",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_C2_e400_model_seed1_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup100_minlr1e-6_wd1e-4",
    ),
    (
        "C3_seed1_minlr1e-5",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_C3_e400_model_seed1_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup10_minlr1e-5_wd1e-4",
    ),
    (
        "C4_seed1_minlr3e-5",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_C4_e400_model_seed1_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup10_minlr3e-5_wd1e-4",
    ),
    (
        "D1_seed1_wd0",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_D1_e400_model_seed1_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup10_minlr1e-6_wd0",
    ),
    (
        "D2_seed1_wd1e-5",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_D2_e400_model_seed1_batchbuild0_batchorder0_graphseed0_adamw_lr3e-4_warmup10_minlr1e-6_wd1e-5",
    ),
    (
        "D3_seed1_adam",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_D3_e400_model_seed1_batchbuild0_batchorder0_graphseed0_adam_lr3e-4_warmup10_minlr1e-6_wd0",
    ),
    (
        "G1_seed1_graphseed1",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_G1_e400_model_seed1_batchbuild0_batchorder0_graphseed1_adamw_lr3e-4_warmup10_minlr1e-6_wd1e-4",
    ),
    (
        "G3_seed0_graphseed1",
        "latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_G3_e400_model_seed0_batchbuild0_batchorder0_graphseed1_adamw_lr3e-4_warmup10_minlr1e-6_wd1e-4",
    ),
)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _epoch_row(history: list[dict[str, Any]], epoch: int) -> dict[str, Any] | None:
    if epoch < 1 or epoch > len(history):
        return None
    row = history[epoch - 1]
    if not isinstance(row, dict):
        return None
    return row


def _selected_epoch_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    history = summary.get("epoch_history")
    if not isinstance(history, list):
        history = []

    rows: list[dict[str, Any]] = []
    for epoch in SELECTED_EPOCHS:
        row = _epoch_row(history, epoch)
        if row is None:
            rows.append(
                {
                    "epoch": epoch,
                    "available": False,
                    "train_loss": None,
                    "valid_iid_loss": None,
                    "valid_stress_loss": None,
                    "lr": None,
                }
            )
            continue
        rows.append(
            {
                "epoch": epoch,
                "available": True,
                "train_loss": _float_or_none(row.get("epoch_mean_train_batch_loss")),
                "valid_iid_loss": _float_or_none(row.get("valid_iid_loss")),
                "valid_stress_loss": _float_or_none(row.get("valid_stress_loss")),
                "lr": _float_or_none(row.get("lr")),
            }
        )
    return rows


def _value_at_epoch(rows: list[dict[str, Any]], epoch: int, field: str) -> float | None:
    for row in rows:
        if row["epoch"] == epoch:
            return _float_or_none(row.get(field))
    return None


def _failure_label(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    final_epoch = int(summary.get("final_epoch") or 0)
    best_epoch = int(summary.get("best_epoch") or 0)
    best = _float_or_none(summary.get("best_valid_iid_loss"))
    final = _float_or_none(summary.get("final_valid_iid_loss"))
    epoch20 = _value_at_epoch(rows, 20, "valid_iid_loss")
    epoch50 = _value_at_epoch(rows, 50, "valid_iid_loss")
    epoch100 = _value_at_epoch(rows, 100, "valid_iid_loss")

    if best is not None and best <= 0.1:
        if best_epoch >= max(int(0.95 * max(final_epoch, 1)), 1):
            return "late_undertrained"
        return "mid_plateau"
    if epoch20 is not None and epoch20 >= 0.75:
        return "early_bad"
    if (
        epoch20 is not None
        and epoch50 is not None
        and epoch20 < 0.75
        and epoch50 >= 0.6
    ):
        return "warmup_split"
    if epoch100 is not None and epoch100 >= 0.5:
        return "mid_plateau"
    if final is not None and best is not None and final > max(best * 1.5, best + 0.1):
        return "generalization_gap"
    if best_epoch >= max(int(0.95 * max(final_epoch, 1)), 1):
        return "late_undertrained"
    return "mid_plateau"


def _summarize_run(label: str, run_dir: Path) -> dict[str, Any]:
    loss_path = run_dir / "loss_summary.json"
    if not loss_path.exists():
        return {
            "label": label,
            "run_dir": str(run_dir),
            "status": "missing_loss_summary",
            "selected_epochs": [],
        }

    summary = json.loads(loss_path.read_text(encoding="utf-8"))
    rows = _selected_epoch_rows(summary)
    final_epoch = int(summary.get("final_epoch") or len(summary.get("epoch_history", [])))
    result = {
        "label": label,
        "run_dir": str(run_dir),
        "status": "complete" if bool(summary.get("status_ok", True)) and final_epoch >= 400 else "partial",
        "optimizer": summary.get("optimizer"),
        "lr_schedule": summary.get("lr_schedule"),
        "lr": summary.get("lr"),
        "min_lr": summary.get("min_lr"),
        "warmup_epochs": summary.get("warmup_epochs"),
        "model_seed": summary.get("model_seed"),
        "batch_order_seed": summary.get("batch_order_seed"),
        "graph_seed": summary.get("graph_seed"),
        "final_epoch": final_epoch,
        "best_epoch": summary.get("best_epoch"),
        "initial_valid_iid_loss": summary.get("initial_valid_iid_loss"),
        "final_valid_iid_loss": summary.get("final_valid_iid_loss"),
        "best_valid_iid_loss": summary.get("best_valid_iid_loss"),
        "initial_valid_stress_loss": summary.get("initial_valid_stress_loss"),
        "final_valid_stress_loss": summary.get("final_valid_stress_loss"),
        "best_valid_stress_loss": summary.get("best_valid_stress_loss"),
        "final_best_ratio": summary.get("final_best_ratio"),
        "failure_label": _failure_label(summary, rows),
        "selected_epochs": rows,
    }
    return result


def _write_csv(results: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "label",
        "status",
        "failure_label",
        "epoch",
        "train_loss",
        "valid_iid_loss",
        "valid_stress_loss",
        "lr",
        "best_epoch",
        "best_valid_iid_loss",
        "final_valid_iid_loss",
        "final_best_ratio",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for row in result.get("selected_epochs", []):
                writer.writerow(
                    {
                        "label": result.get("label"),
                        "status": result.get("status"),
                        "failure_label": result.get("failure_label"),
                        "epoch": row.get("epoch"),
                        "train_loss": row.get("train_loss"),
                        "valid_iid_loss": row.get("valid_iid_loss"),
                        "valid_stress_loss": row.get("valid_stress_loss"),
                        "lr": row.get("lr"),
                        "best_epoch": result.get("best_epoch"),
                        "best_valid_iid_loss": result.get("best_valid_iid_loss"),
                        "final_valid_iid_loss": result.get("final_valid_iid_loss"),
                        "final_best_ratio": result.get("final_best_ratio"),
                    }
                )


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}g}"


def _write_md(results: list[dict[str, Any]], md_path: Path) -> None:
    lines = [
        "# Heat3D v3 Seed Failure Loss-Curve Audit",
        "",
        "Offline audit of completed B88 sample_shuffle e400 loss summaries. No training is run.",
        "",
        "| run | label | best epoch | best iid | final iid | final/best | e20 iid | e100 iid | e400 iid |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        rows = result.get("selected_epochs", [])
        e20 = _value_at_epoch(rows, 20, "valid_iid_loss")
        e100 = _value_at_epoch(rows, 100, "valid_iid_loss")
        e400 = _value_at_epoch(rows, 400, "valid_iid_loss")
        lines.append(
            "| {label} | {failure} | {best_epoch} | {best} | {final} | {ratio} | {e20} | {e100} | {e400} |".format(
                label=result.get("label"),
                failure=result.get("failure_label", "n/a"),
                best_epoch=result.get("best_epoch", "n/a"),
                best=_fmt(result.get("best_valid_iid_loss")),
                final=_fmt(result.get("final_valid_iid_loss")),
                ratio=_fmt(result.get("final_best_ratio")),
                e20=_fmt(e20),
                e100=_fmt(e100),
                e400=_fmt(e400),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="output/heat3d_v2_runs")
    parser.add_argument("--output-dir", default="output/heat3d_v3_seed_loss_curve_audit")
    parser.add_argument("--output-md", default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write missing run entries instead of failing.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    missing = []
    for label, run_name in DEFAULT_RUNS:
        run_dir = runs_root / run_name
        result = _summarize_run(label, run_dir)
        if result["status"] == "missing_loss_summary":
            missing.append(str(run_dir))
        results.append(result)

    if missing and not args.allow_missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"missing required loss summaries:\n{missing_text}")

    json_path = output_dir / "seed_loss_curve_audit.json"
    csv_path = output_dir / "seed_loss_curve_selected_epochs.csv"
    md_path = Path(args.output_md) if args.output_md else output_dir / "seed_loss_curve_audit.md"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_csv(results, csv_path)
    _write_md(results, md_path)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    for result in results:
        print(
            result["label"],
            result["status"],
            result.get("failure_label"),
            "best_iid",
            _fmt(result.get("best_valid_iid_loss")),
            "best_epoch",
            result.get("best_epoch"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
