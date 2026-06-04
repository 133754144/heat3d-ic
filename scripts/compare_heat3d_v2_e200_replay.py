#!/usr/bin/env python3
"""Compare old Heat3D v2 e200, e200 replay, and e300 audit runs.

Read-only: this script only reads run_config.json, loss_summary.json, and
optional diagnostics files under output/heat3d_v2_runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


RUNS = {
    "old_e200": "m1_B192_base_mse_lr3e4_e200_stratified_seed0",
    "replay": "m1_B192_base_mse_lr3e4_e200_stratified_replay_seed0",
    "e300": "m1_B192_base_mse_lr3e4_e300_stratified_seed0",
}

CONFIG_FIELDS = (
    "batch_size",
    "epochs",
    "seed",
    "lr",
    "weight_decay",
    "gradient_clip_norm",
    "loss_mode",
)
HASH_FIELDS = (
    "code_version_or_git_commit",
    "train_group_sample_id_hash",
    "valid_iid_sample_id_hash",
    "valid_stress_sample_id_hash",
)
SUMMARY_FIELDS = (
    "best_epoch",
    "best_valid_iid_loss",
    "final_valid_iid_loss",
    "final_valid_stress_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only comparison for Heat3D v2 e200 replay audit."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("output/heat3d_v2_runs"),
        help="Root containing Heat3D v2 run output directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = {label: _load_run(args.run_root / dirname) for label, dirname in RUNS.items()}

    print("# Heat3D v2 e200 Replay Audit")
    print()
    _print_presence(runs)
    print()
    _print_config_table(runs)
    print()
    _print_hash_table(runs)
    print()
    _print_summary_table(runs)
    print()
    _print_diagnostics_table(runs)
    print()
    _print_conclusion(runs)
    return 0


def _load_run(path: Path) -> dict[str, Any]:
    run_config_path = path / "run_config.json"
    loss_summary_path = path / "loss_summary.json"
    run_config = _read_json(run_config_path)
    loss_summary = _read_json(loss_summary_path)
    return {
        "path": path,
        "exists": path.is_dir(),
        "run_config": run_config,
        "loss_summary": loss_summary,
        "diagnostics": _read_diagnostics(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_diagnostics(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    diag_root = path / "diagnostics" / "split_aware"
    for split in ("valid_iid", "valid_stress"):
        for checkpoint in ("best", "final"):
            payload = _read_json(diag_root / f"{split}_{checkpoint}.json")
            if payload:
                result[f"{split}_{checkpoint}"] = payload
    return result


def _value(run: dict[str, Any], field: str) -> Any:
    summary = run["loss_summary"]
    config = run["run_config"]
    if field in summary:
        return summary[field]
    return config.get(field)


def _epoch_value(run: dict[str, Any], epoch: int, field: str) -> Any:
    for record in run["loss_summary"].get("epoch_history", []):
        if record.get("epoch") == epoch:
            return record.get(field)
    return None


def _e300_epoch200_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_epoch": _epoch_value(run, 200, "best_epoch"),
        "best_valid_iid_loss": _epoch_value(run, 200, "best_valid_iid_loss"),
        "final_valid_iid_loss": _epoch_value(run, 200, "valid_iid_loss"),
        "final_valid_stress_loss": _epoch_value(run, 200, "valid_stress_loss"),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.6g}"


def _print_presence(runs: dict[str, dict[str, Any]]) -> None:
    print("## Presence")
    for label, run in runs.items():
        has_config = bool(run["run_config"])
        has_summary = bool(run["loss_summary"])
        has_diagnostics = bool(run["diagnostics"])
        print(
            f"- {label}: exists={run['exists']} "
            f"run_config={has_config} loss_summary={has_summary} diagnostics={has_diagnostics} "
            f"path={run['path']}"
        )
    if not runs["replay"]["loss_summary"]:
        print("- replay missing: run the e200 replay before judging baseline replacement.")


def _print_config_table(runs: dict[str, dict[str, Any]]) -> None:
    print("## Config Summary")
    print("| run | " + " | ".join(CONFIG_FIELDS) + " |")
    print("|---" + "|---" * len(CONFIG_FIELDS) + "|")
    for label, run in runs.items():
        print("| " + label + " | " + " | ".join(_fmt(_value(run, field)) for field in CONFIG_FIELDS) + " |")


def _print_hash_table(runs: dict[str, dict[str, Any]]) -> None:
    print("## Hash / Version Summary")
    print("| run | " + " | ".join(HASH_FIELDS) + " | order_hash_first5 |")
    print("|---" + "|---" * (len(HASH_FIELDS) + 1) + "|")
    for label, run in runs.items():
        order_hashes = _value(run, "epoch_train_batch_order_hashes") or []
        order_hash_first5 = ",".join(str(value) for value in order_hashes[:5]) if order_hashes else "NA"
        row = [_fmt(_value(run, field)) for field in HASH_FIELDS]
        row.append(order_hash_first5)
        print("| " + label + " | " + " | ".join(row) + " |")


def _print_summary_table(runs: dict[str, dict[str, Any]]) -> None:
    print("## Loss Summary")
    print("| run | " + " | ".join(SUMMARY_FIELDS) + " |")
    print("|---" + "|---" * len(SUMMARY_FIELDS) + "|")
    for label, run in runs.items():
        print("| " + label + " | " + " | ".join(_fmt(_value(run, field)) for field in SUMMARY_FIELDS) + " |")

    e300_epoch200 = _e300_epoch200_summary(runs["e300"])
    print(
        "| e300_epoch200 | "
        + " | ".join(_fmt(e300_epoch200[field]) for field in SUMMARY_FIELDS)
        + " |"
    )


def _print_diagnostics_table(runs: dict[str, dict[str, Any]]) -> None:
    print("## Optional Diagnostics")
    print("| run | split/checkpoint | raw_deltaT_mse | field_variance_ratio | spatial_corr | top_k_overlap |")
    print("|---|---|---:|---:|---:|---:|")
    for label, run in runs.items():
        for key in ("valid_iid_best", "valid_iid_final", "valid_stress_best", "valid_stress_final"):
            payload = run["diagnostics"].get(key)
            if not payload:
                continue
            overall = payload.get("overall", {})
            print(
                f"| {label} | {key} | {_fmt(overall.get('raw_deltaT_mse'))} | "
                f"{_fmt(overall.get('field_variance_ratio'))} | "
                f"{_fmt(overall.get('centered_spatial_correlation'))} | "
                f"{_fmt(overall.get('top_k_overlap'))} |"
            )


def _print_conclusion(runs: dict[str, dict[str, Any]]) -> None:
    print("## Replay Judgment")
    replay = runs["replay"]
    if not replay["loss_summary"]:
        print("- replay_closer_to: unknown; replay loss_summary.json is missing.")
        print("- baseline recommendation: keep old_e200 until replay exists and is compared.")
        return

    old_distance = _distance(replay["loss_summary"], runs["old_e200"]["loss_summary"])
    e300_epoch200 = _e300_epoch200_summary(runs["e300"])
    e300_distance = _distance(
        replay["loss_summary"],
        {
            "best_valid_iid_loss": e300_epoch200["best_valid_iid_loss"],
            "final_valid_iid_loss": e300_epoch200["final_valid_iid_loss"],
            "final_valid_stress_loss": e300_epoch200["final_valid_stress_loss"],
        },
    )
    if old_distance is None and e300_distance is None:
        closer = "unknown"
    elif e300_distance is None or (old_distance is not None and old_distance <= e300_distance):
        closer = "old_e200"
    else:
        closer = "e300_epoch200"
    print(f"- replay_distance_to_old_e200: {_fmt(old_distance)}")
    print(f"- replay_distance_to_e300_epoch200: {_fmt(e300_distance)}")
    print(f"- replay_closer_to: {closer}")
    if closer == "old_e200":
        print("- baseline recommendation: old_e200 remains reproducible; keep old_e200 unless replay diagnostics show a meaningful regression.")
    elif closer == "e300_epoch200":
        print("- baseline recommendation: update baseline only after reviewing replay diagnostics; old_e200 may reflect historical code/runtime state.")
    else:
        print("- baseline recommendation: insufficient data; do not change baseline yet.")


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    fields = ("best_valid_iid_loss", "final_valid_iid_loss", "final_valid_stress_loss")
    values = []
    for field in fields:
        left_value = _as_float(left.get(field))
        right_value = _as_float(right.get(field))
        if left_value is not None and right_value is not None:
            values.append(abs(left_value - right_value))
    if not values:
        return None
    return float(sum(values))


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


if __name__ == "__main__":
    raise SystemExit(main())
