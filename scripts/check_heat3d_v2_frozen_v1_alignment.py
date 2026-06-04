#!/usr/bin/env python3
"""Check strict Heat3D v2 alignment with the historical frozen V1 best run."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_v2_command_plan  # noqa: E402


STRICT_CONFIG = Path("configs/heat3d_v2/frozen_v1_best_e050_seed0.yaml")
REFERENCE_CONFIG = Path("configs/heat3d_v2/frozen_v1_reference.yaml")
PREVIOUS_RUN_DIR = Path("output/heat3d_v2_runs/frozen_v1_equivalent_seed0")

EXPECTED_OPTIONS: tuple[tuple[str, str], ...] = (
    ("--subset", "data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2"),
    ("--epochs", "50"),
    ("--lr", "1e-2"),
    ("--lr-schedule", "constant"),
    ("--seed", "0"),
    ("--report-every", "5"),
    ("--loss-mode", "background_pseudo_negative"),
    ("--background-quantile", "0.50"),
    ("--hotspot-quantile", "0.90"),
    ("--background-l1-weight", "1.0"),
    ("--background-bias-weight", "1.0"),
    ("--background-over-weight", "1.0"),
    ("--background-relative-weight", "0.10"),
    ("--relative-floor", "0.02"),
    ("--relative-floor-mode", "fixed"),
    ("--hotspot-weight", "0.02"),
    ("--pseudo-negative-quantile", "0.25"),
    ("--pseudo-negative-weight", "0.10"),
    ("--pseudo-negative-loss-type", "relative_l1"),
    ("--pseudo-negative-relative-floor", "0.02"),
    ("--pseudo-negative-over-margin", "0.0"),
    ("--pseudo-negative-min-count", "1"),
    ("--loss-weight-schedule", "constant"),
    ("--selection-metric", "valid_loss"),
    ("--best-predictions-name", "best_predictions.npz"),
)
EXPECTED_FLAGS = (
    "--save-best-predictions",
    "--save-predictions",
)
REFERENCE_METRICS = {
    "best_epoch": ("loss_summary.json", ("best_epoch",)),
    "best_overall_rmse": (
        "run_analysis_best.json",
        ("baseline_comparison", "overall", 0, "metrics", "mean_DeltaT_rmse", "trained_prediction"),
    ),
    "best_overall_mae": (
        "run_analysis_best.json",
        ("baseline_comparison", "overall", 0, "metrics", "mean_T_mae", "trained_prediction"),
    ),
    "best_valid_rmse": (
        "baseline_comparison_best.json",
        ("split_summary", ("split", "valid"), ("predictor", "trained_prediction"), "mean_DeltaT_rmse"),
    ),
    "best_valid_mae": (
        "baseline_comparison_best.json",
        ("split_summary", ("split", "valid"), ("predictor", "trained_prediction"), "mean_DeltaT_mae"),
    ),
    "bin_0_bias": (
        "run_analysis_best.json",
        ("error_bins", "bin_summary", "bin_0", "trained_signed_bias"),
    ),
    "bin_0_over_ratio": (
        "run_analysis_best.json",
        ("error_bins", "bin_summary", "bin_0", "trained_overprediction_ratio"),
    ),
}


def main() -> int:
    config = load_v2_config(REPO_ROOT / STRICT_CONFIG)
    reference = load_v2_config(REPO_ROOT / REFERENCE_CONFIG)
    plan = build_v2_command_plan(config, python_executable="python3")
    command = plan["training_command"]

    option_results = _check_training_command(command)
    output_comparison = _compare_previous_output(reference)
    fingerprint = _fingerprint_subset(config["dataset"]["subset_path"])

    print(f"strict config: {STRICT_CONFIG}")
    print(f"reference config: {REFERENCE_CONFIG}")
    print("training command:")
    print(f"  {shlex.join(command)}")
    _print_option_results(option_results)
    _print_fingerprint(fingerprint)
    _print_output_comparison(output_comparison)
    print("No training, diagnostics, output writes, or dataset array reads were performed.")
    print("Heat3D v2 frozen-v1 alignment check passed.")
    return 0


def _check_training_command(command: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for flag, expected in EXPECTED_OPTIONS:
        actual = _optional_value(command, flag)
        if actual is None:
            status = "missing"
        elif _values_equal(actual, expected):
            status = "matched"
        else:
            status = "mismatched"
        results.append({"flag": flag, "expected": expected, "actual": actual, "status": status})

    for flag in EXPECTED_FLAGS:
        results.append(
            {
                "flag": flag,
                "expected": "present",
                "actual": "present" if flag in command else None,
                "status": "matched" if flag in command else "missing",
            }
        )

    failures = [item for item in results if item["status"] != "matched"]
    if failures:
        raise AssertionError(f"strict training command alignment failed: {failures}")
    return results


def _compare_previous_output(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics = reference.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("reference metrics must be a mapping")

    if not PREVIOUS_RUN_DIR.exists():
        return [
            {
                "metric": metric,
                "expected": _reference_expected(reference, metrics, metric),
                "actual": None,
                "status": "unknown",
                "reason": f"{PREVIOUS_RUN_DIR} is not present locally",
            }
            for metric in REFERENCE_METRICS
        ]

    loaded: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    for metric, (file_name, path) in REFERENCE_METRICS.items():
        expected = _reference_expected(reference, metrics, metric)
        file_path = PREVIOUS_RUN_DIR / file_name
        if not file_path.exists():
            results.append(
                {
                    "metric": metric,
                    "expected": expected,
                    "actual": None,
                    "status": "unknown",
                    "reason": f"missing {file_path}",
                }
            )
            continue
        if file_name not in loaded:
            loaded[file_name] = json.loads(file_path.read_text(encoding="utf-8"))
        actual = _get_path(loaded[file_name], path)
        if actual is None or expected is None:
            status = "unknown"
        elif _values_equal(actual, expected):
            status = "matched"
        else:
            status = "mismatched"
        results.append(
            {
                "metric": metric,
                "expected": expected,
                "actual": actual,
                "status": status,
                "reason": "previous P1.5b run comparison" if status != "unknown" else "missing value",
            }
        )
    return results


def _reference_expected(
    reference: Mapping[str, Any], metrics: Mapping[str, Any], metric: str
) -> Any:
    if metric == "best_epoch":
        training = reference.get("training")
        if isinstance(training, Mapping):
            return training.get("best_epoch")
        return None
    return metrics.get(metric)


def _fingerprint_subset(subset_path: str) -> dict[str, Any]:
    subset = REPO_ROOT / subset_path
    result: dict[str, Any] = {
        "subset_path": subset_path,
        "subset_exists": subset.is_dir(),
        "sample_meta_file_count": None,
        "split_counts": None,
        "sample_id_hash": None,
        "sample_checks": [],
    }
    if not subset.is_dir():
        result["status"] = "unknown"
        result["reason"] = "subset path is not present locally"
        return result

    meta_files = sorted(subset.rglob("sample_meta.json"))
    result["sample_meta_file_count"] = len(meta_files)
    sample_records = _load_sample_records(subset, meta_files)
    sample_ids = sorted(record.get("sample_id") for record in sample_records if record.get("sample_id"))
    result["sample_id_hash"] = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest() if sample_ids else None

    split_counts: dict[str, int] = {}
    for record in sample_records:
        split = record.get("split") or record.get("split_name") or "unknown"
        split_counts[split] = split_counts.get(split, 0) + 1
    result["split_counts"] = split_counts or None
    result["sample_checks"] = _sample_file_checks(subset, sample_ids[:3])
    result["status"] = "checked"
    return result


def _load_sample_records(subset: Path, meta_files: list[Path]) -> list[dict[str, Any]]:
    root_meta = subset / "sample_meta.json"
    if root_meta.exists():
        loaded = json.loads(root_meta.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
        if isinstance(loaded, dict):
            for key in ("samples", "sample_meta", "records"):
                value = loaded.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            if "sample_id" in loaded:
                return [loaded]

    records: list[dict[str, Any]] = []
    for meta_file in meta_files:
        loaded = json.loads(meta_file.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            record = dict(loaded)
            record.setdefault("sample_id", meta_file.parent.name)
            records.append(record)
    return records


def _sample_file_checks(subset: Path, sample_ids: list[str]) -> list[dict[str, Any]]:
    checks = []
    samples_root = subset / "samples"
    expected_names = {
        "metadata": ("metadata.json", "sample_meta.json", "meta.json"),
        "q": ("q.npy", "q_power.npy", "heat_source.npy", "source.npy"),
        "k": ("k.npy", "k_field.npy", "thermal_conductivity.npy"),
        "temperature": ("temperature.npy", "T.npy", "temperature_field.npy"),
    }
    for sample_id in sample_ids:
        sample_dir = samples_root / sample_id
        sample_check: dict[str, Any] = {"sample_id": sample_id, "sample_dir_exists": sample_dir.is_dir()}
        for label, candidates in expected_names.items():
            match = next((sample_dir / name for name in candidates if (sample_dir / name).exists()), None)
            sample_check[label] = {
                "exists": match is not None,
                "path": str(match.relative_to(subset)) if match is not None else None,
                "size_bytes": match.stat().st_size if match is not None else None,
            }
        checks.append(sample_check)
    return checks


def _print_option_results(results: list[dict[str, Any]]) -> None:
    print("training command alignment:")
    for item in results:
        print(f"  {item['status']}: {item['flag']} expected={item['expected']} actual={item['actual']}")


def _print_fingerprint(fingerprint: Mapping[str, Any]) -> None:
    print("subset fingerprint:")
    print(f"  subset_exists: {fingerprint['subset_exists']}")
    print(f"  sample_meta_file_count: {fingerprint['sample_meta_file_count']}")
    print(f"  split_counts: {fingerprint['split_counts']}")
    print(f"  sample_id_hash: {fingerprint['sample_id_hash']}")
    if fingerprint.get("reason"):
        print(f"  reason: {fingerprint['reason']}")
    for sample_check in fingerprint.get("sample_checks", []):
        print(f"  sample_check: {sample_check}")


def _print_output_comparison(results: list[dict[str, Any]]) -> None:
    print("previous output vs reference:")
    for item in results:
        print(
            "  "
            f"{item['status']}: {item['metric']} "
            f"expected={item['expected']} actual={item['actual']} "
            f"reason={item.get('reason')}"
        )


def _optional_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag)
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _values_equal(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    try:
        return abs(float(actual) - float(expected)) <= 1e-10
    except (TypeError, ValueError):
        return False


def _get_path(data: Any, path: tuple[Any, ...]) -> Any:
    value = data
    for part in path:
        if isinstance(part, tuple) and len(part) == 2:
            key, expected = part
            if not isinstance(value, list):
                return None
            value = next((item for item in value if isinstance(item, Mapping) and item.get(key) == expected), None)
        elif isinstance(part, int):
            if not isinstance(value, list) or len(value) <= part:
                return None
            value = value[part]
        else:
            if not isinstance(value, Mapping):
                return None
            value = value.get(part)
        if value is None:
            return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
