#!/usr/bin/env python3
"""Analyze generated Heat3D v1 medium1024 Gap-A diversity.

This is diagnostic tooling only; it is not a formal benchmark validator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FILES = ("metadata.json", "coords.npy", "k_field.npy", "q_field.npy", "temperature.npy")
METADATA_COUNTER_KEYS = (
    "split",
    "source_pattern_tag",
    "k_region_mode",
    "k_field_mode",
    "stack_template",
    "bc_category",
    "power_scale_category",
    "k_contrast_category",
    "barrier_k_category",
)
COMBO_KEYS = (
    "split",
    "source_pattern_tag",
    "k_region_mode",
    "k_field_mode",
    "stack_template",
    "bc_category",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Heat3D v1 medium1024 Gap-A generated subset diversity."
    )
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def _sample_dirs(subset: Path) -> list[Path]:
    root = subset / "samples" if (subset / "samples").is_dir() else subset
    if root.is_dir() and (root / "metadata.json").is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(child for child in root.iterdir() if child.is_dir())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return "<missing>" if value is None else str(value)


def _array_hash(path: Path) -> tuple[str | None, dict[str, Any] | None]:
    try:
        array = np.load(path)
    except Exception:
        return None, None
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    stats = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.all(np.isfinite(array))),
        "min": float(np.min(array)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
    }
    return digest.hexdigest(), stats


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in counter.items()}


def _top_counts(counter: Counter[str], top_n: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": int(count)} for key, count in counter.most_common(top_n)]


def _combo_key(metadata: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_metadata_value(metadata, key) for key in COMBO_KEYS)


def _combo_label(combo: tuple[str, ...]) -> str:
    return " | ".join(f"{key}={value}" for key, value in zip(COMBO_KEYS, combo))


def analyze_subset(subset: Path, top_n: int = 30) -> dict[str, Any]:
    sample_dirs = _sample_dirs(subset)
    metadata_counters = {key: Counter() for key in METADATA_COUNTER_KEYS}
    combo_counter: Counter[tuple[str, ...]] = Counter()
    q_hash_counter: Counter[str] = Counter()
    k_hash_counter: Counter[str] = Counter()
    temperature_hash_counter: Counter[str] = Counter()
    combo_records: dict[tuple[str, ...], dict[str, Any]] = {}
    missing_metadata_count = 0
    missing_required_file_count = 0
    nonfinite_array_count = 0
    sample_errors: list[dict[str, Any]] = []

    for sample_dir in sample_dirs:
        missing = [name for name in REQUIRED_FILES if not (sample_dir / name).is_file()]
        if missing:
            missing_required_file_count += len(missing)
            sample_errors.append({"sample_id": sample_dir.name, "missing_files": missing})
        metadata: dict[str, Any] = {}
        if (sample_dir / "metadata.json").is_file():
            try:
                metadata = _read_json(sample_dir / "metadata.json")
            except Exception as exc:
                missing_metadata_count += 1
                sample_errors.append({"sample_id": sample_dir.name, "metadata_error": str(exc)})
        else:
            missing_metadata_count += 1

        for key, counter in metadata_counters.items():
            counter[_metadata_value(metadata, key)] += 1

        combo = _combo_key(metadata)
        combo_counter[combo] += 1
        record = combo_records.setdefault(
            combo,
            {
                "combo": {key: value for key, value in zip(COMBO_KEYS, combo)},
                "sample_count": 0,
                "q_hashes": Counter(),
                "k_hashes": Counter(),
                "temperature_hashes": Counter(),
                "T_max_values": [],
                "q_max_values": [],
            },
        )
        record["sample_count"] += 1

        coords_path = sample_dir / "coords.npy"
        if coords_path.is_file():
            _, coords_stats = _array_hash(coords_path)
            if coords_stats is None:
                sample_errors.append({"sample_id": sample_dir.name, "array_error": "coords.npy"})
            elif not coords_stats["finite"]:
                nonfinite_array_count += 1

        for filename, global_counter, record_key in (
            ("q_field.npy", q_hash_counter, "q_hashes"),
            ("k_field.npy", k_hash_counter, "k_hashes"),
            ("temperature.npy", temperature_hash_counter, "temperature_hashes"),
        ):
            path = sample_dir / filename
            if not path.is_file():
                continue
            digest, stats = _array_hash(path)
            if digest is None or stats is None:
                sample_errors.append({"sample_id": sample_dir.name, "array_error": filename})
                continue
            if not stats["finite"]:
                nonfinite_array_count += 1
            global_counter[digest] += 1
            record[record_key][digest] += 1
            if filename == "q_field.npy" and stats["max"] is not None:
                record["q_max_values"].append(float(stats["max"]))
            if filename == "temperature.npy" and stats["max"] is not None:
                record["T_max_values"].append(float(stats["max"]))

    per_combo = []
    for combo, record in combo_records.items():
        t_values = record["T_max_values"]
        q_values = record["q_max_values"]
        per_combo.append(
            {
                "combo": record["combo"],
                "combo_label": _combo_label(combo),
                "sample_count": int(record["sample_count"]),
                "unique_q_hash_count": len(record["q_hashes"]),
                "unique_k_hash_count": len(record["k_hashes"]),
                "unique_temperature_hash_count": len(record["temperature_hashes"]),
                "T_max_min": float(min(t_values)) if t_values else None,
                "T_max_max": float(max(t_values)) if t_values else None,
                "T_max_range": float(max(t_values) - min(t_values)) if t_values else None,
                "q_max_min": float(min(q_values)) if q_values else None,
                "q_max_max": float(max(q_values)) if q_values else None,
                "q_max_range": float(max(q_values) - min(q_values)) if q_values else None,
            }
        )
    per_combo.sort(key=lambda item: (-item["sample_count"], item["combo_label"]))

    sample_count = len(sample_dirs)
    max_combo_count = max(combo_counter.values(), default=0)
    combo_gt_20_count = sum(1 for count in combo_counter.values() if count > 20)
    max_q_hash_repeat = max(q_hash_counter.values(), default=0)
    max_k_hash_repeat = max(k_hash_counter.values(), default=0)
    max_temperature_hash_repeat = max(temperature_hash_counter.values(), default=0)
    unique_q_hash_fraction = len(q_hash_counter) / sample_count if sample_count else 0.0
    unique_k_hash_fraction = len(k_hash_counter) / sample_count if sample_count else 0.0
    unique_temperature_hash_fraction = len(temperature_hash_counter) / sample_count if sample_count else 0.0
    per_combo_min_unique_q = min((item["unique_q_hash_count"] for item in per_combo), default=0)
    per_combo_min_unique_k = min((item["unique_k_hash_count"] for item in per_combo), default=0)
    per_combo_min_unique_temperature = min((item["unique_temperature_hash_count"] for item in per_combo), default=0)
    likely_coarse_combo_repetition = combo_gt_20_count > 0 or max_combo_count > max(20, len(sample_dirs) // 8)
    likely_true_q_duplicates = any(count > 1 for count in q_hash_counter.values())
    likely_true_k_duplicates = any(count > 1 for count in k_hash_counter.values())
    likely_true_temperature_duplicates = any(count > 1 for count in temperature_hash_counter.values())
    no_file_errors = missing_metadata_count == 0 and missing_required_file_count == 0 and nonfinite_array_count == 0
    gap_a_covered = (
        metadata_counters["source_pattern_tag"]["low_power_near_zero_background_cases"] > 0
        and metadata_counters["source_pattern_tag"]["high_dynamic_range_power_cases"] > 0
        and metadata_counters["k_region_mode"]["high_contrast_interface_k"] > 0
        and metadata_counters["k_region_mode"]["low_k_barrier_or_TIM_variation"] > 0
        and metadata_counters["bc_category"]["very_low_top_h_candidate"] > 0
        and metadata_counters["bc_category"]["very_high_top_h_candidate"] > 0
    )
    training_smoke_ready = bool(sample_dirs) and no_file_errors and gap_a_covered
    e50_probe_ready = (
        sample_count >= 32
        and no_file_errors
        and gap_a_covered
        and unique_q_hash_fraction >= 0.75
        and unique_temperature_hash_fraction >= 0.75
        and unique_k_hash_fraction >= 0.25
        and max_q_hash_repeat <= max(4, int(0.10 * sample_count))
        and max_temperature_hash_repeat <= max(4, int(0.10 * sample_count))
    )
    formal_benchmark_ready = (
        sample_count >= 1024
        and no_file_errors
        and gap_a_covered
        and unique_q_hash_fraction >= 0.95
        and unique_temperature_hash_fraction >= 0.95
        and unique_k_hash_fraction >= 0.50
        and max_q_hash_repeat <= max(8, int(0.025 * sample_count))
        and max_temperature_hash_repeat <= max(8, int(0.025 * sample_count))
        and not likely_coarse_combo_repetition
    )

    return {
        "scope": "diagnostic only; not a formal benchmark",
        "subset": str(subset),
        "sample_count": sample_count,
        "missing_metadata_count": missing_metadata_count,
        "missing_required_file_count": missing_required_file_count,
        "nonfinite_array_count": nonfinite_array_count,
        "metadata_counters": {key: _counter_dict(counter) for key, counter in metadata_counters.items()},
        "combo_count": len(combo_counter),
        "top_repeated_combos": [
            {"combo": {key: value for key, value in zip(COMBO_KEYS, combo)}, "count": int(count)}
            for combo, count in combo_counter.most_common(top_n)
        ],
        "combo_gt_20_count": int(combo_gt_20_count),
        "max_combo_count": int(max_combo_count),
        "unique_q_hash_count": len(q_hash_counter),
        "unique_k_hash_count": len(k_hash_counter),
        "unique_temperature_hash_count": len(temperature_hash_counter),
        "unique_q_hash_fraction": float(unique_q_hash_fraction),
        "unique_k_hash_fraction": float(unique_k_hash_fraction),
        "unique_temperature_hash_fraction": float(unique_temperature_hash_fraction),
        "max_q_hash_repeat": int(max_q_hash_repeat),
        "max_k_hash_repeat": int(max_k_hash_repeat),
        "max_temperature_hash_repeat": int(max_temperature_hash_repeat),
        "per_combo_min_unique_q": int(per_combo_min_unique_q),
        "per_combo_min_unique_k": int(per_combo_min_unique_k),
        "per_combo_min_unique_T": int(per_combo_min_unique_temperature),
        "top_repeated_q_hashes": _top_counts(q_hash_counter, top_n),
        "top_repeated_k_hashes": _top_counts(k_hash_counter, top_n),
        "top_repeated_temperature_hashes": _top_counts(temperature_hash_counter, top_n),
        "per_combo_diversity": per_combo[:top_n],
        "diagnostic_flags": {
            "likely_coarse_combo_repetition": bool(likely_coarse_combo_repetition),
            "likely_true_q_duplicates": bool(likely_true_q_duplicates),
            "likely_true_k_duplicates": bool(likely_true_k_duplicates),
            "likely_true_temperature_duplicates": bool(likely_true_temperature_duplicates),
            "diversity_ready_for_training_smoke": bool(training_smoke_ready),
            "diversity_ready_for_e50_probe": bool(e50_probe_ready),
            "diversity_ready_for_formal_benchmark": bool(formal_benchmark_ready),
        },
        "sample_errors": sample_errors,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Heat3D v1 Medium1024 Gap-A Diversity Diagnostics",
        "",
        "This is diagnostic only; not a formal benchmark.",
        "",
        "## Summary",
        "",
        f"- subset: `{result['subset']}`",
        f"- sample_count: {result['sample_count']}",
        f"- missing_metadata_count: {result['missing_metadata_count']}",
        f"- missing_required_file_count: {result['missing_required_file_count']}",
        f"- nonfinite_array_count: {result['nonfinite_array_count']}",
        f"- combo_count: {result['combo_count']}",
        f"- max_combo_count: {result['max_combo_count']}",
        f"- combo_gt_20_count: {result['combo_gt_20_count']}",
        f"- unique_q_hash_fraction: {_fmt(result['unique_q_hash_fraction'])}",
        f"- unique_k_hash_fraction: {_fmt(result['unique_k_hash_fraction'])}",
        f"- unique_temperature_hash_fraction: {_fmt(result['unique_temperature_hash_fraction'])}",
        f"- max_q_hash_repeat: {result['max_q_hash_repeat']}",
        f"- max_k_hash_repeat: {result['max_k_hash_repeat']}",
        f"- max_temperature_hash_repeat: {result['max_temperature_hash_repeat']}",
        f"- per_combo_min_unique_q: {result['per_combo_min_unique_q']}",
        f"- per_combo_min_unique_k: {result['per_combo_min_unique_k']}",
        f"- per_combo_min_unique_T: {result['per_combo_min_unique_T']}",
        "",
        "## Diagnostic Flags",
        "",
    ]
    for key, value in result["diagnostic_flags"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Metadata Counters", ""])
    for key, counts in result["metadata_counters"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("| value | count |")
        lines.append("| --- | ---: |")
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{value}` | {count} |")
        lines.append("")

    lines.extend(
        [
            "## Array Hash Diversity",
            "",
            f"- unique_q_hash_count: {result['unique_q_hash_count']}",
            f"- unique_k_hash_count: {result['unique_k_hash_count']}",
            f"- unique_temperature_hash_count: {result['unique_temperature_hash_count']}",
            f"- max_q_hash_repeat: {result['max_q_hash_repeat']}",
            f"- max_k_hash_repeat: {result['max_k_hash_repeat']}",
            f"- max_temperature_hash_repeat: {result['max_temperature_hash_repeat']}",
            "",
            "## Top Repeated Combos",
            "",
            "| count | combo |",
            "| ---: | --- |",
        ]
    )
    for item in result["top_repeated_combos"]:
        combo = " / ".join(f"{key}={value}" for key, value in item["combo"].items())
        lines.append(f"| {item['count']} | `{combo}` |")

    lines.extend(["", "## Per-Combo Diversity", ""])
    lines.append("| samples | unique q | unique k | unique T | T_max range | q_max range | combo |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in result["per_combo_diversity"]:
        lines.append(
            "| "
            f"{item['sample_count']} | "
            f"{item['unique_q_hash_count']} | "
            f"{item['unique_k_hash_count']} | "
            f"{item['unique_temperature_hash_count']} | "
            f"{_fmt(item['T_max_range'])} | "
            f"{_fmt(item['q_max_range'])} | "
            f"`{item['combo_label']}` |"
        )

    if result["sample_errors"]:
        lines.extend(["", "## Sample Errors", ""])
        for error in result["sample_errors"]:
            lines.append(f"- `{error}`")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")
    result = analyze_subset(args.subset, top_n=args.top_n)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, result)

    print("Heat3D v1 Medium1024 Gap-A Diversity Diagnostics")
    print("scope: diagnostic only; not a formal benchmark")
    print(f"subset: {args.subset}")
    print(f"sample_count: {result['sample_count']}")
    print(f"combo_count: {result['combo_count']}")
    print(f"max_combo_count: {result['max_combo_count']}")
    print(f"unique_q_hash_count: {result['unique_q_hash_count']}")
    print(f"unique_k_hash_count: {result['unique_k_hash_count']}")
    print(f"unique_temperature_hash_count: {result['unique_temperature_hash_count']}")
    print(f"unique_q_hash_fraction: {result['unique_q_hash_fraction']:.6f}")
    print(f"unique_k_hash_fraction: {result['unique_k_hash_fraction']:.6f}")
    print(f"unique_temperature_hash_fraction: {result['unique_temperature_hash_fraction']:.6f}")
    print(f"max_q_hash_repeat: {result['max_q_hash_repeat']}")
    print(f"max_k_hash_repeat: {result['max_k_hash_repeat']}")
    print(f"max_temperature_hash_repeat: {result['max_temperature_hash_repeat']}")
    print(f"diagnostic_flags: {result['diagnostic_flags']}")
    print(f"output_json: {args.output_json}")
    print(f"output_md: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
