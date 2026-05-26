#!/usr/bin/env python3
"""Read-only train/valid split diagnostics for Heat3D v2 subsets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


DEFAULT_SUBSET = Path(
    "data/heat3d-thermal-simulation/subsets/"
    "v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Heat3D v2 train/valid split metadata and label distributions."
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--splits", default="train,valid")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    return loaded if isinstance(loaded, dict) else {}


def _metadata(sample_dir: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in ("sample_meta.json", "metadata.json", "label_meta.json"):
        data = _load_json(sample_dir / name)
        for key, value in data.items():
            merged.setdefault(key, value)
    return merged


def _field(metadata: dict[str, Any], *keys: str, default: Any = "missing") -> Any:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return default


def _as_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _bottom_temperature(metadata: dict[str, Any]) -> float:
    value = metadata.get("bottom_T_fixed_K")
    if value is not None:
        return float(value)
    assembly = metadata.get("assembly")
    if isinstance(assembly, dict) and assembly.get("bottom_dirichlet_T_K") is not None:
        return float(assembly["bottom_dirichlet_T_K"])
    boundary_params = metadata.get("boundary_params")
    if isinstance(boundary_params, dict):
        bottom = boundary_params.get("bottom")
        if isinstance(bottom, dict) and bottom.get("fixed_temperature_K") is not None:
            return float(bottom["fixed_temperature_K"])
    return 300.0


def _temperature_delta(sample_dir: Path, metadata: dict[str, Any]) -> np.ndarray | None:
    path = sample_dir / "temperature.npy"
    if not path.exists():
        return None
    temperature = np.asarray(np.load(path), dtype=np.float64)
    return temperature - _bottom_temperature(metadata)


def _q_range(sample_dir: Path, metadata: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    path = sample_dir / "q_field.npy"
    if not path.exists():
        integrated = metadata.get("integrated_power_W")
        return (float(integrated), None, None) if integrated is not None else (None, None, None)
    q_field = np.asarray(np.load(path), dtype=np.float64)
    return (
        float(metadata["integrated_power_W"]) if metadata.get("integrated_power_W") is not None else None,
        float(np.min(q_field)),
        float(np.max(q_field)),
    )


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "p50": None, "p95": None}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 100:
        return f"{numeric:.2f}"
    if abs(numeric) >= 1:
        return f"{numeric:.4f}"
    return f"{numeric:.6g}"


def _counter_table(counter_by_split: dict[str, Counter[str]], title: str, splits: list[str], limit: int = 20) -> list[str]:
    values: list[str] = []
    for counter in counter_by_split.values():
        values.extend(counter.keys())
    ordered = sorted(set(values))
    if len(ordered) > limit:
        top = Counter()
        for counter in counter_by_split.values():
            top.update(counter)
        ordered = [value for value, _ in top.most_common(limit)]
    lines = [f"### {title}", "", "| value | " + " | ".join(splits) + " |", "|---|" + "|".join("---:" for _ in splits) + "|"]
    for value in ordered:
        row = [value] + [str(counter_by_split[split].get(value, 0)) for split in splits]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _metric_table(metric_by_split: dict[str, list[float]], title: str, splits: list[str]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| split | count | mean | std | min | p50 | p95 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in splits:
        stats = _summary(metric_by_split[split])
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    _fmt(stats["count"]),
                    _fmt(stats["mean"]),
                    _fmt(stats["std"]),
                    _fmt(stats["min"]),
                    _fmt(stats["p50"]),
                    _fmt(stats["p95"]),
                    _fmt(stats["max"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def build_report(subset: Path, splits: list[str]) -> str:
    sample_root = subset / "samples"
    if not sample_root.exists():
        raise FileNotFoundError(f"Missing samples directory: {sample_root}")

    split_set = set(splits)
    sample_counts: Counter[str] = Counter()
    available_fields: dict[str, set[str]] = defaultdict(set)
    counters: dict[str, dict[str, Counter[str]]] = {
        "source_category": defaultdict(Counter),
        "power_scale_category": defaultdict(Counter),
        "bc_category": defaultdict(Counter),
        "k_mode": defaultdict(Counter),
        "k_region_mode": defaultdict(Counter),
    }
    metrics: dict[str, dict[str, list[float]]] = {
        "integrated_power_W": defaultdict(list),
        "q_field_min": defaultdict(list),
        "q_field_max": defaultdict(list),
        "raw_deltaT": defaultdict(list),
        "sample_deltaT_mean": defaultdict(list),
        "sample_deltaT_std": defaultdict(list),
        "sample_deltaT_max": defaultdict(list),
        "sample_deltaT_p95": defaultdict(list),
        "hotspot_deltaT_p95": defaultdict(list),
        "hotspot_deltaT_max": defaultdict(list),
        "hotspot_fraction_ge_0p05K": defaultdict(list),
        "hotspot_fraction_ge_0p10K": defaultdict(list),
        "low_deltaT_fraction_le_0p01K": defaultdict(list),
        "low_deltaT_fraction_le_0p02K": defaultdict(list),
        "low_deltaT_fraction_le_0p05K": defaultdict(list),
    }

    for sample_dir in sorted(path for path in sample_root.iterdir() if path.is_dir()):
        metadata = _metadata(sample_dir)
        split = str(_field(metadata, "split", default="missing"))
        if split not in split_set:
            continue
        sample_counts[split] += 1
        for json_name in ("sample_meta.json", "metadata.json", "label_meta.json"):
            fields = _load_json(sample_dir / json_name).keys()
            available_fields[json_name].update(fields)

        counters["source_category"][split][_as_label(_field(metadata, "source_category", "source_pattern_tag"))] += 1
        counters["power_scale_category"][split][_as_label(_field(metadata, "power_scale_category"))] += 1
        counters["bc_category"][split][_as_label(_field(metadata, "bc_category"))] += 1
        counters["k_mode"][split][_as_label(_field(metadata, "k_mode", "k_field_mode", "supported_k_mode"))] += 1
        counters["k_region_mode"][split][_as_label(_field(metadata, "k_region_mode"))] += 1

        integrated, q_min, q_max = _q_range(sample_dir, metadata)
        if integrated is not None:
            metrics["integrated_power_W"][split].append(integrated)
        if q_min is not None:
            metrics["q_field_min"][split].append(q_min)
        if q_max is not None:
            metrics["q_field_max"][split].append(q_max)

        delta = _temperature_delta(sample_dir, metadata)
        if delta is None:
            continue
        flat = delta.reshape(-1).astype(np.float64)
        metrics["raw_deltaT"][split].extend(float(value) for value in flat)
        metrics["sample_deltaT_mean"][split].append(float(np.mean(flat)))
        metrics["sample_deltaT_std"][split].append(float(np.std(flat)))
        metrics["sample_deltaT_max"][split].append(float(np.max(flat)))
        metrics["sample_deltaT_p95"][split].append(float(np.percentile(flat, 95)))
        metrics["hotspot_deltaT_p95"][split].append(float(np.percentile(flat, 95)))
        metrics["hotspot_deltaT_max"][split].append(float(np.max(flat)))
        metrics["hotspot_fraction_ge_0p05K"][split].append(float(np.mean(flat >= 0.05)))
        metrics["hotspot_fraction_ge_0p10K"][split].append(float(np.mean(flat >= 0.10)))
        metrics["low_deltaT_fraction_le_0p01K"][split].append(float(np.mean(flat <= 0.01)))
        metrics["low_deltaT_fraction_le_0p02K"][split].append(float(np.mean(flat <= 0.02)))
        metrics["low_deltaT_fraction_le_0p05K"][split].append(float(np.mean(flat <= 0.05)))

    lines = [
        "# Heat3D v2 train-valid split diagnostics",
        "",
        "Scope: read-only split diagnostics; not a formal benchmark.",
        "",
        f"Subset: `{subset}`",
        "",
        "## Sample Counts",
        "",
        "| split | sample_count |",
        "|---|---:|",
    ]
    for split in splits:
        lines.append(f"| {split} | {sample_counts.get(split, 0)} |")
    lines.append("")

    for key, title in (
        ("source_category", "Source Category"),
        ("power_scale_category", "Power Scale Category"),
        ("bc_category", "BC Category"),
        ("k_mode", "K Mode"),
        ("k_region_mode", "K Region Mode"),
    ):
        lines.extend(_counter_table(counters[key], title, splits))

    for key, title in (
        ("integrated_power_W", "Integrated Power W"),
        ("q_field_max", "Q Field Max"),
        ("raw_deltaT", "Raw DeltaT Node Distribution"),
        ("sample_deltaT_mean", "Sample Raw DeltaT Mean"),
        ("sample_deltaT_std", "Sample Raw DeltaT Std"),
        ("sample_deltaT_max", "Sample Raw DeltaT Max"),
        ("sample_deltaT_p95", "Sample Raw DeltaT P95"),
        ("hotspot_deltaT_p95", "Hotspot DeltaT P95"),
        ("hotspot_deltaT_max", "Hotspot DeltaT Max"),
        ("hotspot_fraction_ge_0p05K", "Hotspot Fraction >= 0.05 K"),
        ("hotspot_fraction_ge_0p10K", "Hotspot Fraction >= 0.10 K"),
        ("low_deltaT_fraction_le_0p01K", "Low DeltaT Fraction <= 0.01 K"),
        ("low_deltaT_fraction_le_0p02K", "Low DeltaT Fraction <= 0.02 K"),
        ("low_deltaT_fraction_le_0p05K", "Low DeltaT Fraction <= 0.05 K"),
    ):
        lines.extend(_metric_table(metrics[key], title, splits))

    lines.extend(["## Available Metadata Fields", ""])
    for json_name in ("sample_meta.json", "metadata.json", "label_meta.json"):
        fields = sorted(available_fields.get(json_name, set()))
        lines.append(f"### {json_name}")
        lines.append("")
        lines.append(", ".join(f"`{field}`" for field in fields) if fields else "No fields found.")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- `raw_deltaT` is computed as `temperature.npy - bottom_T_fixed_K` when available, falling back to 300 K.",
            "- Hotspot and low-DeltaT fractions are simple node-level threshold summaries for split comparison.",
            "- This report is intended to identify train/valid distribution mismatch candidates, not to rank model performance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    report = build_report(args.subset, splits)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report, encoding="utf-8")
    else:
        print(report)
    print("Heat3D v2 train-valid split diagnostics passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
