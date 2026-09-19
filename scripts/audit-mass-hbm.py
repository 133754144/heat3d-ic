#!/usr/bin/env python3
"""Read-only forensic inventory for the MASS-HBM handoff dataset.

The script never writes below ``dataset_root`` and never imports Heat3D
training code.  It reads the packaged CSV/JSON metadata for all cases and
loads only a deterministic, stratified subset of field arrays.  The output is
a compact JSON receipt suitable for review and report generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


KEYWORDS = {
    "geometry": ("geometry", "mesh", "grid", "pitch", "slab", "layer", "thickness", "extent", "dimension"),
    "material": ("material", "region", "conduct", "kx", "ky", "kz"),
    "interface": ("interface", "tbr", "rint", "resistance"),
    "boundary": ("boundary", "ambient", "htc", "robin", "sink", "cool", "channel", "convection"),
    "power": ("power", "heat", "activity", "workload", "leakage", "refresh"),
    "temperature": ("temperature", "temp", "tmax", "tmin", "delta_t"),
    "mechanical": ("stress", "strain", "sigma", "warpage"),
    "solver": ("solver", "iteration", "converg", "residual", "runtime", "timing", "elapsed", "wall"),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _finite_number(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _column_summary(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    values = [row.get(field, "").strip() for row in rows]
    populated = [value for value in values if value != ""]
    numbers = [_finite_number(value) for value in populated]
    numeric = [value for value in numbers if value is not None]
    summary: dict[str, Any] = {
        "populated": len(populated),
        "missing": len(rows) - len(populated),
        "unique_count": len(set(populated)),
    }
    if populated and len(numeric) == len(populated):
        array = np.asarray(numeric, dtype=np.float64)
        summary.update(
            {
                "kind": "numeric",
                "min": float(array.min()),
                "max": float(array.max()),
                "quantiles": {
                    "q00": float(np.quantile(array, 0.0)),
                    "q25": float(np.quantile(array, 0.25)),
                    "q50": float(np.quantile(array, 0.5)),
                    "q75": float(np.quantile(array, 0.75)),
                    "q100": float(np.quantile(array, 1.0)),
                },
            }
        )
    else:
        counts = Counter(populated)
        summary.update(
            {
                "kind": "categorical_or_text",
                "values": dict(counts.most_common(100)),
                "values_truncated": len(counts) > 100,
            }
        )
    return summary


def _case_dir(root: Path, row: dict[str, str]) -> Path | None:
    priority = (
        "packaged_case_dir",
        "case_dir",
        "relative_case_dir",
        "dataset_path",
        "path",
    )
    for key in priority + tuple(row):
        value = row.get(key, "").strip()
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_dir() and (candidate / "labels.json").exists():
            return candidate
    return None


def _pick_field(row: dict[str, str], fragments: Iterable[str]) -> str:
    for key, value in row.items():
        low = key.lower()
        if any(fragment in low for fragment in fragments) and value.strip():
            return value.strip()
    return "UNKNOWN"


def _stable_representatives(root: Path, rows: list[dict[str, str]], cap: int) -> list[tuple[dict[str, str], Path]]:
    candidates: list[tuple[dict[str, str], Path]] = []
    for row in rows:
        path = _case_dir(root, row)
        if path is not None:
            candidates.append((row, path))
    candidates.sort(key=lambda item: str(item[1]))
    chosen: dict[str, tuple[dict[str, str], Path]] = {}
    grouping_specs = (
        ("architecture",),
        ("experiment",),
        ("sweep", "scan", "family"),
        ("workload", "activity", "model"),
        ("pitch",),
        ("slab", "stack"),
        ("cool", "htc", "channel"),
    )
    for fragments in grouping_specs:
        groups: dict[str, list[tuple[dict[str, str], Path]]] = defaultdict(list)
        for row, path in candidates:
            groups[_pick_field(row, fragments)].append((row, path))
        for group in sorted(groups):
            items = groups[group]
            for index in sorted({0, len(items) // 2, len(items) - 1}):
                row, path = items[index]
                chosen[str(path)] = (row, path)
                if len(chosen) >= cap:
                    return list(chosen.values())
    for row, path in candidates:
        chosen.setdefault(str(path), (row, path))
        if len(chosen) >= cap:
            break
    return list(chosen.values())


def _flatten_json(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_json(child, child_prefix)
    elif isinstance(value, list):
        if len(value) <= 32 and all(not isinstance(item, (dict, list)) for item in value):
            yield prefix, value
        else:
            yield prefix, {"type": "list", "length": len(value)}
    else:
        yield prefix, value


def _keyword_hits(json_files: list[Path], case_dir: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {key: [] for key in KEYWORDS}
    for path in json_files:
        try:
            data = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            result.setdefault("read_errors", []).append({"file": str(path.relative_to(case_dir)), "error": str(exc)})
            continue
        for key_path, value in _flatten_json(data):
            low = key_path.lower()
            for category, needles in KEYWORDS.items():
                if any(needle in low for needle in needles):
                    hits = result[category]
                    if len(hits) < 250:
                        hits.append(
                            {
                                "file": str(path.relative_to(case_dir)),
                                "key": key_path,
                                "value": value,
                            }
                        )
    return result


def _csv_profile(path: Path, case_dir: Path) -> dict[str, Any]:
    fields, rows = _read_csv(path)
    return {
        "relative_file": str(path.relative_to(case_dir)),
        "row_count": len(rows),
        "fields": fields,
        "column_summaries": {field: _column_summary(rows, field) for field in fields},
        "head": rows[:5],
        "tail": rows[-2:] if len(rows) > 5 else [],
    }


def _aggregate_case_summaries(root: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    values: dict[str, list[Any]] = defaultdict(list)
    found = 0
    for row in rows:
        case_dir = _case_dir(root, row)
        if case_dir is None:
            continue
        path = case_dir / "temperature_output" / "case_summary.json"
        if not path.is_file():
            continue
        found += 1
        data = _read_json(path)
        for key, value in _flatten_json(data):
            low = key.lower()
            if any(needle in low for needles in KEYWORDS.values() for needle in needles):
                if isinstance(value, (str, int, float, bool)) or value is None:
                    values[key].append(value)
    result: dict[str, Any] = {"case_summary_count": found, "fields": {}}
    for key, field_values in sorted(values.items()):
        finite = [float(value) for value in field_values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))]
        if finite and len(finite) == len([value for value in field_values if value is not None]):
            array = np.asarray(finite, dtype=np.float64)
            result["fields"][key] = {
                "kind": "numeric",
                "count": len(finite),
                "min": float(array.min()),
                "max": float(array.max()),
                "quantiles": {name: float(np.quantile(array, q)) for name, q in (("q00", 0.0), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q100", 1.0))},
            }
        else:
            counts = Counter(str(value) for value in field_values)
            result["fields"][key] = {
                "kind": "categorical",
                "count": len(field_values),
                "values": dict(counts.most_common(50)),
                "values_truncated": len(counts) > 50,
            }
    return result


def _shared_config_evidence(root: Path) -> list[dict[str, Any]]:
    base = root / "shared_configs"
    result: list[dict[str, Any]] = []
    if not base.is_dir():
        return result
    needles = tuple(needle for values in KEYWORDS.values() for needle in values)
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".csv", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            if any(needle in low for needle in needles):
                lines.append({"line": line_number, "text": line[:1000]})
        result.append(
            {
                "relative_file": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "keyword_lines": lines[:500],
                "keyword_lines_truncated": len(lines) > 500,
            }
        )
    return result


def _derived_index_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Compute only unit-explicit quantities derivable without guessing."""

    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        t_min = _finite_number(row.get("temperature_min_C", ""))
        t_max = _finite_number(row.get("temperature_max_C", ""))
        if t_min is not None and t_max is not None:
            values["temperature_delta_C"].append(t_max - t_min)
        shape = row.get("shape_zyx", "").strip().lower()
        try:
            dimensions = [int(item) for item in shape.split("x")]
        except ValueError:
            dimensions = []
        if len(dimensions) == 3 and all(item > 0 for item in dimensions):
            values["field_point_count"].append(float(math.prod(dimensions)))

    result: dict[str, Any] = {}
    for key, field_values in values.items():
        array = np.asarray(field_values, dtype=np.float64)
        result[key] = {
            "count": int(array.size),
            "min": float(array.min()),
            "max": float(array.max()),
            "quantiles": {
                name: float(np.quantile(array, q))
                for name, q in (("q00", 0.0), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q100", 1.0))
            },
        }
    return result


def _sample_array(array: np.ndarray, sample_limit: int) -> np.ndarray:
    flat = np.asarray(array).reshape(-1)
    if flat.size <= sample_limit:
        return np.asarray(flat)
    step = max(1, flat.size // sample_limit)
    return np.asarray(flat[::step][:sample_limit])


def _array_stats(array: np.ndarray, sample_limit: int) -> dict[str, Any]:
    sample = _sample_array(array, sample_limit).astype(np.float64, copy=False)
    finite = sample[np.isfinite(sample)]
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "element_count": int(array.size),
        "sample_count": int(sample.size),
        "finite_sample_count": int(finite.size),
        "sampling": "all_values" if array.size <= sample_limit else "deterministic_uniform_stride",
    }
    if finite.size:
        result.update(
            {
                "sample_min": float(finite.min()),
                "sample_max": float(finite.max()),
                "sample_quantiles": {
                    "q00": float(np.quantile(finite, 0.0)),
                    "q01": float(np.quantile(finite, 0.01)),
                    "q05": float(np.quantile(finite, 0.05)),
                    "q25": float(np.quantile(finite, 0.25)),
                    "q50": float(np.quantile(finite, 0.5)),
                    "q75": float(np.quantile(finite, 0.75)),
                    "q95": float(np.quantile(finite, 0.95)),
                    "q99": float(np.quantile(finite, 0.99)),
                    "q100": float(np.quantile(finite, 1.0)),
                },
                "zero_fraction_in_sample": float(np.mean(finite == 0.0)),
            }
        )
    return result


def _inspect_array_file(path: Path, sample_limit: int) -> dict[str, Any]:
    result: dict[str, Any] = {"relative_file": str(path), "size_bytes": path.stat().st_size}
    try:
        if path.suffix.lower() == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            result["arrays"] = {"array": _array_stats(array, sample_limit)}
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                result["arrays"] = {name: _array_stats(archive[name], sample_limit) for name in archive.files}
    except Exception as exc:  # fail closed and preserve forensic evidence
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _aggregate_array_samples(case_receipts: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = defaultdict(list)
    shapes: dict[str, Counter[str]] = defaultdict(Counter)
    for case in case_receipts:
        for file_receipt in case["array_files"]:
            filename = Path(file_receipt["relative_file"]).name
            for array_name, stats in file_receipt.get("arrays", {}).items():
                key = f"{filename}:{array_name}"
                shapes[key][str(stats["shape"])] += 1
                quantiles = stats.get("sample_quantiles", {})
                for qkey, value in quantiles.items():
                    buckets[f"{key}:{qkey}"].append(float(value))
    return {
        "per_array_shape_counts": {key: dict(counter) for key, counter in shapes.items()},
        "case_stat_envelopes": {
            key: {"min": min(values), "max": max(values), "case_count": len(values)}
            for key, values in buckets.items()
            if values
        },
        "interpretation": "Envelope of per-case deterministic sample statistics, not a pooled voxel quantile.",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/mass-audit.json"))
    parser.add_argument("--representative-cap", type=int, default=32)
    parser.add_argument("--array-sample-limit", type=int, default=100_000)
    parser.add_argument("--skip-arrays", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"dataset root does not exist: {root}")
    required = [root / "dataset_metadata.json", root / "dataset_index.csv", root / "files_manifest.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required metadata: {missing}")

    metadata = _read_json(root / "dataset_metadata.json")
    index_fields, rows = _read_csv(root / "dataset_index.csv")
    manifest_fields, manifest_rows = _read_csv(root / "files_manifest.csv")
    manifest_path_field = next((field for field in manifest_fields if "path" in field.lower()), manifest_fields[0])
    manifest_size_field = next((field for field in manifest_fields if "size" in field.lower()), "")

    suffix_counts: Counter[str] = Counter()
    basename_counts: Counter[str] = Counter()
    total_bytes = 0
    for row in manifest_rows:
        rel = row.get(manifest_path_field, "")
        path = Path(rel)
        suffix_counts[path.suffix.lower() or "<none>"] += 1
        basename_counts[path.name] += 1
        if manifest_size_field:
            size = _finite_number(row.get(manifest_size_field, ""))
            if size is not None:
                total_bytes += int(size)

    representatives = _stable_representatives(root, rows, args.representative_cap)
    case_receipts: list[dict[str, Any]] = []
    for row, case_dir in representatives:
        json_files = sorted(case_dir.rglob("*.json"))
        csv_files = sorted(case_dir.rglob("*.csv"))
        array_files = sorted(
            path
            for path in case_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".npy", ".npz"}
        )
        case_receipts.append(
            {
                "case_dir": str(case_dir.relative_to(root)),
                "index_identity": {
                    key: value
                    for key, value in row.items()
                    if value and any(fragment in key.lower() for fragment in ("case", "experiment", "architecture", "sweep", "scan", "pitch", "slab", "workload", "cool"))
                },
                "file_basenames": [path.name for path in sorted(case_dir.rglob("*")) if path.is_file()],
                "json_keyword_hits": _keyword_hits(json_files, case_dir),
                "csv_profiles": [_csv_profile(path, case_dir) for path in csv_files],
                "array_files": [
                    _inspect_array_file(path, args.array_sample_limit) | {"relative_file": str(path.relative_to(case_dir))}
                    for path in ([] if args.skip_arrays else array_files)
                ],
            }
        )

    output = {
        "audit_schema_version": "mass-hbm-readonly-forensic-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "read_only_contract": {
            "dataset_writes": 0,
            "training_calls": 0,
            "solver_calls": 0,
            "full_array_copy": False,
            "array_policy": "deterministic stratified case subset; mmap NPY; selected NPZ only",
        },
        "root_metadata": metadata,
        "metadata_sha256": {path.name: _sha256(path) for path in required},
        "index": {
            "row_count": len(rows),
            "fields": index_fields,
            "column_summaries": {field: _column_summary(rows, field) for field in index_fields},
            "derived_stats": _derived_index_stats(rows),
            "resolved_case_dir_count": sum(_case_dir(root, row) is not None for row in rows),
        },
        "manifest": {
            "row_count": len(manifest_rows),
            "fields": manifest_fields,
            "suffix_counts": dict(suffix_counts),
            "basename_counts": dict(basename_counts.most_common()),
            "total_bytes_from_manifest": total_bytes if manifest_size_field else None,
        },
        "all_case_summary_aggregates": _aggregate_case_summaries(root, rows),
        "shared_config_evidence": _shared_config_evidence(root),
        "representative_selection": {
            "method": "deterministic coverage of architecture, experiment, sweep family, workload, pitch, slab/stack, and cooling fields",
            "requested_cap": args.representative_cap,
            "selected_count": len(representatives),
            "array_sample_limit_per_array": args.array_sample_limit,
        },
        "representative_cases": case_receipts,
        "representative_array_aggregate": _aggregate_array_samples(case_receipts),
        "limitations": [
            "Array quantiles are deterministic representative-case samples, not pooled quantiles over every voxel in all 314 cases.",
            "NPZ arrays are decompressed only for selected representative cases.",
            "Semantic classifications require cross-checking packaged metadata against the paper and author confirmation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "representative_cases": len(case_receipts), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
