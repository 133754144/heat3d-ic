"""Audit Heat3D v2 train batch composition for controlled-run YAML configs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
SCRIPTS_DIR = REPO_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402


CATEGORY_FIELDS = (
    "stack_template",
    "source_category",
    "k_region_mode",
    "k_field_mode",
    "bc_category",
    "power_scale_category",
)

DEFAULT_OUTPUT_DIR = REPO_DIR / "output" / "heat3d_v3_batch_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit train batch sample IDs and metadata composition for a Heat3D v2 YAML."
    )
    parser.add_argument("config", type=Path, help="Heat3D v2 controlled YAML config.")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch index used for shuffled batch order.")
    parser.add_argument(
        "--batch-plan",
        choices=("current_graph_shape", "sample_shuffle"),
        default="current_graph_shape",
        help="Batch plan to audit. current_graph_shape follows the runner; sample_shuffle shuffles samples then chunks.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override run.batch_size for audit only.")
    parser.add_argument(
        "--batch-build-seed",
        type=int,
        default=None,
        help="Seed for sample_shuffle, or runner-compatible train-batch order in current_graph_shape.",
    )
    parser.add_argument(
        "--metadata-only",
        "--no-graph-build",
        dest="metadata_only",
        action="store_true",
        help="Do not load dataset or build graphs. Only sample_shuffle supports this mode.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def _output_paths(config_path: Path, output_json: Path | None, output_md: Path | None) -> tuple[Path, Path]:
    stem = config_path.stem
    json_path = output_json or DEFAULT_OUTPUT_DIR / f"{stem}_batch_composition.json"
    md_path = output_md or DEFAULT_OUTPUT_DIR / f"{stem}_batch_composition.md"
    return json_path, md_path


def _sample_root(path: Path) -> Path:
    samples = path / "samples"
    if samples.is_dir():
        return samples
    return path


def _resolve_path(path_value: str | Path, *, config_path: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, REPO_DIR / path]
    if config_path is not None:
        parent = config_path.parent
        candidates.append(parent / path)
        candidates.extend(ancestor / path for ancestor in parent.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_external_split_map(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    mapping = loaded.get("sample_splits", loaded)
    if not isinstance(mapping, dict):
        raise ValueError(f"split map must be a mapping or contain sample_splits: {path}")
    split_ids: dict[str, list[str]] = {}
    for sample_id, split in mapping.items():
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"split map contains invalid sample_id: {sample_id!r}")
        if not isinstance(split, str) or not split:
            raise ValueError(f"split map contains invalid split for {sample_id!r}: {split!r}")
        split_ids.setdefault(split, []).append(sample_id)
    return {split: sorted(ids) for split, ids in split_ids.items()}


def _resolve_training_splits(split_map_path: Path | None) -> tuple[dict[str, list[str]], str, str, str | None]:
    if split_map_path is None:
        raise ValueError("batch composition audit currently requires dataset.split_map_path")
    split_ids = _load_external_split_map(split_map_path)
    train_ids = split_ids.get("train", [])
    valid_iid_ids = split_ids.get("valid_iid", [])
    if not train_ids or not valid_iid_ids:
        raise ValueError(
            "Expected non-empty train and valid_iid splits, "
            f"found train={len(train_ids)} valid_iid={len(valid_iid_ids)}"
        )
    return split_ids, "split_map", "valid_iid", "valid_stress" if split_ids.get("valid_stress") else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _chunk_items(items: list[str], *, batch_size: int, drop_last: bool) -> list[list[str]]:
    chunks = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        if len(chunk) < batch_size and drop_last:
            continue
        chunks.append(chunk)
    return chunks


def _config_graph(config: dict[str, Any]) -> dict[str, Any]:
    graph = config.get("graph") if isinstance(config.get("graph"), dict) else {}
    return {
        "radius_policy": graph.get("radius_policy", "legacy_kdtree_mean4"),
        "coverage_repair_policy": graph.get("coverage_repair_policy", "none"),
        "repair_p2r": bool(graph.get("repair_p2r", True)),
        "repair_r2p": bool(graph.get("repair_r2p", True)),
        "min_physical_coverage": int(graph.get("min_physical_coverage", 1)),
    }


def _optimizer_seed(config: dict[str, Any], field: str, default: int) -> int:
    optimizer = config.get("optimizer") if isinstance(config.get("optimizer"), dict) else {}
    value = optimizer.get(field)
    return default if value is None else int(value)


def _meta_value(meta: dict[str, Any], field: str) -> str:
    value = _nested_meta_value(meta, field)
    if value is None and field == "source_category":
        value = _nested_meta_value(meta, "source_pattern_tag")
    if value is None:
        return "unknown"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(_json_safe(value), sort_keys=True)


def _nested_meta_value(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        if field in value:
            return value[field]
        for nested in value.values():
            found = _nested_meta_value(nested, field)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _nested_meta_value(item, field)
            if found is not None:
                return found
    return None


def _target_delta_stats(group: dict[str, Any]) -> dict[str, float]:
    values = np.asarray(group["target_delta_raw"], dtype=np.float64).reshape(-1)
    return {
        "available": True,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _target_delta_stats_from_examples(sample_ids: tuple[str, ...], example_by_id: dict[str, Any]) -> dict[str, Any]:
    if not example_by_id:
        return {
            "available": False,
            "reason": "metadata_only_or_dataset_unavailable",
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    values = []
    for sample_id in sample_ids:
        example = example_by_id.get(sample_id)
        if example is None:
            continue
        bridge = example.build_temperature_rise_legacy_inputs_from_relative_features(
            bridge_policy="zero_delta_u_bridge"
        )
        values.append(np.asarray(bridge.target_delta_u, dtype=np.float64).reshape(-1))
    if not values:
        return {
            "available": False,
            "reason": "no_target_arrays",
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    all_values = np.concatenate(values)
    return {
        "available": True,
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "min": float(np.min(all_values)),
        "max": float(np.max(all_values)),
    }


def _category_counts(sample_ids: tuple[str, ...], example_by_id: dict[str, Any]) -> dict[str, dict[str, int]]:
    result = {}
    for field in CATEGORY_FIELDS:
        counter = Counter()
        for sample_id in sample_ids:
            example = example_by_id.get(sample_id)
            value = "unknown" if example is None else _meta_value(example.meta, field)
            counter[value] += 1
        result[field] = dict(sorted(counter.items()))
    return result


def _max_fraction(counts: dict[str, int], sample_count: int) -> float | None:
    if sample_count <= 0 or not counts:
        return None
    return float(max(counts.values()) / sample_count)


def _fraction_payload(counts: dict[str, dict[str, int]], sample_count: int) -> dict[str, float | None]:
    return {
        "max_stack_fraction": _max_fraction(counts["stack_template"], sample_count),
        "max_source_fraction": _max_fraction(counts["source_category"], sample_count),
        "max_k_region_fraction": _max_fraction(counts["k_region_mode"], sample_count),
        "max_bc_fraction": _max_fraction(counts["bc_category"], sample_count),
        "max_power_scale_fraction": _max_fraction(counts["power_scale_category"], sample_count),
    }


def _batch_record(index: int, group: dict[str, Any], example_by_id: dict[str, Any], batch_shape_signature) -> dict[str, Any]:
    sample_ids = tuple(str(sample_id) for sample_id in group["sample_ids"])
    signature = batch_shape_signature(group)
    sample_count = int(signature.get("sample_count") or len(sample_ids))
    counts = _category_counts(sample_ids, example_by_id)
    return {
        "batch_index": int(index),
        "group_name": str(group["name"]),
        "sample_count": sample_count,
        "sample_ids": list(sample_ids),
        "category_counts": counts,
        **_fraction_payload(counts, sample_count),
        "target_delta": _target_delta_stats(group),
        "graph_shape": signature,
        "graph_total_edges": signature.get("total_edges"),
    }


def _sample_batch_record(index: int, sample_ids: list[str], example_by_id: dict[str, Any]) -> dict[str, Any]:
    sample_ids_tuple = tuple(str(sample_id) for sample_id in sample_ids)
    sample_count = len(sample_ids_tuple)
    counts = _category_counts(sample_ids_tuple, example_by_id)
    return {
        "batch_index": int(index),
        "group_name": f"sample_shuffle_batch_{index:04d}_B{sample_count}",
        "sample_count": int(sample_count),
        "sample_ids": list(sample_ids_tuple),
        "category_counts": counts,
        **_fraction_payload(counts, sample_count),
        "target_delta": _target_delta_stats_from_examples(sample_ids_tuple, example_by_id),
        "graph_shape": None,
        "graph_total_edges": None,
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Heat3D v2 Batch Composition Audit",
        "",
        f"- config: `{payload['config_path']}`",
        f"- batch plan: `{payload['batch_plan']}`",
        f"- train samples: {payload['train_sample_count']}",
        f"- epoch: {payload['epoch']}",
        f"- batch size: {payload['batch_size']}",
        f"- batch build seed: {payload['batch_build_seed']}",
        f"- shuffle_train_batches: {payload['shuffle_train_batches']}",
        f"- batch_order_seed: {payload['batch_order_seed']}",
        f"- graph_seed: {payload['graph_seed']}",
        f"- metadata_only: {payload['metadata_only']}",
        f"- batch_count: {payload['batch_count']}",
        "",
        "| batch | samples | total_edges | max stack | max source | max k_region | max bc | max power | stack_template | source_category | k_region_mode | bc_category | target_delta_mean | target_delta_std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for batch in payload["batches"]:
        counts = batch["category_counts"]
        target = batch["target_delta"]
        lines.append(
            "| {idx} | {samples} | {edges} | {max_stack} | {max_source} | {max_k_region} | {max_bc} | {max_power} | {stack} | {source} | {k_region} | {bc} | {mean} | {std} |".format(
                idx=batch["batch_index"],
                samples=batch["sample_count"],
                edges=batch["graph_total_edges"],
                max_stack=_format_fraction(batch["max_stack_fraction"]),
                max_source=_format_fraction(batch["max_source_fraction"]),
                max_k_region=_format_fraction(batch["max_k_region_fraction"]),
                max_bc=_format_fraction(batch["max_bc_fraction"]),
                max_power=_format_fraction(batch["max_power_scale_fraction"]),
                stack=_format_counts(counts["stack_template"]),
                source=_format_counts(counts["source_category"]),
                k_region=_format_counts(counts["k_region_mode"]),
                bc=_format_counts(counts["bc_category"]),
                mean=_format_optional_float(target.get("mean")),
                std=_format_optional_float(target.get("std")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in counts.items())


def _format_fraction(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6e}"


def main() -> int:
    args = parse_args()
    if args.epoch < 1:
        raise ValueError("--epoch must be >= 1")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batch_build_seed is not None and args.batch_build_seed < 0:
        raise ValueError("--batch-build-seed must be >= 0")
    if args.metadata_only and args.batch_plan != "sample_shuffle":
        raise ValueError("--metadata-only/--no-graph-build is only supported for --batch-plan sample_shuffle")

    config = load_v2_config(args.config)
    dataset_config = config["dataset"]
    run_config = config["run"]
    split_map_path = dataset_config.get("split_map_path")
    config_path = args.config
    sample_root = _sample_root(_resolve_path(dataset_config["subset_path"], config_path=config_path))
    resolved_split_map_path = _resolve_path(split_map_path, config_path=config_path) if split_map_path else None
    split_ids, split_source, primary_validation_split, stress_validation_split = _resolve_training_splits(
        resolved_split_map_path,
    )
    train_ids = split_ids["train"]
    graph_config = _config_graph(config)
    graph_seed = _optimizer_seed(config, "graph_seed", 0)
    batch_order_seed = _optimizer_seed(config, "batch_order_seed", 0)
    batch_build_seed = batch_order_seed if args.batch_build_seed is None else int(args.batch_build_seed)
    batch_size = args.batch_size if args.batch_size is not None else run_config.get("batch_size")
    batch_size = None if batch_size is None else int(batch_size)
    if batch_size is None:
        raise ValueError("batch composition audit requires run.batch_size or --batch-size")
    drop_last = bool(run_config.get("drop_last", False))
    shuffle = bool(run_config.get("shuffle_train_batches", False))

    train_examples = []
    example_by_id: dict[str, Any] = {}
    if not args.metadata_only:
        from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: WPS433

        dataset = Heat3DV1NativeSupervisedDataset(
            sample_root,
            k_encoding_mode=dataset_config.get("k_encoding_mode", "diag3"),
            boundary_mask_fallback=bool(dataset_config.get("boundary_mask_fallback", True)),
        )
        index_by_id = dataset.sample_index_by_id()
        missing = [sample_id for sample_id in train_ids if sample_id not in index_by_id]
        if missing:
            raise FileNotFoundError(f"Dataset loader did not expose train samples: {missing}")
        train_examples = [dataset[index_by_id[sample_id]] for sample_id in train_ids]
        example_by_id = {example.sample_id: example for example in train_examples}

    if args.batch_plan == "current_graph_shape":
        from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: WPS433
        from run_heat3d_v1_medium_controlled_training_export import (  # noqa: WPS433
            _batch_shape_signature,
            _epoch_train_groups,
            _make_groups_with_progress,
            _train_only_stats,
        )

        stats = _train_only_stats(train_examples)
        builder = Heat3DGraphBuilder(**graph_config)
        groups = _make_groups_with_progress(
            train_examples,
            stats,
            builder,
            "train",
            False,
            "basic",
            graph_seed,
            batch_size=batch_size,
            drop_last=drop_last,
            profile_counts=None,
        )
        ordered_groups = _epoch_train_groups(groups, epoch=args.epoch, seed=batch_build_seed, shuffle=shuffle)
        batches = [
            _batch_record(index, group, example_by_id, _batch_shape_signature)
            for index, group in enumerate(ordered_groups, start=1)
        ]
    else:
        ids = list(train_ids)
        rng = np.random.default_rng(batch_build_seed)
        ids = [ids[int(index)] for index in rng.permutation(len(ids))]
        chunks = _chunk_items(ids, batch_size=batch_size, drop_last=drop_last)
        batches = [
            _sample_batch_record(index, chunk, example_by_id)
            for index, chunk in enumerate(chunks, start=1)
        ]
    payload = {
        "config_path": str(args.config),
        "subset": str(sample_root),
        "batch_plan": args.batch_plan,
        "split_source": split_source,
        "primary_validation_split": primary_validation_split,
        "stress_validation_split": stress_validation_split,
        "train_sample_count": int(len(train_ids)),
        "batch_count": int(len(batches)),
        "epoch": int(args.epoch),
        "batch_size": batch_size,
        "batch_build_seed": int(batch_build_seed),
        "drop_last": drop_last,
        "shuffle_train_batches": shuffle,
        "batch_order_seed": int(batch_order_seed),
        "graph_seed": int(graph_seed),
        "metadata_only": bool(args.metadata_only),
        "graph_config": graph_config,
        "batches": batches,
    }

    output_json, output_md = _output_paths(args.config, args.output_json, args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_md(output_md, payload)
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
