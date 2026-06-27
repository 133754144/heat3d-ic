"""Split-map utilities for Heat3D V4 train-consistent audits."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CATEGORY_KEYS = (
    "source_category",
    "power_scale_category",
    "bc_category",
    "k_field_mode",
    "k_region_mode",
    "stack_template",
)
REGULAR_SPLITS = {"train", "valid_iid", "test_id"}
STRESS_HOLDOUT_SPLIT = "valid_stress"
OOD_HOLDOUT_SPLITS = {"test_ood_bc", "test_ood_stack", "test_ood_combined"}
LEGAL_SPLITS = REGULAR_SPLITS | {STRESS_HOLDOUT_SPLIT} | OOD_HOLDOUT_SPLITS


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_sample_split_map(path: str | Path | None) -> dict[str, str]:
    """Load a sample_id -> split map from either raw mapping or split-map JSON."""

    if path is None:
        return {}
    loaded = load_json(path)
    mapping = loaded.get("sample_splits", loaded) if isinstance(loaded, Mapping) else None
    if not isinstance(mapping, Mapping):
        raise ValueError(f"split_map must be a mapping or contain sample_splits: {path}")

    result: dict[str, str] = {}
    for sample_id, split in mapping.items():
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"split_map contains invalid sample_id: {sample_id!r}")
        if not isinstance(split, str) or not split:
            raise ValueError(f"split_map contains invalid split for {sample_id!r}: {split!r}")
        result[sample_id] = split
    return dict(sorted(result.items()))


def split_ids_from_sample_splits(sample_splits: Mapping[str, str]) -> dict[str, list[str]]:
    split_ids: dict[str, list[str]] = defaultdict(list)
    for sample_id, split in sample_splits.items():
        split_ids[str(split)].append(str(sample_id))
    return {split: sorted(ids) for split, ids in sorted(split_ids.items())}


def resolve_sample_split(
    sample_id: str,
    sample_meta: Mapping[str, Any] | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    split_map: Mapping[str, str] | None = None,
    default: str = "unknown",
) -> str:
    """Resolve sample split from explicit split_map first, then metadata."""

    if split_map and sample_id in split_map:
        return str(split_map[sample_id])
    for source in (metadata, sample_meta):
        if isinstance(source, Mapping) and source.get("split") not in (None, ""):
            return str(source["split"])
    return default


def split_source_label(split_map: Mapping[str, str] | None) -> str:
    return "split_map" if split_map else "sample_meta"


def sample_root(path: str | Path) -> Path:
    root = Path(path)
    samples = root / "samples"
    return samples if samples.is_dir() else root


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = load_json(path)
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _plan(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    generation_config = meta.get("generation_config")
    if isinstance(generation_config, Mapping):
        sample_plan = generation_config.get("sample_plan")
        if isinstance(sample_plan, Mapping):
            return sample_plan
    return {}


def _field(record: Mapping[str, Any], key: str, default: str = "missing") -> str:
    value = record.get(key)
    if value not in (None, ""):
        return str(value)
    return default


def _record_value(merged: Mapping[str, Any], sample_meta: Mapping[str, Any], key: str) -> str:
    if key == "source_category":
        for candidate in ("source_category", "source_pattern_tag"):
            value = merged.get(candidate)
            if value not in (None, ""):
                return str(value)
    if key == "stack_template":
        stack = sample_meta.get("stack")
        if isinstance(stack, Mapping) and stack.get("stack_template") not in (None, ""):
            return str(stack["stack_template"])
    value = merged.get(key)
    if value not in (None, ""):
        return str(value)
    plan = _plan(sample_meta)
    if key == "source_category":
        value = plan.get("source_category") or plan.get("source_pattern_tag")
    else:
        value = plan.get(key)
    return str(value) if value not in (None, "") else "missing"


def read_sample_records(subset: str | Path) -> list[dict[str, Any]]:
    """Read lightweight sample metadata records from a Heat3D subset."""

    root = sample_root(subset)
    if not root.is_dir():
        raise FileNotFoundError(f"missing sample root: {root}")

    records: list[dict[str, Any]] = []
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        sample_meta = _read_optional_json(sample_dir / "sample_meta.json")
        metadata = _read_optional_json(sample_dir / "metadata.json")
        label_meta = _read_optional_json(sample_dir / "label_meta.json")
        merged = {**sample_meta, **metadata, **label_meta}
        sample_id = str(metadata.get("sample_id") or sample_meta.get("sample_id") or sample_dir.name)
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "sample_dir_name": sample_dir.name,
            "sample_meta_split": str(sample_meta.get("split", "unknown")),
        }
        for key in DEFAULT_CATEGORY_KEYS:
            record[key] = _record_value(merged, sample_meta, key)
        records.append(record)
    return records


def load_sample_records_json(path: str | Path) -> list[dict[str, Any]]:
    loaded = load_json(path)
    records = loaded.get("records", loaded) if isinstance(loaded, Mapping) else loaded
    if not isinstance(records, list):
        raise ValueError(f"sample records JSON must be a list or contain records: {path}")
    result = []
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError(f"sample record must be an object, got {type(item).__name__}")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"sample record missing sample_id: {item!r}")
        result.append(dict(item))
    return result


def _hash_key(seed: int, sample_id: str, salt: str = "") -> str:
    return hashlib.sha256(f"{seed}:{salt}:{sample_id}".encode("utf-8")).hexdigest()


def _stratum(record: Mapping[str, Any], category_keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(_field(record, key) for key in category_keys)


def _split_counts(sample_splits: Mapping[str, str]) -> dict[str, int]:
    counts = Counter(sample_splits.values())
    return {split: int(counts.get(split, 0)) for split in sorted(set(counts) | LEGAL_SPLITS)}


def _category_counts(
    records: Sequence[Mapping[str, Any]],
    sample_splits: Mapping[str, str],
    category_keys: Sequence[str],
) -> dict[str, dict[str, dict[str, int]]]:
    by_id = {str(record["sample_id"]): record for record in records}
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for key in category_keys:
        buckets: dict[str, Counter[str]] = defaultdict(Counter)
        for sample_id, split in sample_splits.items():
            record = by_id.get(sample_id)
            if record is None:
                continue
            buckets[split][_field(record, key)] += 1
        summary[key] = {
            split: {value: int(count) for value, count in sorted(counter.items())}
            for split, counter in sorted(buckets.items())
        }
    return summary


def _old_split_counts(old_splits: Mapping[str, str]) -> dict[str, int]:
    counts = Counter(old_splits.values())
    return {split: int(count) for split, count in sorted(counts.items())}


def _allocate_proportional_counts(
    capacities: Mapping[tuple[str, ...], int],
    target: int,
) -> dict[tuple[str, ...], int]:
    total = sum(capacities.values())
    if target < 0 or target > total:
        raise ValueError(f"target={target} is outside available capacity={total}")
    if total == 0:
        return {key: 0 for key in capacities}

    expected = {key: (count * target / total) for key, count in capacities.items()}
    allocated = {
        key: min(int(expected[key]), int(capacities[key]))
        for key in capacities
    }
    remaining = target - sum(allocated.values())
    order = sorted(
        capacities,
        key=lambda key: (
            expected[key] - int(expected[key]),
            capacities[key],
            repr(key),
        ),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key in order:
            if remaining <= 0:
                break
            if allocated[key] >= capacities[key]:
                continue
            allocated[key] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise ValueError("could not allocate requested proportional counts")
    return allocated


def build_train_consistent_split_map(
    *,
    records: Sequence[Mapping[str, Any]],
    old_sample_splits: Mapping[str, str],
    seed: int = 0,
    train_count: int = 704,
    valid_iid_count: int = 84,
    test_id_count: int = 84,
    category_keys: Sequence[str] = DEFAULT_CATEGORY_KEYS,
    subset_path: str = "",
) -> dict[str, Any]:
    """Build the V4 train-consistent IID split map from an old split map."""

    record_by_id = {str(record["sample_id"]): dict(record) for record in records}
    if len(record_by_id) != len(records):
        raise ValueError("duplicate sample_id in sample records")
    missing_records = sorted(set(old_sample_splits) - set(record_by_id))
    if missing_records:
        raise ValueError(f"sample records missing ids from old split map: {missing_records[:5]}")

    regular_ids = sorted(
        sample_id for sample_id, split in old_sample_splits.items() if split in REGULAR_SPLITS
    )
    stress_ids = sorted(
        sample_id for sample_id, split in old_sample_splits.items() if split == STRESS_HOLDOUT_SPLIT
    )
    ood_ids = sorted(
        sample_id for sample_id, split in old_sample_splits.items() if split in OOD_HOLDOUT_SPLITS
    )
    unknown_old = sorted(set(old_sample_splits.values()) - LEGAL_SPLITS)
    if unknown_old:
        raise ValueError(f"old split map contains unknown split names: {unknown_old}")
    if train_count + valid_iid_count + test_id_count != len(regular_ids):
        raise ValueError(
            "requested regular split counts must sum to regular_pool size: "
            f"{train_count}+{valid_iid_count}+{test_id_count}!={len(regular_ids)}"
        )

    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for sample_id in regular_ids:
        strata[_stratum(record_by_id[sample_id], category_keys)].append(sample_id)
    for ids in strata.values():
        ids.sort(key=lambda sample_id: _hash_key(seed, sample_id, "regular_pool"))

    capacities = {key: len(ids) for key, ids in strata.items()}
    valid_alloc = _allocate_proportional_counts(capacities, valid_iid_count)
    remaining_after_valid = {
        key: capacities[key] - valid_alloc[key]
        for key in capacities
    }
    test_alloc = _allocate_proportional_counts(remaining_after_valid, test_id_count)

    sample_splits: dict[str, str] = {}
    for sample_id in stress_ids:
        sample_splits[sample_id] = STRESS_HOLDOUT_SPLIT
    for sample_id in ood_ids:
        sample_splits[sample_id] = str(old_sample_splits[sample_id])

    for key, ids in sorted(strata.items(), key=lambda item: repr(item[0])):
        valid_n = valid_alloc[key]
        test_n = test_alloc[key]
        valid_ids = ids[:valid_n]
        test_ids = ids[valid_n : valid_n + test_n]
        train_ids = ids[valid_n + test_n :]
        for sample_id in valid_ids:
            sample_splits[sample_id] = "valid_iid"
        for sample_id in test_ids:
            sample_splits[sample_id] = "test_id"
        for sample_id in train_ids:
            sample_splits[sample_id] = "train"

    if set(sample_splits) != set(old_sample_splits):
        missing = sorted(set(old_sample_splits) - set(sample_splits))
        extra = sorted(set(sample_splits) - set(old_sample_splits))
        raise ValueError(f"split map sample mismatch: missing={missing[:5]} extra={extra[:5]}")

    return {
        "schema_version": "heat3d_v4_train_consistent_split_map_v0",
        "dataset_name": "medium1024_gapA_full1024_v2",
        "subset_path": subset_path,
        "seed": int(seed),
        "strategy": "train_consistent_regular_pool_proportional_iid",
        "source_split_map": "configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json",
        "regular_pool_rule": "old train + valid_iid + test_id",
        "stress_holdout_rule": "old valid_stress retained",
        "ood_holdout_rule": "old test_ood_bc + test_ood_stack + test_ood_combined retained",
        "regular_targets": {
            "train": int(train_count),
            "valid_iid_proportional": int(valid_iid_count),
            "test_iid_proportional_as_test_id": int(test_id_count),
        },
        "category_keys": list(category_keys),
        "notes": [
            "Does not copy arrays or modify sample_meta.json.",
            "valid_iid and test_id are proportional holdouts from the old regular pool.",
            "The new test_id split is the train-consistent IID test split for V4 P1 audits.",
            "valid/test are not hard-case upsampled; old valid_stress and OOD splits remain holdouts.",
        ],
        "legal_splits": sorted(LEGAL_SPLITS),
        "sample_splits": dict(sorted(sample_splits.items())),
        "split_counts": _split_counts(sample_splits),
        "old_split_counts": _old_split_counts(old_sample_splits),
        "regular_pool_counts": {
            "regular_pool": len(regular_ids),
            "stress_holdout": len(stress_ids),
            "ood_holdout": len(ood_ids),
        },
        "category_counts": _category_counts(records, sample_splits, category_keys),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-records")
    export_parser.add_argument("--subset", type=Path, required=True)
    export_parser.add_argument("--output-json", type=Path, required=True)

    build_parser = subparsers.add_parser("build-train-consistent")
    source_group = build_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--subset", type=Path)
    source_group.add_argument("--sample-records-json", type=Path)
    build_parser.add_argument("--source-split-map", type=Path, required=True)
    build_parser.add_argument("--output-json", type=Path, required=True)
    build_parser.add_argument("--seed", type=int, default=0)
    build_parser.add_argument("--train-count", type=int, default=704)
    build_parser.add_argument("--valid-iid-count", type=int, default=84)
    build_parser.add_argument("--test-id-count", type=int, default=84)
    build_parser.add_argument(
        "--subset-path",
        default="data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "export-records":
        records = read_sample_records(args.subset)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"records": records}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote sample records: {args.output_json} records={len(records)}")
        return 0

    if args.command == "build-train-consistent":
        records = (
            read_sample_records(args.subset)
            if args.subset is not None
            else load_sample_records_json(args.sample_records_json)
        )
        old_splits = load_sample_split_map(args.source_split_map)
        payload = build_train_consistent_split_map(
            records=records,
            old_sample_splits=old_splits,
            seed=int(args.seed),
            train_count=int(args.train_count),
            valid_iid_count=int(args.valid_iid_count),
            test_id_count=int(args.test_id_count),
            subset_path=str(args.subset_path),
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote split map: {args.output_json}")
        print(f"split_counts: {payload['split_counts']}")
        return 0

    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
