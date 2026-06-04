#!/usr/bin/env python3
"""Build a worktree-local stratified split map for Heat3D v2 medium1024."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_SUBSET = Path(
    "data/heat3d-thermal-simulation/subsets/"
    "v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2"
)
DEFAULT_OUTPUT = Path("configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json")
DEFAULT_SEED = 0
LEGAL_SPLITS = {
    "train",
    "valid_iid",
    "valid_stress",
    "test_id",
    "test_ood_bc",
    "test_ood_stack",
    "test_ood_combined",
}
CATEGORY_KEYS = (
    "source_category",
    "power_scale_category",
    "bc_category",
    "k_field_mode",
    "k_region_mode",
    "stack_template",
)
HELD_OUT_BC = {
    "held_out_top_h_candidate",
    "very_low_top_h_candidate",
    "very_high_top_h_candidate",
}
HELD_OUT_STACK = {"held_out_interposer_like_candidate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stratified split-map JSON for an existing Heat3D subset."
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--valid-iid-count", type=int, default=104)
    parser.add_argument("--valid-stress-count", type=int, default=88)
    parser.add_argument("--test-id-count", type=int, default=64)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    return loaded if isinstance(loaded, dict) else {}


def _sample_root(subset: Path) -> Path:
    samples = subset / "samples"
    return samples if samples.is_dir() else subset


def _field(sample: dict[str, Any], key: str, default: str = "missing") -> str:
    value = sample.get(key)
    if value in (None, ""):
        return default
    return str(value)


def _read_samples(subset: Path) -> list[dict[str, Any]]:
    root = _sample_root(subset)
    if not root.exists():
        raise FileNotFoundError(f"Missing subset samples directory: {root}")

    samples: list[dict[str, Any]] = []
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        merged: dict[str, Any] = {}
        for name in ("sample_meta.json", "metadata.json", "label_meta.json"):
            data = _load_json(sample_dir / name)
            for key, value in data.items():
                merged.setdefault(key, value)
        sample_id = str(merged.get("sample_id") or sample_dir.name)
        merged["sample_id"] = sample_id
        merged["sample_dir_name"] = sample_dir.name
        merged["old_split"] = str(merged.get("split", "missing"))
        if "source_category" not in merged and merged.get("source_pattern_tag"):
            merged["source_category"] = merged["source_pattern_tag"]
        samples.append(merged)
    return samples


def _hash_key(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def _is_held_out_bc(sample: dict[str, Any]) -> bool:
    return _field(sample, "bc_category") in HELD_OUT_BC


def _is_held_out_stack(sample: dict[str, Any]) -> bool:
    return _field(sample, "stack_template") in HELD_OUT_STACK


def _ood_split(sample: dict[str, Any]) -> str | None:
    has_bc = _is_held_out_bc(sample)
    has_stack = _is_held_out_stack(sample)
    if has_bc and has_stack:
        return "test_ood_combined"
    if has_bc:
        return "test_ood_bc"
    if has_stack:
        return "test_ood_stack"
    return None


def _stress_score(sample: dict[str, Any]) -> int:
    score = 0
    if _field(sample, "power_scale_category") in {"low_power", "high_dynamic_range"}:
        score += 1
    if _field(sample, "source_category") in {
        "low_power_near_zero_background_cases",
        "high_dynamic_range_power_cases",
    }:
        score += 1
    if _field(sample, "bc_category") == "high_top_h":
        score += 1
    if _field(sample, "k_field_mode") == "diag3":
        score += 1
    if _field(sample, "k_region_mode") in {
        "low_k_barrier_or_TIM_variation",
        "high_contrast_interface_k",
    }:
        score += 1
    return score


def _diverse_select(
    candidates: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    prefer_stress: bool = False,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count >= len(candidates):
        return sorted(candidates, key=lambda sample: _hash_key(_field(sample, "sample_id"), seed))

    remaining = sorted(candidates, key=lambda sample: _hash_key(_field(sample, "sample_id"), seed))
    selected: list[dict[str, Any]] = []
    category_counts: dict[str, Counter[str]] = {key: Counter() for key in CATEGORY_KEYS}
    while len(selected) < count:
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                tuple(category_counts[key][_field(remaining[index], key)] for key in CATEGORY_KEYS),
                -_stress_score(remaining[index]) if prefer_stress else _stress_score(remaining[index]),
                _hash_key(_field(remaining[index], "sample_id"), seed),
            ),
        )
        sample = remaining.pop(best_index)
        selected.append(sample)
        for key in CATEGORY_KEYS:
            category_counts[key][_field(sample, key)] += 1
    return selected


def _split_counts_from_map(sample_splits: dict[str, str]) -> dict[str, int]:
    counts = Counter(sample_splits.values())
    return {split: int(counts.get(split, 0)) for split in sorted(LEGAL_SPLITS)}


def _category_counts(samples: list[dict[str, Any]], sample_splits: dict[str, str]) -> dict[str, dict[str, dict[str, int]]]:
    sample_by_id = {_field(sample, "sample_id"): sample for sample in samples}
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for key in CATEGORY_KEYS:
        by_split: dict[str, Counter[str]] = defaultdict(Counter)
        for sample_id, split in sample_splits.items():
            sample = sample_by_id[sample_id]
            by_split[split][_field(sample, key)] += 1
        summary[key] = {
            split: {value: int(count) for value, count in sorted(counter.items())}
            for split, counter in sorted(by_split.items())
        }
    return summary


def _old_split_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(_field(sample, "old_split") for sample in samples)
    return {split: int(count) for split, count in sorted(counts.items())}


def _build_split_map(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    sample_ids = [_field(sample, "sample_id") for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_id found in subset metadata")

    sample_splits: dict[str, str] = {}
    regular: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = _field(sample, "sample_id")
        ood = _ood_split(sample)
        if ood is None:
            regular.append(sample)
        else:
            sample_splits[sample_id] = ood

    regular_total = len(regular)
    valid_iid_target = min(int(args.valid_iid_count), max(0, regular_total))
    valid_stress_target = min(int(args.valid_stress_count), max(0, regular_total - valid_iid_target))
    test_id_target = min(
        int(args.test_id_count),
        max(0, regular_total - valid_iid_target - valid_stress_target),
    )
    if args.train_count is not None:
        requested = int(args.train_count)
        remaining_targets = valid_iid_target + valid_stress_target + test_id_target
        if requested + remaining_targets > regular_total:
            raise ValueError("Requested split counts exceed regular sample count")

    stress_pool = [sample for sample in regular if _stress_score(sample) > 0]
    valid_stress = _diverse_select(
        stress_pool,
        valid_stress_target,
        seed=int(args.seed),
        prefer_stress=True,
    )
    taken = {_field(sample, "sample_id") for sample in valid_stress}

    remaining = [sample for sample in regular if _field(sample, "sample_id") not in taken]
    valid_iid = _diverse_select(remaining, valid_iid_target, seed=int(args.seed))
    taken.update(_field(sample, "sample_id") for sample in valid_iid)

    remaining = [sample for sample in regular if _field(sample, "sample_id") not in taken]
    test_id = _diverse_select(remaining, test_id_target, seed=int(args.seed) + 17)
    taken.update(_field(sample, "sample_id") for sample in test_id)

    for sample in valid_stress:
        sample_splits[_field(sample, "sample_id")] = "valid_stress"
    for sample in valid_iid:
        sample_splits[_field(sample, "sample_id")] = "valid_iid"
    for sample in test_id:
        sample_splits[_field(sample, "sample_id")] = "test_id"
    for sample in regular:
        sample_id = _field(sample, "sample_id")
        sample_splits.setdefault(sample_id, "train")

    if set(sample_splits) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(sample_splits))
        extra = sorted(set(sample_splits) - set(sample_ids))
        raise ValueError(f"Split map sample mismatch: missing={missing[:5]} extra={extra[:5]}")
    illegal = sorted(set(sample_splits.values()) - LEGAL_SPLITS)
    if illegal:
        raise ValueError(f"Illegal split names generated: {illegal}")

    return {
        "schema_version": "heat3d_v2_split_map_v0",
        "dataset_name": "medium1024_gapA_full1024_v2",
        "subset_path": str(args.subset),
        "seed": int(args.seed),
        "strategy": "metadata_stratified_regular_plus_stress_and_preserved_ood",
        "notes": [
            "Does not copy arrays or modify sample_meta.json.",
            "valid_iid is drawn from the regular non-held-out pool with category diversity.",
            "valid_stress keeps stress cases separate while leaving stress coverage in train.",
            "Held-out BC/stack candidates remain test-only splits.",
        ],
        "legal_splits": sorted(LEGAL_SPLITS),
        "sample_splits": dict(sorted(sample_splits.items())),
        "split_counts": _split_counts_from_map(sample_splits),
        "old_split_counts": _old_split_counts(samples),
        "category_counts": _category_counts(samples, sample_splits),
    }


def _validate_existing(path: Path) -> dict[str, Any]:
    loaded = _load_json(path)
    sample_splits = loaded.get("sample_splits")
    if not isinstance(sample_splits, dict) or not sample_splits:
        raise ValueError(f"{path}: missing non-empty sample_splits mapping")
    illegal = sorted(set(str(value) for value in sample_splits.values()) - LEGAL_SPLITS)
    if illegal:
        raise ValueError(f"{path}: illegal split names: {illegal}")
    return loaded


def main() -> int:
    args = parse_args()
    try:
        samples = _read_samples(args.subset)
    except FileNotFoundError:
        if args.output_json.exists():
            existing = _validate_existing(args.output_json)
            print(
                "subset missing locally; existing stratified split map validated: "
                f"{args.output_json} sample_count={len(existing['sample_splits'])}"
            )
            print("Heat3D v2 stratified split builder passed.")
            return 0
        raise

    payload = _build_split_map(samples, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"split_counts: {payload['split_counts']}")
    print("Heat3D v2 stratified split builder passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
