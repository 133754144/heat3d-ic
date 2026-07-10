#!/usr/bin/env python3
"""Generate and solve local V4 P3c random-block datasets.

This script writes only user-scoped P3c dataset and audit directories, calls
the V4 reference solver for labels, and never starts model training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heat3d_v4_p3c_dryrun_generator import (  # noqa: E402
    DEFAULT_REGISTRY,
    PLANNED_SAMPLE_FILES,
    SMOKE16_DATASET_DIR,
    SMOKE16_OUTPUT_DIR,
    SMOKE16_SAMPLE_COUNT,
    SMOKE16_SEED,
    build_smoke16_write_plan,
    control_volume_weights_for_domain,
    generate_dryrun_batch,
    load_registry,
    materialize_scene_arrays,
)
from rigno.heat3d_v4_reference_solver import (  # noqa: E402
    SolverOptions,
    extract_problem_from_arrays,
    solve_temperature_from_problem,
)


DELTA_T_BINS = (
    ("reject_low", None, 0.02),
    ("low", 0.02, 0.2),
    ("nominal", 0.2, 2.0),
    ("hard", 2.0, 8.0),
    ("review_high", 8.0, 15.0),
    ("reject_high", 15.0, None),
)
SPLIT_AUDIT_FIELDS = (
    "k_mode",
    "diag3_policy",
    "q_family",
    "cooling_regime",
    "DeltaT_bin",
    "high_deltaT_triage",
    "dataset_action",
)
SPLIT_POLICY_NAME = "deterministic_stratified_random_v0"
DEFAULT_TRAIN_FRACTION = 0.75
P3B_LITE_SUBSET_POLICY_NAME = "p3b_lite_validation_subset_v0"
P3B_LITE_SUBSET_SIZE = 64
P3B_LITE_AUDIT_FIELDS = (
    "qc_class",
    "k_mode",
    "diag3_policy",
    "q_family",
    "cooling_regime",
    "DeltaT_bin",
    "high_deltaT_triage",
)
QC_POWER_ERROR_TOL_W = 1.0e-10
QC_HIGH_DELTAT_K = 15.0
QC_HIGH_POWER_W = 2.0
ACCEPTED_QC_CLASSES = {"clean_keep", "physical_hard_keep", "review_hold"}


def _target_split_counts(sample_count: int, train_fraction: float) -> tuple[int, int]:
    if sample_count < 2:
        return sample_count, 0
    train_count = int(round(float(sample_count) * float(train_fraction)))
    train_count = min(max(1, train_count), sample_count - 1)
    return train_count, sample_count - train_count


def _stable_unit_hash(seed: int, sample_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{seed}:{salt}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _desired_test_counts(
    samples: list[dict[str, Any]],
    *,
    test_count: int,
) -> dict[str, dict[str, int]]:
    if not samples:
        return {}
    test_fraction = test_count / float(len(samples))
    desired: dict[str, dict[str, int]] = {}
    for field in SPLIT_AUDIT_FIELDS:
        counts = Counter(str(sample.get(field, "missing")) for sample in samples)
        desired[field] = {}
        for value, count in counts.items():
            if count <= 1:
                desired[field][value] = 0
            else:
                desired[field][value] = min(count - 1, max(1, int(round(count * test_fraction))))
    return desired


def assign_stratified_splits(
    samples: list[dict[str, Any]],
    *,
    seed: int,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> dict[str, str]:
    """Assign train/test using deterministic random tie-breaks and audit fields."""

    train_count, test_count = _target_split_counts(len(samples), train_fraction)
    if test_count == 0:
        return {sample["sample_id"]: "train" for sample in samples}
    desired = _desired_test_counts(samples, test_count=test_count)
    field_counts = {
        field: Counter(str(sample.get(field, "missing")) for sample in samples)
        for field in SPLIT_AUDIT_FIELDS
    }
    selected_test: set[str] = set()
    selected_counts = {field: Counter() for field in SPLIT_AUDIT_FIELDS}
    remaining = {sample["sample_id"]: sample for sample in samples}

    while len(selected_test) < test_count:
        unsatisfied = []
        for field in SPLIT_AUDIT_FIELDS:
            for value, target in desired.get(field, {}).items():
                current = selected_counts[field][value]
                if target > current:
                    unsatisfied.append(
                        {
                            "field": field,
                            "value": value,
                            "target": target,
                            "current": current,
                            "total": field_counts[field][value],
                        }
                    )
        if not unsatisfied:
            break
        best_id = None
        best_score = None
        for sample_id, sample in remaining.items():
            coverage_count = 0
            weighted_coverage = 0.0
            singleton_penalty = 0
            for item in unsatisfied:
                field = item["field"]
                value = str(sample.get(field, "missing"))
                if value == item["value"]:
                    coverage_count += 1
                    weighted_coverage += 1.0 / float(max(1, item["total"]))
            if coverage_count == 0:
                continue
            for field in SPLIT_AUDIT_FIELDS:
                value = str(sample.get(field, "missing"))
                if field_counts[field][value] <= 1:
                    singleton_penalty += 1
            score = (
                float(coverage_count),
                weighted_coverage,
                -float(singleton_penalty),
                _stable_unit_hash(seed, sample_id, "strata_coverage"),
            )
            if best_score is None or score > best_score:
                best_id = sample_id
                best_score = score
        if best_id is None:
            break
        selected_test.add(best_id)
        chosen = remaining.pop(best_id)
        for field in SPLIT_AUDIT_FIELDS:
            selected_counts[field][str(chosen.get(field, "missing"))] += 1

    while len(selected_test) < test_count:
        best_id: str | None = None
        best_score: tuple[float, ...] | None = None
        for sample_id, sample in remaining.items():
            need_count = 0
            need_gap = 0
            singleton_penalty = 0
            for field in SPLIT_AUDIT_FIELDS:
                value = str(sample.get(field, "missing"))
                target = desired.get(field, {}).get(value, 0)
                current = selected_counts[field][value]
                gap = max(0, target - current)
                if gap:
                    need_count += 1
                    need_gap += gap
                if field_counts[field][value] <= 1:
                    singleton_penalty += 1
            score = (
                float(need_count),
                float(need_gap),
                -float(singleton_penalty),
                _stable_unit_hash(seed, sample_id, "split_tie_break"),
            )
            if best_score is None or score > best_score:
                best_id = sample_id
                best_score = score
        if best_id is None:
            break
        selected_test.add(best_id)
        chosen = remaining.pop(best_id)
        for field in SPLIT_AUDIT_FIELDS:
            selected_counts[field][str(chosen.get(field, "missing"))] += 1

    split_map = {
        sample["sample_id"]: ("test" if sample["sample_id"] in selected_test else "train")
        for sample in samples
    }
    if sum(1 for split in split_map.values() if split == "train") != train_count:
        raise RuntimeError("internal split count mismatch")
    return split_map


def build_split_audit(
    samples: list[dict[str, Any]],
    *,
    split_map: dict[str, str],
    seed: int,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> dict[str, Any]:
    split_counts = Counter(split_map.values())
    fields: dict[str, Any] = {}
    for field in SPLIT_AUDIT_FIELDS:
        total = Counter(str(sample.get(field, "missing")) for sample in samples)
        by_split = {
            split: Counter(
                str(sample.get(field, "missing"))
                for sample in samples
                if split_map.get(sample["sample_id"]) == split
            )
            for split in ("train", "test")
        }
        fields[field] = {
            "total": dict(sorted(total.items())),
            "by_split": {
                split: dict(sorted(counts.items()))
                for split, counts in by_split.items()
            },
            "missing_in_train": sorted(
                value for value in total if by_split["train"].get(value, 0) == 0
            ),
            "missing_in_test": sorted(
                value for value in total if by_split["test"].get(value, 0) == 0
            ),
        }
    return {
        "policy": SPLIT_POLICY_NAME,
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "sample_count": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "audit_fields": list(SPLIT_AUDIT_FIELDS),
        "fields": fields,
        "notes": [
            "Split is assigned after solver audit so DeltaT/action strata are available.",
            "Singleton categories cannot appear in both train and test; missing lists are expected for such categories.",
        ],
    }


def _field_distribution(
    samples: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    return {
        field: dict(
            sorted(Counter(str(sample.get(field, "missing")) for sample in samples).items())
        )
        for field in fields
    }


def select_p3b_lite_validation_subset(
    samples: list[dict[str, Any]],
    *,
    seed: int,
    subset_size: int = P3B_LITE_SUBSET_SIZE,
) -> dict[str, Any]:
    """Select a fixed audit subset from accepted samples without changing splits."""

    target_size = min(int(subset_size), len(samples))
    selected_ids: set[str] = set()
    by_id = {sample["sample_id"]: sample for sample in samples}

    def add_ordered(candidates: list[dict[str, Any]], salt: str, limit: int | None = None) -> None:
        remaining_slots = target_size - len(selected_ids)
        if remaining_slots <= 0:
            return
        cap = remaining_slots if limit is None else min(remaining_slots, limit)
        ordered = sorted(
            candidates,
            key=lambda sample: _stable_unit_hash(seed, sample["sample_id"], salt),
            reverse=True,
        )
        for sample in ordered[:cap]:
            selected_ids.add(sample["sample_id"])

    review_hold = [sample for sample in samples if sample.get("qc_class") == "review_hold"]
    physical_keep = [
        sample for sample in samples if sample.get("qc_class") == "physical_hard_keep"
    ]
    add_ordered(review_hold, "p3b_lite_review_hold")
    add_ordered(
        physical_keep,
        "p3b_lite_physical_hard_keep",
        limit=max(1, target_size // 4) if target_size else 0,
    )

    total_counts = {
        field: Counter(str(sample.get(field, "missing")) for sample in samples)
        for field in P3B_LITE_AUDIT_FIELDS
    }
    selected_counts = {
        field: Counter(
            str(by_id[sample_id].get(field, "missing")) for sample_id in selected_ids
        )
        for field in P3B_LITE_AUDIT_FIELDS
    }
    remaining = {
        sample["sample_id"]: sample
        for sample in samples
        if sample["sample_id"] not in selected_ids
    }
    while len(selected_ids) < target_size and remaining:
        missing_values = []
        for field in P3B_LITE_AUDIT_FIELDS:
            for value, count in total_counts[field].items():
                if count > 0 and selected_counts[field][value] == 0:
                    missing_values.append((field, value, count))
        if not missing_values:
            break
        best_id = None
        best_score = None
        for sample_id, sample in remaining.items():
            coverage_count = 0
            weighted_coverage = 0.0
            for field, value, count in missing_values:
                if str(sample.get(field, "missing")) == value:
                    coverage_count += 1
                    weighted_coverage += 1.0 / float(max(1, count))
            if coverage_count == 0:
                continue
            score = (
                float(coverage_count),
                weighted_coverage,
                _stable_unit_hash(seed, sample_id, "p3b_lite_coverage"),
            )
            if best_score is None or score > best_score:
                best_id = sample_id
                best_score = score
        if best_id is None:
            break
        selected_ids.add(best_id)
        chosen = remaining.pop(best_id)
        for field in P3B_LITE_AUDIT_FIELDS:
            selected_counts[field][str(chosen.get(field, "missing"))] += 1

    while len(selected_ids) < target_size and remaining:
        sample_id, sample = max(
            remaining.items(),
            key=lambda item: _stable_unit_hash(seed, item[0], "p3b_lite_fill"),
        )
        selected_ids.add(sample_id)
        remaining.pop(sample_id)
        for field in P3B_LITE_AUDIT_FIELDS:
            selected_counts[field][str(sample.get(field, "missing"))] += 1

    selected_samples = [by_id[sample_id] for sample_id in sorted(selected_ids)]
    selected_distribution = _field_distribution(
        selected_samples,
        fields=P3B_LITE_AUDIT_FIELDS,
    )
    total_distribution = _field_distribution(samples, fields=P3B_LITE_AUDIT_FIELDS)
    missing_in_subset = {
        field: sorted(
            value
            for value in total_distribution[field]
            if selected_distribution[field].get(value, 0) == 0
        )
        for field in P3B_LITE_AUDIT_FIELDS
    }
    return {
        "policy": P3B_LITE_SUBSET_POLICY_NAME,
        "seed": int(seed),
        "target_size": int(subset_size),
        "selected_count": len(selected_samples),
        "sample_ids": [sample["sample_id"] for sample in selected_samples],
        "selection_fields": list(P3B_LITE_AUDIT_FIELDS),
        "rules": [
            "Use accepted samples only.",
            "Include review_hold samples first, capped by subset size.",
            "Include deterministic representatives of physical_hard_keep samples, capped at one quarter of the subset.",
            "Greedily cover missing QC/k/q/cooling/DeltaT categories, then fill by stable hash.",
            "This subset is validation/audit support only; it is not a separate stress split or pass/fail gate.",
        ],
        "total_distribution": total_distribution,
        "selected_distribution": selected_distribution,
        "missing_in_subset": missing_in_subset,
    }


def build_review_closeout(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for sample in samples:
        if sample.get("qc_class", sample.get("dataset_action")) == "clean_keep":
            continue
        conclusion = sample.get("qc_class", "review")
        reasons = []
        if sample.get("DeltaT_bin") in {"review_high", "reject_high"}:
            reasons.append(
                "high_deltaT_peak_without_solver_or_boundary_failure"
            )
        if abs(float(sample.get("q_total_power_error_W", 0.0))) <= 1.0e-10:
            reasons.append("q_power_consistent")
        if float(sample.get("q_power_on_boundary_W", 0.0)) == 0.0:
            reasons.append("no_boundary_q_deposition")
        if float(sample.get("low_k_q_overlap_fraction", 0.0)) >= 0.5:
            reasons.append("substantial_low_k_source_overlap")
        if sample.get("cooling_regime") == "weak_effective_air":
            reasons.append("weak_cooling_regime")
        if float(sample.get("q_total_target_power_W", 0.0)) > 2.0:
            reasons.append("high_integrated_power_target")
        records.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample.get("split", "unassigned"),
                "conclusion": conclusion,
                "dataset_action": sample.get("dataset_action"),
                "qc_class": sample.get("qc_class"),
                "qc_physical_keep_reasons": sample.get("qc_physical_keep_reasons", []),
                "qc_review_reasons": sample.get("qc_review_reasons", []),
                "qc_reject_reasons": sample.get("qc_reject_reasons", []),
                "DeltaT_bin": sample.get("DeltaT_bin"),
                "DeltaT_peak_K": sample.get("DeltaT_peak_K"),
                "DeltaT_p95_K": sample.get("DeltaT_p95_K"),
                "q_family": sample.get("q_family"),
                "cooling_regime": sample.get("cooling_regime"),
                "q_total_target_power_W": sample.get("q_total_target_power_W"),
                "q_max_after_sum_W_m3": sample.get("q_max_after_sum_W_m3"),
                "low_k_q_overlap_fraction": sample.get("low_k_q_overlap_fraction"),
                "high_deltaT_triage": sample.get("high_deltaT_triage"),
                "solver_status": sample.get("solver_status"),
                "q_total_power_error_W": sample.get("q_total_power_error_W"),
                "q_power_on_boundary_W": sample.get("q_power_on_boundary_W"),
                "reason_tags": reasons,
            }
        )
    return records


def _loader_contract_errors(
    bundle: dict[str, Any],
    meta: dict[str, Any],
    temperature: np.ndarray | None,
) -> list[str]:
    errors = []
    node_count = int(bundle["coords"].shape[0])
    for key, shape in (
        ("coords", (node_count, 3)),
        ("layer_id", (node_count,)),
        ("region_id", (node_count,)),
        ("material_id", (node_count,)),
        ("q_field", (node_count, 1)),
        ("bc_features", (node_count, 4)),
    ):
        array = bundle.get(key)
        if array is None:
            errors.append(f"missing_{key}")
        elif tuple(array.shape) != shape:
            errors.append(f"{key}_shape_{array.shape}_expected_{shape}")
        elif not np.all(np.isfinite(array)):
            errors.append(f"{key}_nan_inf")
    k_field = bundle.get("k_field")
    if k_field is None:
        errors.append("missing_k_field")
    elif k_field.ndim != 2 or k_field.shape[0] != node_count or k_field.shape[1] not in (1, 3):
        errors.append(f"k_field_shape_{k_field.shape}_expected_N_1_or_3")
    elif not np.all(np.isfinite(k_field)):
        errors.append("k_field_nan_inf")
    if temperature is not None:
        if tuple(temperature.shape) != (node_count, 1):
            errors.append(f"temperature_shape_{temperature.shape}_expected_{(node_count, 1)}")
        elif not np.all(np.isfinite(temperature)):
            errors.append("temperature_nan_inf")
    for key in ("stage", "subset_name", "boundary_params", "boundary_types", "interfaces"):
        if key not in meta:
            errors.append(f"sample_meta_missing_{key}")
    boundary = meta.get("boundary_params", {})
    if isinstance(boundary, dict):
        for path in (
            ("top", "h_W_m2K"),
            ("top", "T_inf_K"),
            ("bottom", "T_fixed_K"),
            ("side", "type"),
        ):
            cursor = boundary
            for part in path:
                cursor = cursor.get(part, None) if isinstance(cursor, dict) else None
            if cursor is None:
                errors.append("boundary_params_missing_" + "_".join(path))
    return errors


def classify_qc_sample(sample: dict[str, Any]) -> dict[str, Any]:
    reject_reasons = []
    physical_reasons = []
    review_reasons = []

    if sample.get("solver_status") != "solved":
        reject_reasons.append("solver_failure")
    if not bool(sample.get("nan_inf_ok", False)):
        reject_reasons.append("nan_inf")
    if not bool(sample.get("schema_loader_ok", False)):
        reject_reasons.append("schema_loader_failure")
        reject_reasons.extend(sample.get("schema_loader_errors", []))
    if abs(float(sample.get("q_total_power_error_W", 0.0))) > QC_POWER_ERROR_TOL_W:
        reject_reasons.append("q_power_error")
    if float(sample.get("q_power_on_boundary_W", 0.0)) != 0.0:
        reject_reasons.append("boundary_q_deposition")
    if int(sample.get("q_source_boundary_violation_count", 0)) != 0:
        reject_reasons.append("boundary_q_deposition")
    if int(sample.get("q_deposited_on_boundary_node_count", 0)) != 0:
        reject_reasons.append("boundary_q_deposition")
    if sample.get("DeltaT_bin") == "reject_low":
        reject_reasons.append("reject_low")

    is_high = float(sample.get("DeltaT_peak_K", 0.0)) > QC_HIGH_DELTAT_K
    if is_high:
        if float(sample.get("low_k_q_overlap_fraction", 0.0)) >= 0.5:
            physical_reasons.append("low_k_trapped_hotspot")
        if sample.get("cooling_regime") == "weak_effective_air":
            physical_reasons.append("weak_cooling")
        if (
            int(sample.get("q_block_count", 0)) > 1
            or float(sample.get("q_total_target_power_W", 0.0)) > QC_HIGH_POWER_W
            or str(sample.get("q_family", "")).startswith("multi_block")
        ):
            physical_reasons.append("multi_source_or_high_power_bottleneck")
        if not physical_reasons:
            reject_reasons.append("unclassified_high_deltaT")

    if reject_reasons:
        qc_class = "reject_resample"
        accepted = False
    elif physical_reasons:
        qc_class = "physical_hard_keep"
        accepted = True
    elif sample.get("DeltaT_bin") == "review_high":
        qc_class = "review_hold"
        accepted = True
        review_reasons.append("review_high_deltaT_bin")
    else:
        qc_class = "clean_keep"
        accepted = True

    return {
        "qc_class": qc_class,
        "qc_accept": accepted,
        "qc_reject_reasons": sorted(set(reject_reasons)),
        "qc_physical_keep_reasons": sorted(set(physical_reasons)),
        "qc_review_reasons": sorted(set(review_reasons)),
        "qc_policy": "p3c_qc_freeze_v0",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sha256_manifest(dataset_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.name == "sha256_manifest.json":
            continue
        rel = path.relative_to(dataset_dir).as_posix()
        files.append(
            {
                "path": rel,
                "size_bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    by_path = {entry["path"]: entry["sha256"] for entry in files}
    return {
        "schema_version": "heat3d_v4_p3c_sha256_manifest_v0",
        "dataset_id": dataset_dir.name,
        "file_count": len(files),
        "total_size_bytes": int(sum(entry["size_bytes"] for entry in files)),
        "manifest_json_sha256": by_path.get("manifest.json"),
        "audit_summary_json_sha256": by_path.get("audit_summary.json"),
        "files": files,
    }


def _ensure_clean_target(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    if force:
        shutil.rmtree(path)
        return
    raise FileExistsError(f"target already exists; pass --force to replace: {path}")


def _sample_id_for_index(index: int, sample_count: int) -> str:
    width = max(3, len(str(max(0, sample_count - 1))))
    return f"sample_{index:0{width}d}"


def _delta_t_bin(delta_t_peak: float) -> tuple[str, str | None]:
    for name, low, high in DELTA_T_BINS:
        if low is not None and delta_t_peak < low:
            continue
        if high is not None and delta_t_peak >= high:
            continue
        reason = None if name in {"low", "nominal", "hard"} else f"deltaT_peak_bin={name}"
        return name, reason
    return "reject_high", "deltaT_peak_bin=reject_high"


def _finite_ok(*arrays: np.ndarray) -> bool:
    return all(bool(np.all(np.isfinite(array))) for array in arrays)


def _low_k_overlap_fraction(bundle: dict[str, Any]) -> float:
    q_active = bundle["q_field"].reshape(-1) > 0.0
    active_count = int(np.count_nonzero(q_active))
    if active_count == 0:
        return 0.0
    block_family = {
        block["block_id"]: block["k_family"]
        for block in bundle["scene"]["k"]["blocks"]
    }
    winners = bundle["sample_meta"]["k_node_metadata"]["winning_block_id"]
    low_k_count = 0
    for index, active in enumerate(q_active):
        if active and block_family.get(winners[index]) == "low_k_dielectric_underfill":
            low_k_count += 1
    return low_k_count / float(active_count)


def _triage_delta_t(sample: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    high_delta_t = float(sample["DeltaT_peak_K"]) > 15.0
    audit_passed = (
        sample["solver_status"] == "solved"
        and sample["nan_inf_ok"]
        and abs(float(sample["q_total_power_error_W"])) <= 1.0e-10
        and float(sample["q_power_on_boundary_W"]) == 0.0
        and int(sample["q_source_boundary_violation_count"]) == 0
        and int(sample["q_source_side_boundary_violation_count"]) == 0
        and int(sample["q_deposited_on_boundary_node_count"]) == 0
    )
    low_k_fraction = _low_k_overlap_fraction(bundle)
    if not high_delta_t:
        return {
            "high_deltaT_triage": "not_high_deltaT",
            "low_k_q_overlap_fraction": low_k_fraction,
            "physical_keep_reason": None,
            "dataset_action": "keep_for_pilot",
        }
    if not audit_passed:
        return {
            "high_deltaT_triage": "reject_policy_or_solver_violation",
            "low_k_q_overlap_fraction": low_k_fraction,
            "physical_keep_reason": None,
            "dataset_action": "reject",
        }
    if low_k_fraction >= 0.5:
        return {
            "high_deltaT_triage": "physical_low_k_enclosed_compact_hotspot",
            "low_k_q_overlap_fraction": low_k_fraction,
            "physical_keep_reason": "low_k_enclosed_compact_hotspot",
            "dataset_action": "keep_for_pilot",
        }
    if sample["cooling_regime"] == "weak_effective_air":
        return {
            "high_deltaT_triage": "physical_weak_cooling_high_deltaT",
            "low_k_q_overlap_fraction": low_k_fraction,
            "physical_keep_reason": "weak_cooling",
            "dataset_action": "keep_for_pilot",
        }
    if (
        int(sample.get("q_block_count", 0)) > 1
        or float(sample.get("q_total_target_power_W", 0.0)) > QC_HIGH_POWER_W
        or str(sample.get("q_family", "")).startswith("multi_block")
    ):
        return {
            "high_deltaT_triage": "physical_multi_source_or_high_power_bottleneck",
            "low_k_q_overlap_fraction": low_k_fraction,
            "physical_keep_reason": "multi_source_or_high_power_bottleneck",
            "dataset_action": "keep_for_pilot",
        }
    return {
        "high_deltaT_triage": "audit_passed_high_deltaT_unclassified",
        "low_k_q_overlap_fraction": low_k_fraction,
        "physical_keep_reason": None,
        "dataset_action": "review_for_pilot",
    }


def _sample_audit(
    *,
    sample_id: str,
    bundle: dict[str, Any],
    temperature: np.ndarray,
    solve_meta: dict[str, Any],
) -> dict[str, Any]:
    scene = bundle["scene"]
    meta = bundle["sample_meta"]
    bottom_t = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
    delta_t = temperature.reshape(-1) - bottom_t
    delta_t_peak = float(np.max(delta_t))
    delta_t_p95 = float(np.percentile(delta_t, 95))
    delta_t_bin, reject_reason = _delta_t_bin(delta_t_peak)
    q_meta = meta["q_block_metadata"]
    q_total_realized = float(sum(block["realized_power_W"] for block in q_meta))
    control_volume_weights = control_volume_weights_for_domain(scene["domain"])
    q_integral_from_array = float(np.sum(bundle["q_field"].reshape(-1) * control_volume_weights))
    q_power_audit = meta["q_power_audit"]
    solution_audit = solve_meta["solution_audit"]
    nan_inf_ok = _finite_ok(
        bundle["coords"],
        bundle["k_field"],
        bundle["q_field"],
        bundle["bc_features"],
        temperature,
    )
    sample = {
        "sample_id": sample_id,
        "scene_id": scene["scene_id"],
        "solver_status": solution_audit["status"],
        "residual_norm": solution_audit["residual_norm"],
        "energy_balance_residual": solution_audit["energy_balance_residual"],
        "bottom_dirichlet_error": solution_audit["bottom_dirichlet_error"],
        "DeltaT_peak_K": delta_t_peak,
        "DeltaT_p95_K": delta_t_p95,
        "DeltaT_bin": delta_t_bin,
        "q_total_target_power_W": q_power_audit["q_total_target_power_W"],
        "q_total_realized_power_W": q_total_realized,
        "q_integral_from_array_W": q_integral_from_array,
        "q_total_power_error_W": q_power_audit["q_total_power_error_W"],
        "q_power_integration_policy": q_power_audit["q_power_integration_policy"],
        "q_power_on_bottom_W": q_power_audit["q_power_on_bottom_W"],
        "q_power_on_top_W": q_power_audit["q_power_on_top_W"],
        "q_power_on_xmin_W": q_power_audit["q_power_on_xmin_W"],
        "q_power_on_xmax_W": q_power_audit["q_power_on_xmax_W"],
        "q_power_on_ymin_W": q_power_audit["q_power_on_ymin_W"],
        "q_power_on_ymax_W": q_power_audit["q_power_on_ymax_W"],
        "q_power_on_side_W": q_power_audit["q_power_on_side_W"],
        "q_power_on_boundary_W": q_power_audit["q_power_on_boundary_W"],
        "q_power_on_bottom_fraction": q_power_audit["q_power_on_bottom_fraction"],
        "q_power_on_top_fraction": q_power_audit["q_power_on_top_fraction"],
        "q_power_on_side_fraction": q_power_audit["q_power_on_side_fraction"],
        "q_source_boundary_violation_count": q_power_audit["q_source_boundary_violation_count"],
        "q_source_side_boundary_violation_count": q_power_audit["q_source_side_boundary_violation_count"],
        "q_active_z_min": q_power_audit["q_active_z_min"],
        "q_active_z_max": q_power_audit["q_active_z_max"],
        "semantic_boundary_inset_fraction": q_power_audit["semantic_boundary_inset_fraction"],
        "semantic_inset_domain_xyz": q_power_audit["semantic_inset_domain_xyz"],
        "solver_safe_deposition_mask": q_power_audit["solver_safe_deposition_mask"],
        "q_deposited_on_boundary_node_count": q_power_audit["q_deposited_on_boundary_node_count"],
        "q_max_after_sum_W_m3": float(np.max(bundle["q_field"])),
        "background_k_family": meta["background_k"]["background_k_family"],
        "background_k_value": meta["background_k"]["background_k_value"],
        "material_block_count": len(scene["k"]["blocks"]),
        "k_mode": scene["k"]["mode"],
        "diag3_policy": scene["k"]["diag3_policy"],
        "q_family": scene["q"]["family"],
        "q_block_count": len(scene["q"]["blocks"]),
        "cooling_regime": scene["BC"]["cooling_regime"],
        "top_h_W_m2K": scene["BC"]["top_h_W_m2K"],
        "contact_model": meta["contact"]["contact_model"],
        "nan_inf_ok": nan_inf_ok,
        "reject_or_review_reason": reject_reason,
        "operator_checksum": solution_audit["operator_checksum"],
    }
    sample.update(_triage_delta_t(sample, bundle))
    return sample


def _summary(samples: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = len(samples)
    total = pass_count + len(failures)
    finite_energy = [
        abs(float(sample["energy_balance_residual"]))
        for sample in samples
        if np.isfinite(float(sample["energy_balance_residual"]))
    ]
    bottom_errors = [
        abs(float(sample["bottom_dirichlet_error"]))
        for sample in samples
        if np.isfinite(float(sample["bottom_dirichlet_error"]))
    ]
    q_boundary_power = [
        abs(float(sample["q_power_on_boundary_W"]))
        for sample in samples
        if np.isfinite(float(sample["q_power_on_boundary_W"]))
    ]
    q_side_power = [
        abs(float(sample["q_power_on_side_W"]))
        for sample in samples
        if np.isfinite(float(sample["q_power_on_side_W"]))
    ]
    q_power_errors = [
        abs(float(sample["q_total_power_error_W"]))
        for sample in samples
        if np.isfinite(float(sample["q_total_power_error_W"]))
    ]
    dataset_actions = sorted({sample["dataset_action"] for sample in samples})
    high_triage = sorted({sample["high_deltaT_triage"] for sample in samples})
    qc_classes = sorted({sample.get("qc_class", "missing") for sample in samples})
    return {
        "schema_version": "heat3d_v4_p3c_smoke16_audit_v3",
        "sample_count": total,
        "pass_count": pass_count,
        "failure_count": len(failures),
        "solver_pass_rate": pass_count / total if total else 0.0,
        "max_abs_energy_balance_residual": max(finite_energy) if finite_energy else None,
        "max_bottom_dirichlet_error": max(bottom_errors) if bottom_errors else None,
        "max_abs_q_total_power_error_W": max(q_power_errors) if q_power_errors else None,
        "max_q_power_on_boundary_W": max(q_boundary_power) if q_boundary_power else None,
        "max_q_power_on_side_W": max(q_side_power) if q_side_power else None,
        "q_source_boundary_violation_count": sum(
            int(sample["q_source_boundary_violation_count"]) for sample in samples
        ),
        "q_source_side_boundary_violation_count": sum(
            int(sample["q_source_side_boundary_violation_count"]) for sample in samples
        ),
        "q_deposited_on_boundary_node_count": sum(
            int(sample["q_deposited_on_boundary_node_count"]) for sample in samples
        ),
        "high_deltaT_count": sum(1 for sample in samples if float(sample["DeltaT_peak_K"]) > 15.0),
        "dataset_action_counts": {
            name: sum(1 for sample in samples if sample["dataset_action"] == name)
            for name in dataset_actions
        },
        "qc_class_counts": {
            name: sum(1 for sample in samples if sample.get("qc_class", "missing") == name)
            for name in qc_classes
        },
        "high_deltaT_triage_counts": {
            name: sum(1 for sample in samples if sample["high_deltaT_triage"] == name)
            for name in high_triage
        },
        "DeltaT_bin_counts": {
            name: sum(1 for sample in samples if sample["DeltaT_bin"] == name)
            for name, _, _ in DELTA_T_BINS
        },
        "k_mode_counts": {
            name: sum(1 for sample in samples if sample["k_mode"] == name)
            for name in sorted({sample["k_mode"] for sample in samples})
        },
        "diag3_policy_counts": {
            name: sum(1 for sample in samples if sample["diag3_policy"] == name)
            for name in sorted({sample["diag3_policy"] for sample in samples})
        },
        "q_family_counts": {
            name: sum(1 for sample in samples if sample["q_family"] == name)
            for name in sorted({sample["q_family"] for sample in samples})
        },
        "cooling_regime_counts": {
            name: sum(1 for sample in samples if sample["cooling_regime"] == name)
            for name in sorted({sample["cooling_regime"] for sample in samples})
        },
        "nan_inf_ok": all(sample["nan_inf_ok"] for sample in samples) and not failures,
        "samples": samples,
        "failures": failures,
    }


def generate_smoke16(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    dataset_dir: Path = REPO_ROOT / SMOKE16_DATASET_DIR,
    output_dir: Path = REPO_ROOT / SMOKE16_OUTPUT_DIR,
    sample_count: int = SMOKE16_SAMPLE_COUNT,
    seed: int = SMOKE16_SEED,
    force: bool = False,
    reject_resample: bool = False,
    max_candidates: int | None = None,
    accepted_qc_classes: set[str] | None = None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    qc_filter_enabled = accepted_qc_classes is not None
    selected_qc_classes = (
        set(ACCEPTED_QC_CLASSES)
        if accepted_qc_classes is None
        else set(accepted_qc_classes)
    )
    unknown_qc_classes = sorted(selected_qc_classes - ACCEPTED_QC_CLASSES)
    if unknown_qc_classes:
        raise ValueError(f"unknown accepted QC classes: {unknown_qc_classes}")
    if not selected_qc_classes:
        raise ValueError("accepted_qc_classes must not be empty")
    resample_enabled = bool(reject_resample or qc_filter_enabled)
    _ensure_clean_target(dataset_dir, force=force)
    _ensure_clean_target(output_dir, force=force)
    dataset_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    candidate_count = (
        int(max_candidates)
        if max_candidates is not None
        else (int(sample_count) * 4 if resample_enabled else int(sample_count))
    )
    if candidate_count < sample_count:
        raise ValueError("--max-candidates must be >= --samples")
    write_plan = build_smoke16_write_plan(registry, sample_count=sample_count, seed=seed)
    batch = generate_dryrun_batch(registry, sample_count=candidate_count, seed=seed)
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    manifest_samples = []
    sample_records: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    solver_options = SolverOptions(solver_mode="perfect_contact", matrix_backend="sparse_csr")

    for candidate_index, scene in enumerate(batch["scenes"]):
        if len(samples) >= sample_count:
            break
        candidate_id = f"candidate_{candidate_index:06d}"
        bundle = materialize_scene_arrays(scene, registry)
        meta = dict(bundle["sample_meta"])
        meta.update(
            {
                "array_preflight_only": False,
                "artifact_writes": True,
                "solver_called": False,
                "dataset_id": dataset_dir.name,
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "sample_id": candidate_id,
                "split": "unassigned",
            }
        )

        try:
            problem = extract_problem_from_arrays(
                coords=bundle["coords"],
                k_field=bundle["k_field"],
                q_field=bundle["q_field"],
                sample_meta=meta,
                sample_dir=None,
            )
            temperature, solve_meta = solve_temperature_from_problem(problem, solver_options)
            meta["solver_called"] = True
            meta["validation"]["solver_label_pending"] = False
            meta["validation"]["solver_status"] = "solved"
            meta["label_solver"] = {
                "solver_family": solve_meta["solver_family"],
                "solver_mode": solve_meta["solver_mode"],
                "matrix_backend": solve_meta["matrix_backend"],
                "operator_checksum": solve_meta["solution_audit"]["operator_checksum"],
            }
            schema_errors = _loader_contract_errors(bundle, meta, temperature)
            sample_audit = _sample_audit(
                sample_id=candidate_id,
                bundle={**bundle, "sample_meta": meta},
                temperature=temperature,
                solve_meta=solve_meta,
            )
            sample_audit["candidate_id"] = candidate_id
            sample_audit["candidate_index"] = candidate_index
            sample_audit["schema_loader_ok"] = not schema_errors
            sample_audit["schema_loader_errors"] = schema_errors
            sample_audit.update(classify_qc_sample(sample_audit))
            sample_audit["dataset_action"] = sample_audit["qc_class"]
            qc_class_filtered = (
                qc_filter_enabled and sample_audit["qc_class"] not in selected_qc_classes
            )
            reject_resampled = (
                reject_resample and sample_audit["qc_class"] == "reject_resample"
            )
            if qc_class_filtered or reject_resampled:
                sample_audit["acceptance_filter"] = {
                    "accepted_qc_classes": sorted(selected_qc_classes),
                    "reason": "qc_class_not_selected",
                }
                rejected_candidates.append(sample_audit)
                continue

            sample_id = _sample_id_for_index(len(samples), sample_count)
            sample_dir = dataset_dir / sample_id
            sample_dir.mkdir()
            meta["sample_id"] = sample_id
            meta["accepted_index"] = len(samples)
            meta["qc_policy"] = {
                "policy": "p3c_qc_freeze_v0",
                "qc_class": sample_audit["qc_class"],
                "qc_accept": bool(sample_audit["qc_accept"]),
                "qc_reject_reasons": sample_audit["qc_reject_reasons"],
                "qc_physical_keep_reasons": sample_audit["qc_physical_keep_reasons"],
                "qc_review_reasons": sample_audit["qc_review_reasons"],
            }
            sample_audit["sample_id"] = sample_id
            sample_audit["accepted_index"] = len(samples)
            np.save(sample_dir / "coords.npy", bundle["coords"])
            np.save(sample_dir / "layer_id.npy", bundle["layer_id"])
            np.save(sample_dir / "region_id.npy", bundle["region_id"])
            np.save(sample_dir / "material_id.npy", bundle["material_id"])
            np.save(sample_dir / "k_field.npy", bundle["k_field"])
            np.save(sample_dir / "q_field.npy", bundle["q_field"])
            np.save(sample_dir / "bc_features.npy", bundle["bc_features"])
            np.save(sample_dir / "temperature.npy", temperature)
            _write_json(sample_dir / "sample_meta.json", meta)
            samples.append(sample_audit)
            manifest_samples.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "sample_dir": sample_id,
                    "files": [*PLANNED_SAMPLE_FILES, "temperature.npy"],
                    "DeltaT_bin": sample_audit["DeltaT_bin"],
                    "q_family": sample_audit["q_family"],
                    "cooling_regime": sample_audit["cooling_regime"],
                    "k_mode": sample_audit["k_mode"],
                    "diag3_policy": sample_audit["diag3_policy"],
                    "high_deltaT_triage": sample_audit["high_deltaT_triage"],
                    "physical_keep_reason": sample_audit["physical_keep_reason"],
                    "dataset_action": sample_audit["dataset_action"],
                    "qc_class": sample_audit["qc_class"],
                    "split": "unassigned",
                }
            )
            sample_records.append(
                {
                    "sample_id": sample_id,
                    "sample_dir": sample_dir,
                    "meta": meta,
                    "sample_audit": sample_audit,
                    "manifest_sample": manifest_samples[-1],
                }
            )
        except Exception as exc:  # noqa: BLE001
            failure = {
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "scene_id": scene["scene_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "qc_class": "reject_resample",
                "qc_reject_reasons": ["solver_failure"],
                "input_summary": {
                    "k_mode": scene["k"]["mode"],
                    "diag3_policy": scene["k"]["diag3_policy"],
                    "q_family": scene["q"]["family"],
                    "cooling_regime": scene["BC"]["cooling_regime"],
                    "top_h_W_m2K": scene["BC"]["top_h_W_m2K"],
                    "q_block_count": len(scene["q"]["blocks"]),
                },
            }
            if resample_enabled:
                rejected_candidates.append(failure)
                continue
            failures.append(failure)
            break

    if len(samples) < sample_count:
        failures.append(
            {
                "error_type": "InsufficientAcceptedSamples",
                "error": (
                    f"accepted_count={len(samples)} below requested sample_count={sample_count}; "
                    f"candidate_count={candidate_count}"
                ),
                "accepted_count": len(samples),
                "requested_count": sample_count,
                "candidate_count": candidate_count,
                "rejected_candidate_count": len(rejected_candidates),
            }
        )

    split_map: dict[str, str] = {}
    split_audit: dict[str, Any] = {}
    p3b_lite_subset: dict[str, Any] = {}
    if samples:
        split_map = assign_stratified_splits(samples, seed=seed)
        split_audit = build_split_audit(samples, split_map=split_map, seed=seed)
        for record in sample_records:
            split = split_map[record["sample_id"]]
            record["meta"]["split"] = split
            record["meta"]["split_policy"] = {
                "policy": SPLIT_POLICY_NAME,
                "seed": int(seed),
                "assigned_after_solver_audit": True,
                "audit_fields": list(SPLIT_AUDIT_FIELDS),
            }
            record["sample_audit"]["split"] = split
            record["manifest_sample"]["split"] = split
            _write_json(record["sample_dir"] / "sample_meta.json", record["meta"])
        p3b_lite_subset = select_p3b_lite_validation_subset(samples, seed=seed)

    manifest = {
        "schema_version": "heat3d_v4_p3c_dataset_manifest_v4",
        "dataset_id": dataset_dir.name,
        "accepted_count_requested": sample_count,
        "sample_count_written": len(manifest_samples),
        "candidate_count_generated": candidate_count,
        "candidate_count_consumed": len(samples) + len(rejected_candidates),
        "rejected_candidate_count": len(rejected_candidates),
        "reject_resample_enabled": resample_enabled,
        "accepted_qc_classes": sorted(selected_qc_classes),
        "seed": seed,
        "split_map": split_map,
        "split_audit": split_audit,
        "p3b_lite_validation_subset": p3b_lite_subset,
        "registry": str(registry_path.relative_to(REPO_ROOT)),
        "write_plan": write_plan,
        "sample_schema": {
            "required_files": [*PLANNED_SAMPLE_FILES, "temperature.npy"],
        },
        "samples": manifest_samples,
    }
    audit = _summary(samples, failures)
    audit["schema_version"] = "heat3d_v4_p3c_dataset_audit_v4"
    audit["split_map"] = split_map
    audit["split_audit"] = split_audit
    audit["p3b_lite_validation_subset"] = p3b_lite_subset
    audit["review_sample_closeout"] = build_review_closeout(samples)
    audit["qc_policy"] = {
        "policy": "p3c_qc_freeze_v0",
        "accepted_classes": sorted(selected_qc_classes),
        "reject_resample_reasons": [
            "solver_failure",
            "nan_inf",
            "q_power_error",
            "boundary_q_deposition",
            "schema_loader_failure",
            "reject_low",
            "unclassified_high_deltaT",
        ],
        "physical_hard_keep_reasons": [
            "low_k_trapped_hotspot",
            "weak_cooling",
            "multi_source_or_high_power_bottleneck",
        ],
    }
    audit["candidate_generation"] = {
        "accepted_count": len(samples),
        "accepted_count_requested": sample_count,
        "candidate_count_generated": candidate_count,
        "candidate_count_consumed": len(samples) + len(rejected_candidates),
        "rejected_candidate_count": len(rejected_candidates),
        "reject_resample_enabled": resample_enabled,
        "accepted_qc_classes": sorted(selected_qc_classes),
    }
    audit["rejected_candidates"] = rejected_candidates
    _write_json(dataset_dir / "manifest.json", manifest)
    _write_json(dataset_dir / "audit_summary.json", audit)
    sha_manifest = build_sha256_manifest(dataset_dir)
    _write_json(dataset_dir / "sha256_manifest.json", sha_manifest)
    _write_json(output_dir / "audit_summary.json", audit)
    _write_json(output_dir / "sha256_manifest.json", sha_manifest)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / SMOKE16_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / SMOKE16_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=SMOKE16_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=SMOKE16_SEED)
    parser.add_argument("--reject-resample", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--accept-qc-class",
        action="append",
        choices=sorted(ACCEPTED_QC_CLASSES),
        default=None,
        help="QC class to write; repeat for multiple classes. Others are resampled.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    audit = generate_smoke16(
        registry_path=args.registry,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        sample_count=args.samples,
        seed=args.seed,
        force=args.force,
        reject_resample=args.reject_resample,
        max_candidates=args.max_candidates,
        accepted_qc_classes=(
            set(args.accept_qc_class) if args.accept_qc_class is not None else None
        ),
    )
    print(
        json.dumps(
            {
                "dataset": str(args.dataset_dir),
                "output": str(args.output_dir),
                "sample_count": audit["sample_count"],
                "solver_pass_rate": audit["solver_pass_rate"],
                "failure_count": audit["failure_count"],
                "DeltaT_bin_counts": audit["DeltaT_bin_counts"],
                "max_abs_energy_balance_residual": audit["max_abs_energy_balance_residual"],
                "max_bottom_dirichlet_error": audit["max_bottom_dirichlet_error"],
                "max_q_power_on_boundary_W": audit["max_q_power_on_boundary_W"],
                "max_q_power_on_side_W": audit["max_q_power_on_side_W"],
                "q_source_boundary_violation_count": audit["q_source_boundary_violation_count"],
                "q_source_side_boundary_violation_count": audit["q_source_side_boundary_violation_count"],
                "q_deposited_on_boundary_node_count": audit["q_deposited_on_boundary_node_count"],
                "high_deltaT_count": audit["high_deltaT_count"],
                "dataset_action_counts": audit["dataset_action_counts"],
                "qc_class_counts": audit.get("qc_class_counts", {}),
                "high_deltaT_triage_counts": audit["high_deltaT_triage_counts"],
                "split_counts": audit.get("split_audit", {}).get("split_counts", {}),
                "p3b_lite_selected_count": audit.get("p3b_lite_validation_subset", {}).get("selected_count"),
                "review_sample_count": len(audit.get("review_sample_closeout", [])),
                "accepted_count": audit.get("candidate_generation", {}).get("accepted_count"),
                "rejected_candidate_count": audit.get("candidate_generation", {}).get("rejected_candidate_count"),
                "candidate_count_consumed": audit.get("candidate_generation", {}).get("candidate_count_consumed"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
