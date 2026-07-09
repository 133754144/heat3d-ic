#!/usr/bin/env python3
"""Build the V4 P5 clean-IID dataset while retaining hard challenge samples."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_heat3d_v4_p3c_smoke16 import (  # noqa: E402
    build_sha256_manifest,
    generate_smoke16,
)


DEFAULT_SOURCE_DATASET = REPO_ROOT / "data/heat3d_v4_p3c_candidate1024_v0"
DEFAULT_SOURCE_SPLIT = (
    REPO_ROOT
    / "configs/heat3d_v4/"
    "candidate1024_v0_train768_valid128_test128_stratified_seed0.json"
)
DEFAULT_TARGET_DATASET = REPO_ROOT / "data/heat3d_v4_p5_clean_nohard_v0"
DEFAULT_TARGET_SPLIT = (
    REPO_ROOT
    / "configs/heat3d_v4/"
    "candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json"
)
DATASET_ID = "heat3d_v4_p5_clean_nohard_v0"
GENERATION_SEED = 5301
SPLIT_SEED = 0
NEW_COUNTS = {"train": 25, "valid_iid": 12, "test_iid": 12}
EXPECTED_COUNTS = {
    "train": 672,
    "valid_iid": 128,
    "test_iid": 128,
    "hard_train_holdout": 121,
    "hard_challenge_valid": 12,
    "hard_challenge_test": 12,
}
HARD_SPLIT_MAP = {
    "train": "hard_train_holdout",
    "valid_iid": "hard_challenge_valid",
    "test_iid": "hard_challenge_test",
}
FINGERPRINT_FILES = (
    "coords.npy",
    "layer_id.npy",
    "region_id.npy",
    "material_id.npy",
    "k_field.npy",
    "q_field.npy",
    "bc_features.npy",
)


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected JSON object")
    return loaded


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_fingerprint(sample_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        path = sample_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{sample_dir}: missing fingerprint file {name}")
        digest.update(name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _qc_class(meta: dict[str, Any]) -> str:
    policy = meta.get("qc_policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{meta.get('sample_id')}: missing qc_policy")
    return str(policy.get("qc_class") or "")


def _stable_order_key(sample_id: str, candidate_id: str) -> str:
    return hashlib.sha256(
        f"{SPLIT_SEED}:p5_clean_replacement:{sample_id}:{candidate_id}".encode("utf-8")
    ).hexdigest()


def _copy_sample_files(source: Path, target: Path, required_files: list[str]) -> None:
    target.mkdir(parents=True)
    for name in required_files:
        if name == "sample_meta.json":
            continue
        source_path = source / name
        target_path = target / name
        if not source_path.is_file():
            raise FileNotFoundError(f"{source}: missing required file {name}")
        try:
            os.link(source_path, target_path)
        except OSError:
            shutil.copy2(source_path, target_path)


def _geometry_bc_signature(sample_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
    bc_features = np.asarray(np.load(sample_dir / "bc_features.npy"))
    return {
        "coords_shape": list(coords.shape),
        "coords_min": np.min(coords, axis=0).tolist(),
        "coords_max": np.max(coords, axis=0).tolist(),
        "bc_features_shape": list(bc_features.shape),
        "bc_feature_names": list(meta.get("bc_feature_names") or []),
        "bc_counts": dict(meta.get("bc_counts") or {}),
        "boundary_types": dict(meta.get("boundary_types") or {}),
    }


def _validate_integrity(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    validation = meta.get("validation") or {}
    q_audit = meta.get("q_power_audit") or {}
    if not bool(meta.get("solver_called")):
        errors.append("solver_not_called")
    if validation.get("solver_status") != "solved":
        errors.append("solver_status_not_solved")
    if not bool(validation.get("array_preflight_passed")):
        errors.append("array_preflight_failed")
    if int(q_audit.get("q_source_boundary_violation_count") or 0) != 0:
        errors.append("source_boundary_violation")
    if int(q_audit.get("q_source_side_boundary_violation_count") or 0) != 0:
        errors.append("side_boundary_violation")
    if float(q_audit.get("q_power_on_boundary_W") or 0.0) != 0.0:
        errors.append("boundary_power_violation")
    return errors


def _assign_new_samples(manifest_rows: list[dict[str, Any]]) -> dict[str, str]:
    ordered = sorted(
        manifest_rows,
        key=lambda row: _stable_order_key(
            str(row["sample_id"]),
            str(row.get("candidate_id") or ""),
        ),
    )
    assignments: dict[str, str] = {}
    offset = 0
    for split in ("train", "valid_iid", "test_iid"):
        count = NEW_COUNTS[split]
        for row in ordered[offset : offset + count]:
            assignments[str(row["sample_id"])] = split
        offset += count
    if offset != len(ordered):
        raise RuntimeError(f"new sample assignment mismatch: {offset}!={len(ordered)}")
    return assignments


def _manifest_row(
    *,
    source_row: dict[str, Any],
    sample_id: str,
    split: str,
    origin: str,
    source_sample_id: str,
) -> dict[str, Any]:
    row = dict(source_row)
    row.update(
        {
            "sample_id": sample_id,
            "sample_dir": sample_id,
            "split": split,
            "p5_origin": origin,
            "p5_source_sample_id": source_sample_id,
        }
    )
    return row


def build_dataset(
    *,
    source_dataset: Path,
    source_split_path: Path,
    target_dataset: Path,
    target_split_path: Path,
    force: bool,
    keep_staging: bool,
) -> dict[str, Any]:
    if target_dataset.exists():
        if not force:
            raise FileExistsError(f"target dataset exists: {target_dataset}")
        shutil.rmtree(target_dataset)
    if target_split_path.exists() and not force:
        raise FileExistsError(f"target split exists: {target_split_path}")
    if not source_dataset.is_dir():
        raise FileNotFoundError(f"source dataset missing: {source_dataset}")

    source_manifest = _read_json(source_dataset / "manifest.json")
    source_split_payload = _read_json(source_split_path)
    source_splits = {
        str(sample_id): str(split)
        for sample_id, split in source_split_payload["sample_splits"].items()
    }
    source_rows = {
        str(row["sample_id"]): dict(row) for row in source_manifest["samples"]
    }
    if set(source_splits) != set(source_rows):
        raise ValueError("source manifest and formal split map sample IDs differ")

    required_files = list(source_manifest["sample_schema"]["required_files"])
    source_counts = Counter()
    source_meta: dict[str, dict[str, Any]] = {}
    source_fingerprints: dict[str, str] = {}
    integrity_failures: list[dict[str, Any]] = []
    for sample_id in sorted(source_splits):
        sample_dir = source_dataset / sample_id
        meta = _read_json(sample_dir / "sample_meta.json")
        qc_class = _qc_class(meta)
        source_counts[(source_splits[sample_id], qc_class)] += 1
        errors = _validate_integrity(meta)
        if errors:
            integrity_failures.append({"sample_id": sample_id, "errors": errors})
        source_meta[sample_id] = meta
        source_fingerprints[_sample_fingerprint(sample_dir)] = sample_id

    expected_source_counts = {
        ("train", "physical_hard_keep"): 121,
        ("valid_iid", "physical_hard_keep"): 12,
        ("test_iid", "physical_hard_keep"): 12,
    }
    for split, expected_total in (("train", 768), ("valid_iid", 128), ("test_iid", 128)):
        actual_total = sum(
            count for (actual_split, _qc), count in source_counts.items() if actual_split == split
        )
        if actual_total != expected_total:
            raise ValueError(f"source {split} count mismatch: {actual_total}!={expected_total}")
    for key, expected in expected_source_counts.items():
        if source_counts[key] != expected:
            raise ValueError(f"source hard count mismatch for {key}: {source_counts[key]}!={expected}")
    if integrity_failures:
        raise ValueError(f"source integrity failures: {integrity_failures[:3]}")

    temp_root = Path(tempfile.gettempdir())
    replacement_dataset = temp_root / f"{DATASET_ID}_replacement_seed{GENERATION_SEED}"
    replacement_output = temp_root / f"{DATASET_ID}_replacement_audit_seed{GENERATION_SEED}"
    replacement_audit = generate_smoke16(
        dataset_dir=replacement_dataset,
        output_dir=replacement_output,
        sample_count=sum(NEW_COUNTS.values()),
        seed=GENERATION_SEED,
        force=True,
        reject_resample=True,
        max_candidates=256,
        accepted_qc_classes={"clean_keep"},
    )
    if replacement_audit.get("failure_count") != 0:
        raise RuntimeError(f"replacement generation failed: {replacement_audit.get('failures')}")
    replacement_manifest = _read_json(replacement_dataset / "manifest.json")
    replacement_rows = list(replacement_manifest["samples"])
    if len(replacement_rows) != sum(NEW_COUNTS.values()):
        raise RuntimeError("replacement sample count mismatch")
    replacement_assignments = _assign_new_samples(replacement_rows)

    reference_sample_id = min(source_splits)
    reference_signature = _geometry_bc_signature(
        source_dataset / reference_sample_id,
        source_meta[reference_sample_id],
    )
    duplicate_failures = []
    replacement_fingerprints: set[str] = set()
    replacement_meta: dict[str, dict[str, Any]] = {}
    for row in replacement_rows:
        sample_id = str(row["sample_id"])
        sample_dir = replacement_dataset / sample_id
        meta = _read_json(sample_dir / "sample_meta.json")
        if _qc_class(meta) != "clean_keep":
            raise RuntimeError(f"{sample_id}: replacement is not clean_keep")
        errors = _validate_integrity(meta)
        if errors:
            raise RuntimeError(f"{sample_id}: replacement integrity errors: {errors}")
        if _geometry_bc_signature(sample_dir, meta) != reference_signature:
            raise RuntimeError(f"{sample_id}: geometry/BC signature differs from source")
        fingerprint = _sample_fingerprint(sample_dir)
        if fingerprint in source_fingerprints:
            duplicate_failures.append(
                {
                    "replacement_sample_id": sample_id,
                    "source_sample_id": source_fingerprints[fingerprint],
                }
            )
        if fingerprint in replacement_fingerprints:
            duplicate_failures.append(
                {"replacement_sample_id": sample_id, "source_sample_id": "replacement_duplicate"}
            )
        replacement_fingerprints.add(fingerprint)
        replacement_meta[sample_id] = meta
    if duplicate_failures:
        raise RuntimeError(f"duplicate replacement samples: {duplicate_failures}")

    building_dir = target_dataset.with_name(target_dataset.name + ".building")
    if building_dir.exists():
        shutil.rmtree(building_dir)
    building_dir.mkdir(parents=True)
    final_splits: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []

    for sample_id in sorted(source_splits):
        source_formal_split = source_splits[sample_id]
        meta = dict(source_meta[sample_id])
        hard = _qc_class(meta) == "physical_hard_keep"
        split = HARD_SPLIT_MAP[source_formal_split] if hard else source_formal_split
        target_sample_dir = building_dir / sample_id
        _copy_sample_files(source_dataset / sample_id, target_sample_dir, required_files)
        meta.update(
            {
                "dataset_id": DATASET_ID,
                "sample_id": sample_id,
                "split": split,
                "p5_provenance": {
                    "origin": "candidate1024_v0",
                    "source_dataset_id": source_manifest["dataset_id"],
                    "source_sample_id": sample_id,
                    "source_formal_split": source_formal_split,
                    "qc_class_preserved": True,
                },
                "split_policy": {
                    "policy": "p5_clean_nohard_fixed_roles_seed0",
                    "seed": SPLIT_SEED,
                    "source_split_map": str(source_split_path.relative_to(REPO_ROOT)),
                },
            }
        )
        _write_json(target_sample_dir / "sample_meta.json", meta)
        final_splits[sample_id] = split
        manifest_rows.append(
            _manifest_row(
                source_row=source_rows[sample_id],
                sample_id=sample_id,
                split=split,
                origin="candidate1024_v0",
                source_sample_id=sample_id,
            )
        )

    replacement_rows_by_id = {
        str(row["sample_id"]): dict(row) for row in replacement_rows
    }
    next_index = len(source_splits)
    for replacement_id in sorted(
        replacement_rows_by_id,
        key=lambda sample_id: (
            ("train", "valid_iid", "test_iid").index(replacement_assignments[sample_id]),
            _stable_order_key(
                sample_id,
                str(replacement_rows_by_id[sample_id].get("candidate_id") or ""),
            ),
        ),
    ):
        sample_id = f"sample_{next_index:04d}"
        next_index += 1
        split = replacement_assignments[replacement_id]
        meta = dict(replacement_meta[replacement_id])
        target_sample_dir = building_dir / sample_id
        _copy_sample_files(replacement_dataset / replacement_id, target_sample_dir, required_files)
        meta.update(
            {
                "dataset_id": DATASET_ID,
                "sample_id": sample_id,
                "accepted_index": next_index - 1,
                "split": split,
                "p5_provenance": {
                    "origin": "generated_clean_replacement",
                    "generation_seed": GENERATION_SEED,
                    "source_sample_id": replacement_id,
                    "candidate_id": meta.get("candidate_id"),
                    "accepted_qc_classes": ["clean_keep"],
                },
                "split_policy": {
                    "policy": "p5_clean_replacement_fixed_roles_seed0",
                    "seed": SPLIT_SEED,
                    "role": split,
                },
            }
        )
        _write_json(target_sample_dir / "sample_meta.json", meta)
        final_splits[sample_id] = split
        manifest_rows.append(
            _manifest_row(
                source_row=replacement_rows_by_id[replacement_id],
                sample_id=sample_id,
                split=split,
                origin="generated_clean_replacement",
                source_sample_id=replacement_id,
            )
        )

    final_counts = Counter(final_splits.values())
    if dict(sorted(final_counts.items())) != dict(sorted(EXPECTED_COUNTS.items())):
        raise RuntimeError(f"final split counts mismatch: {dict(final_counts)}")
    for sample_id, split in final_splits.items():
        meta = _read_json(building_dir / sample_id / "sample_meta.json")
        if split in {"train", "valid_iid", "test_iid"} and _qc_class(meta) == "physical_hard_keep":
            raise RuntimeError(f"{sample_id}: physical hard sample leaked into clean split {split}")

    split_rel = target_split_path.relative_to(REPO_ROOT).as_posix()
    manifest = {
        "schema_version": "heat3d_v4_p5_clean_nohard_dataset_manifest_v0",
        "dataset_id": DATASET_ID,
        "dataset_path": target_dataset.relative_to(REPO_ROOT).as_posix(),
        "source_dataset_id": source_manifest["dataset_id"],
        "source_dataset_path": source_dataset.relative_to(REPO_ROOT).as_posix(),
        "source_split_map": source_split_path.relative_to(REPO_ROOT).as_posix(),
        "formal_split_map": split_rel,
        "sample_count_written": len(manifest_rows),
        "split_counts": dict(sorted(final_counts.items())),
        "replacement_generation": {
            "generation_seed": GENERATION_SEED,
            "split_seed": SPLIT_SEED,
            "accepted_qc_classes": ["clean_keep"],
            "new_counts": NEW_COUNTS,
            "candidate_count_generated": replacement_manifest["candidate_count_generated"],
            "candidate_count_consumed": replacement_manifest["candidate_count_consumed"],
            "filtered_candidate_count": replacement_manifest["rejected_candidate_count"],
            "duplicate_count": 0,
        },
        "sample_schema": {"required_files": required_files},
        "samples": sorted(manifest_rows, key=lambda row: str(row["sample_id"])),
    }
    _write_json(building_dir / "manifest.json", manifest)
    audit = {
        "schema_version": "heat3d_v4_p5_clean_nohard_dataset_audit_v0",
        "dataset_id": DATASET_ID,
        "status_ok": True,
        "sample_count": len(final_splits),
        "split_counts": dict(sorted(final_counts.items())),
        "source_counts": {
            f"{split}:{qc_class}": count
            for (split, qc_class), count in sorted(source_counts.items())
        },
        "replacement_qc_class_counts": replacement_audit["qc_class_counts"],
        "replacement_integrity": {
            "solver_pass_rate": replacement_audit["solver_pass_rate"],
            "failure_count": replacement_audit["failure_count"],
            "q_source_boundary_violation_count": replacement_audit[
                "q_source_boundary_violation_count"
            ],
            "q_source_side_boundary_violation_count": replacement_audit[
                "q_source_side_boundary_violation_count"
            ],
            "max_q_power_on_boundary_W": replacement_audit["max_q_power_on_boundary_W"],
            "max_q_power_on_side_W": replacement_audit["max_q_power_on_side_W"],
            "nan_inf_ok": replacement_audit["nan_inf_ok"],
        },
        "geometry_bc_signature": reference_signature,
        "duplicate_count": 0,
        "original_samples_modified": False,
        "original_hard_labels_preserved": True,
    }
    _write_json(building_dir / "audit_summary.json", audit)
    sha_manifest = build_sha256_manifest(building_dir)
    sha_manifest["dataset_id"] = DATASET_ID
    _write_json(building_dir / "sha256_manifest.json", sha_manifest)

    if target_dataset.exists():
        shutil.rmtree(target_dataset)
    building_dir.replace(target_dataset)

    split_payload = {
        "schema_version": "heat3d_v4_p5_clean_nohard_split_map_v0",
        "dataset_id": DATASET_ID,
        "dataset_path": target_dataset.relative_to(REPO_ROOT).as_posix(),
        "seed": SPLIT_SEED,
        "source_dataset_id": source_manifest["dataset_id"],
        "source_split_map": source_split_path.relative_to(REPO_ROOT).as_posix(),
        "assignment_method": "preserve_source_nonhard_roles_plus_fixed_clean_replacements_v0",
        "target_counts": EXPECTED_COUNTS,
        "actual_counts": dict(sorted(final_counts.items())),
        "new_clean_sample_counts": NEW_COUNTS,
        "clean_split_keys": ["train", "valid_iid", "test_iid"],
        "hard_split_keys": [
            "hard_train_holdout",
            "hard_challenge_valid",
            "hard_challenge_test",
        ],
        "notes": [
            "Original non-hard train/valid/test roles are preserved.",
            "Original physical_hard_keep samples retain their QC labels and move only to hard role keys.",
            "Generated replacements are clean_keep, unique against all source samples, and assigned 25/12/12.",
            "clean_iid uses train/valid_iid/test_iid; hard_challenge uses the three hard role keys.",
            "all_iid is the union of the corresponding clean and hard role keys.",
        ],
        "sample_splits": dict(sorted(final_splits.items())),
    }
    _write_json(target_split_path, split_payload)

    if not keep_staging:
        shutil.rmtree(replacement_dataset, ignore_errors=True)
        shutil.rmtree(replacement_output, ignore_errors=True)
    return {
        "dataset": str(target_dataset),
        "split_map": str(target_split_path),
        "sample_count": len(final_splits),
        "split_counts": dict(sorted(final_counts.items())),
        "replacement_qc_class_counts": replacement_audit["qc_class_counts"],
        "filtered_candidate_count": replacement_manifest["rejected_candidate_count"],
        "duplicate_count": 0,
        "sha256_file_count": sha_manifest["file_count"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--source-split", type=Path, default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--target-dataset", type=Path, default=DEFAULT_TARGET_DATASET)
    parser.add_argument("--target-split", type=Path, default=DEFAULT_TARGET_SPLIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_dataset(
        source_dataset=args.source_dataset.resolve(),
        source_split_path=args.source_split.resolve(),
        target_dataset=args.target_dataset.resolve(),
        target_split_path=args.target_split.resolve(),
        force=args.force,
        keep_staging=args.keep_staging,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
