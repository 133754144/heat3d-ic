#!/usr/bin/env python3
"""Deterministic launch preflight for frozen P1i and V6_05 transfer."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from rigno.heat3d_v6_dataset import Heat3DV6DualRobinDataset  # noqa: E402


BASELINE = ROOT / "configs/heat3d_v6/V6_03_V5best_P1h.yaml"
CANDIDATE = ROOT / "configs/heat3d_v6_p1i/V6_05_V5best_P1i_seed0.yaml"
MANIFEST = ROOT / "configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_manifest.json"
CONFIG = ROOT / "configs/heat3d_v6_p1i/v6_p1i_formal1024_v1.yaml"
SPLIT = ROOT / "configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_split_manifest.json"
SEALED = ROOT / "configs/heat3d_v6_p1i/v6_p1i_sealed_iid_confirmatory_preregistration.json"
EXPECTED = {
    MANIFEST: "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514",
    CONFIG: "1e15a77fe51eea7ec64614566bb6bb12bfcf05948f3b7c8c6f3c85ec759a58f8",
    SPLIT: "87aaa84af4b203d0a8ba93ed33b3757d46adc3143dad970371daa0f893ea491f",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = resolve_inherited_yaml(payload, path)
    value["config_id"] = payload["config_id"]
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], name))
        return result
    return {prefix: value}


def _allowed_diff(path: str) -> bool:
    exact = {
        "config_id", "description", "dataset.name", "dataset.subset_path",
        "dataset.manifest_path", "dataset.split_map_path", "dataset.split_source",
        "dataset.loader", "dataset.bc_adapter", "dataset.feature_view",
        "run.micro_batch_size", "run.validation_batch_size", "run.prediction_batch_size",
    }
    return path in exact or path.startswith("export.") or path.startswith("metadata.")


def check(dataset_root: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["frozen_hashes"] = all(_sha(path) == digest for path, digest in EXPECTED.items())
    baseline, candidate = _resolved(BASELINE), _resolved(CANDIDATE)
    left, right = _flatten(baseline), _flatten(candidate)
    differences = [
        {"path": key, "baseline": left.get(key), "candidate": right.get(key)}
        for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)
    ]
    forbidden = [row for row in differences if not _allowed_diff(row["path"])]
    checks["resolved_diff_only_dataset_runtime_identity"] = not forbidden
    for section in ("model", "graph", "loss", "optimizer"):
        checks[f"frozen_{section}"] = baseline[section] == candidate[section]
    checks["epochs_600"] = candidate["run"]["epochs"] == baseline["run"]["epochs"] == 600
    checks["effective_batch_24"] = candidate["run"]["batch_size"] == baseline["run"]["batch_size"] == 24
    checks["varying_support_micro_b8"] = candidate["run"]["micro_batch_size"] == 8
    checks["random_initialization"] = candidate["run"].get("init_checkpoint") is None
    checks["valid_only_prediction"] = candidate["export"]["prediction_split"] == "valid_iid"
    checks["output_not_started"] = not (ROOT / candidate["export"]["output_dir"]).exists()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))["assignment"]
    manifest_roles = {row["sample_id"]: row["split_role"] for row in manifest["samples"]}
    counts = Counter(split.values())
    checks["sample_count_1024"] = len(manifest_roles) == len(split) == 1024
    checks["split_counts_768_128_128"] = counts == Counter({"train": 768, "valid_iid": 128, "test_iid": 128})
    checks["split_exact_match"] = manifest_roles == split
    file_failures: list[str] = []
    checked_files = 0
    for row in manifest["samples"]:
        sample_dir = dataset_root / str(row["relative_path"])
        for name, expected in row["file_sha256"].items():
            checked_files += 1
            path = sample_dir / name
            if not path.is_file() or _sha(path) != expected:
                file_failures.append(str(Path(row["relative_path"]) / name))
    checks["dataset_9216_file_sha256"] = checked_files == 9216 and not file_failures
    dataset = Heat3DV6DualRobinDataset(
        dataset_root / "samples", MANIFEST, include_roles={"train", "valid_iid"}
    )
    checks["loader_train_valid_896"] = len(dataset) == 896
    checks["test_not_materialized"] = all(
        row.meta["v6_adapter"]["manifest_split_role"] != "test_iid" for row in dataset.samples
    )
    checks["varying_support"] = len({row.meta["coordinate_sha256"] for row in dataset.samples[:32]}) > 1
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    checks["sealed_definition_only"] = (
        sealed["status"] == "definition_frozen_labels_not_generated_or_opened"
        and sealed["labels_generated"] is False
        and sealed["labels_opened"] is False
        and sealed["target_statistics_computed"] is False
    )
    checks["sealed_id_no_overlap"] = not (
        {f"v6p1isealed1_{index:04d}" for index in range(128)} & set(manifest_roles)
    )
    passed = all(checks.values())
    return {
        "schema_version": "heat3d_v6_p1i_training_launch_check_v1",
        "status": "passed" if passed else "failed",
        "checks": checks,
        "baseline_config": str(BASELINE.relative_to(ROOT)),
        "candidate_config": str(CANDIDATE.relative_to(ROOT)),
        "resolved_differences": differences,
        "forbidden_resolved_differences": forbidden,
        "split_counts": dict(sorted(counts.items())),
        "materialized_sample_count": len(dataset),
        "materialized_roles": sorted(dataset.materialized_roles),
        "dataset_files_checked": checked_files,
        "dataset_file_hash_failures": file_failures,
        "holdout_file_integrity_bytes_hashed": True,
        "holdout_target_semantics_opened": False,
        "training_started": False,
        "test_or_sealed_target_accessed_by_checker": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/heat3d_v6_p1i_continuous_physics1024_v1")
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args()
    result = check(args.dataset_root.resolve())
    if args.write_json:
        output = args.write_json if args.write_json.is_absolute() else ROOT / args.write_json
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
