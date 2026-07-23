#!/usr/bin/env python3
"""Validate the V6_02 -> V6_03 P1h single-variable configuration."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v6_dataset import (  # noqa: E402
    CANONICAL_V6_DATASET_ID,
    EXPECTED_SPLIT_COUNTS,
    SHARED_SUPPORT_V6_DATASET_ID,
    Heat3DV6DualRobinDataset,
)


BASE_PATH = ROOT / "configs/heat3d_v6/V6_02_V5best.yaml"
CANDIDATE_PATH = ROOT / "configs/heat3d_v6/V6_03_V5best_P1h.yaml"
MANIFEST_PATH = ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json"
DATASET_PATH = ROOT / "data" / SHARED_SUPPORT_V6_DATASET_ID
ACCEPTANCE_PATH = ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_acceptance.json"
LIFECYCLE_PATH = ROOT / "configs/heat3d_v6/v6_training_dataset_lifecycle.csv"
RESOLVED_DIR = ROOT / "configs/heat3d_v6/resolved"
DIFF_JSON = ROOT / "configs/heat3d_v6/v6_03_p1h_resolved_diff.json"
DIFF_MD = ROOT / "docs/v6_03_p1h_resolved_diff.md"

EXPECTED_MANIFEST_SHA256 = "324ca50a85698223d36c12a05d3e26b5cbc9aa00b559d067619baeb37f11e9d5"
EXPECTED_ARCHIVE_SHA256 = "f58141b3f365c5c90a57ec3802ae57c7e7afbf83ba0ab988060a617164b14c00"

ALLOWED_DIFF_PATHS = {
    "config_id",
    "description",
    "dataset.manifest_path",
    "dataset.name",
    "dataset.subset_path",
    "export.output_dir",
    "export.run_name",
    "metadata.candidate_dataset_id",
    "metadata.dataset_lifecycle_status",
    "metadata.execution_host",
    "metadata.launch_timestamp_utc",
    "metadata.log_path",
    "metadata.runner_pid",
    "metadata.training_commit",
    "metadata.training_started",
}


def _sha256(path: Path) -> str:
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


def _leaf_diffs(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_leaf_diffs(left.get(key), right.get(key), path))
        return result
    if left == right:
        return []
    return [{"path": prefix, "v6_02": left, "v6_03": right}]


def _scientific_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(config))
    payload.pop("config_id", None)
    payload.pop("description", None)
    payload.pop("metadata", None)
    for key in ("name", "subset_path", "manifest_path"):
        payload["dataset"].pop(key, None)
    for key in ("output_dir", "run_name"):
        payload["export"].pop(key, None)
    return payload


def _dry_run_command(config: Mapping[str, Any]) -> list[str]:
    command = build_training_command(config, python_executable="python")
    assert command[0:2] == ["python", "scripts/run_heat3d_v4_controlled_training.py"]
    assert "--subset" in command
    assert command[command.index("--subset") + 1] == "data/heat3d_v6_p1h_shared_support1024_v0"
    assert "--dataset-manifest" in command
    assert (
        command[command.index("--dataset-manifest") + 1]
        == "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json"
    )
    assert "--batch-size" in command and command[command.index("--batch-size") + 1] == "24"
    assert "--micro-batch-size" in command and command[command.index("--micro-batch-size") + 1] == "24"
    assert "--epoch-wise-batch-regrouping" not in command
    return command


def _report(dataset_root: Path) -> dict[str, Any]:
    base = _resolved(BASE_PATH)
    candidate = _resolved(CANDIDATE_PATH)
    validate_v2_config(base, config_path=BASE_PATH)
    validate_v2_config(candidate, config_path=CANDIDATE_PATH)

    diffs = _leaf_diffs(base, candidate)
    paths = {row["path"] for row in diffs}
    unexpected = sorted(paths - ALLOWED_DIFF_PATHS)
    missing = sorted(ALLOWED_DIFF_PATHS - paths)
    assert not unexpected, f"unexpected V6_02/V6_03 resolved diffs: {unexpected}"
    assert not missing, f"expected identity/dataset diffs missing: {missing}"
    assert _scientific_payload(base) == _scientific_payload(candidate)

    for section in ("model", "graph", "loss", "optimizer"):
        assert candidate[section] == base[section], f"{section} drift"
    assert candidate["run"] == base["run"], "run/scientific batching drift"
    assert candidate["diagnostics"] == base["diagnostics"], "diagnostics drift"
    assert candidate.get("baseline_reference") == base.get("baseline_reference")
    assert candidate["run"]["epochs"] == 600
    assert candidate["run"]["batch_size"] == candidate["run"]["micro_batch_size"] == 24
    assert candidate["run"]["validation_batch_size"] == candidate["run"]["prediction_batch_size"] == 32
    assert candidate["run"]["drop_last"] is False
    assert candidate["run"]["init_checkpoint"] is None
    assert candidate["run"]["epoch_wise_batch_regrouping"] is False
    assert candidate["export"]["selection_metric"] == base["export"]["selection_metric"]
    assert candidate["metadata"]["training_started"] is False
    assert candidate["metadata"]["canonical_dataset_id"] == CANONICAL_V6_DATASET_ID
    assert candidate["metadata"]["candidate_dataset_id"] == SHARED_SUPPORT_V6_DATASET_ID
    assert candidate["metadata"]["dataset_lifecycle_status"] == "canonical_candidate"
    for key in ("execution_host", "training_commit", "runner_pid", "launch_timestamp_utc"):
        assert candidate["metadata"][key] is None, f"historical metadata retained: {key}"
    assert candidate["metadata"]["micro_batches_per_epoch"] == 32
    assert candidate["metadata"]["optimizer_updates_per_epoch"] == 32
    assert candidate["metadata"]["b24_execution_mode"] == "one_real_B24_forward_backward_per_update"

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == SHARED_SUPPORT_V6_DATASET_ID
    assert manifest["sample_count"] == 1024 and manifest["group_count"] == 128
    assert _sha256(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256
    assert len({row["point_coordinates_sha256"] for row in manifest["samples"]}) == 1
    assert len({row["graph_sha256"] for row in manifest["samples"]}) == 1
    assert manifest["shared_coordinate_sha256"] == next(
        iter({row["point_coordinates_sha256"] for row in manifest["samples"]})
    )
    assert manifest["shared_graph_sha256"] == next(
        iter({row["graph_sha256"] for row in manifest["samples"]})
    )
    assert manifest["full_field_archive"]["sha256"] == EXPECTED_ARCHIVE_SHA256

    acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))
    assert acceptance["status"] == "passed"
    assert acceptance["guardrails"]["canonical_dataset_changed"] is False
    assert acceptance["guardrails"]["formal_training_started"] is False
    assert acceptance["full_field_archive_sha256"] == EXPECTED_ARCHIVE_SHA256
    assert acceptance["split_counts"] == {"train": 768, "valid": 128, "test": 128}

    dataset = Heat3DV6DualRobinDataset(
        dataset_root,
        MANIFEST_PATH,
        include_roles={"train", "valid"},
    )
    assert dataset.materialized_roles == {"train", "valid"}
    assert len(dataset) == 896
    assert {key: len(value) for key, value in dataset.split_ids.items()} == EXPECTED_SPLIT_COUNTS
    assert not any(sample.meta["split_role"] == "test" for sample in dataset.samples)
    assert all(sample.condition.coords.shape == (1024, 3) for sample in dataset.samples)
    coordinate_hashes = {
        hashlib.sha256(np.ascontiguousarray(sample.condition.coords).tobytes()).hexdigest()
        for sample in dataset.samples
    }
    assert len(coordinate_hashes) == 1

    with LIFECYCLE_PATH.open(newline="", encoding="utf-8") as handle:
        lifecycle = list(csv.DictReader(handle))
    canonical = [row for row in lifecycle if row["lifecycle_status"] == "canonical"]
    candidates = [row for row in lifecycle if row["lifecycle_status"] == "canonical_candidate"]
    assert len(canonical) == 1 and canonical[0]["dataset_id"] == CANONICAL_V6_DATASET_ID
    assert len(candidates) == 1 and candidates[0]["dataset_id"] == SHARED_SUPPORT_V6_DATASET_ID

    command = _dry_run_command(candidate)
    return {
        "schema_version": "heat3d_v6_03_p1h_resolved_diff_v1",
        "status": "passed",
        "baseline_config_id": base["config_id"],
        "candidate_config_id": candidate["config_id"],
        "baseline_config": str(BASE_PATH.relative_to(ROOT)),
        "candidate_config": str(CANDIDATE_PATH.relative_to(ROOT)),
        "resolved_diff_paths": [row["path"] for row in diffs],
        "resolved_diffs": diffs,
        "allowed_diff_paths": sorted(ALLOWED_DIFF_PATHS),
        "unexpected_scientific_diff_paths": unexpected,
        "scientific_payload_equal": True,
        "invariants": {
            "model_equal": True,
            "graph_equal": True,
            "loss_equal": True,
            "optimizer_equal": True,
            "lr_schedule_equal": True,
            "epochs": 600,
            "batch_size": 24,
            "micro_batch_size": 24,
            "forward_backward_per_epoch": 32,
            "optimizer_updates_per_epoch": 32,
            "epoch_wise_batch_regrouping": False,
            "selection_metric": candidate["export"]["selection_metric"],
            "training_started": False,
        },
        "dataset": {
            "dataset_id": SHARED_SUPPORT_V6_DATASET_ID,
            "dataset_root": str(dataset_root),
            "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "sample_count": 1024,
            "train_count": 768,
            "valid_count": 128,
            "test_count": 128,
            "materialized_roles": sorted(dataset.materialized_roles),
            "node_count": 1024,
            "unique_coordinate_hash_count": 1,
            "unique_graph_hash_count": 1,
            "shared_coordinate_sha256": manifest["shared_coordinate_sha256"],
            "shared_graph_sha256": manifest["shared_graph_sha256"],
            "test_target_materialized": False,
        },
        "dry_run": {
            "training_executed": False,
            "optimizer_update_executed": False,
            "command": command,
            "manual_command": (
                "python scripts/run_heat3d_v4_config.py "
                "--config configs/heat3d_v6/V6_03_V5best_P1h.yaml"
            ),
        },
        "canonical": {
            "global_dataset_id": CANONICAL_V6_DATASET_ID,
            "candidate_dataset_id": SHARED_SUPPORT_V6_DATASET_ID,
            "global_canonical_changed": False,
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    rows = "\n".join(
        f"| `{row['path']}` | `{row['v6_02']}` | `{row['v6_03']}` |"
        for row in report["resolved_diffs"]
    )
    return f"""# V6_03 P1h resolved-config diff

Status: **{report['status']}**. `V6_03_V5best_P1h` resolves from
`V6_02_V5best`; the dataset binding is the only scientific variable.
P1g-v0 remains the sole global canonical dataset and P1h-v0 is a
`canonical_candidate`.

## Resolved leaf differences

| path | V6_02 | V6_03 |
|---|---|---|
{rows}

## Frozen scientific invariants

- model / graph / loss / optimizer / LR schedule: exactly equal
- epochs / effective batch / micro batch: 600 / 24 / 24
- forward-backward / optimizer updates per epoch: 32 / 32
- epoch-wise batch regrouping: false
- checkpoint selection: `{report['invariants']['selection_metric']}`
- train / valid: 768 / 128; test target materialized: false
- shared coordinates / graph: one / one
- formal training or optimizer update executed by this checker: false / false

## Manual command

```bash
{report['dry_run']['manual_command']}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--write-artifacts", action="store_true")
    args = parser.parse_args()
    report = _report(args.dataset.resolve())
    if args.write_artifacts:
        RESOLVED_DIR.mkdir(parents=True, exist_ok=True)
        base = _resolved(BASE_PATH)
        candidate = _resolved(CANDIDATE_PATH)
        (RESOLVED_DIR / "V6_02_V5best.resolved.yaml").write_text(
            yaml.safe_dump(base, sort_keys=False), encoding="utf-8"
        )
        (RESOLVED_DIR / "V6_03_V5best_P1h.resolved.yaml").write_text(
            yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
        )
        DIFF_JSON.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        DIFF_MD.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
