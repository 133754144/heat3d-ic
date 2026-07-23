#!/usr/bin/env python3
"""Validate the V6_03 -> V6_04 shape-attention-only ablation."""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping
from unittest.mock import patch

import jax
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v6_dataset import (  # noqa: E402
    CANONICAL_V6_DATASET_ID,
    EXPECTED_SPLIT_COUNTS,
    SHARED_SUPPORT_V6_DATASET_ID,
    Heat3DV6DualRobinDataset,
)
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


BASE_PATH = ROOT / "configs/heat3d_v6/V6_03_V5best_P1h.yaml"
CANDIDATE_PATH = ROOT / "configs/heat3d_v6/V6_04_V5best_P1h_DualAttention.yaml"
MANIFEST_PATH = ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json"
DATASET_PATH = ROOT / "data" / SHARED_SUPPORT_V6_DATASET_ID
RESOLVED_PATH = (
    ROOT / "configs/heat3d_v6/resolved/V6_04_V5best_P1h_DualAttention.resolved.yaml"
)
DIFF_JSON = ROOT / "configs/heat3d_v6/v6_04_dual_attention_resolved_diff.json"
DIFF_MD = ROOT / "docs/v6_04_dual_attention_resolved_diff.md"

EXPECTED_MANIFEST_SHA256 = "324ca50a85698223d36c12a05d3e26b5cbc9aa00b559d067619baeb37f11e9d5"
EXPECTED_ARCHIVE_SHA256 = "f58141b3f365c5c90a57ec3802ae57c7e7afbf83ba0ab988060a617164b14c00"

SCIENTIFIC_DIFF_PATH = "model.shape_attention_mode"
IDENTITY_DIFF_PATHS = {
    "config_id",
    "description",
    "export.output_dir",
    "export.run_name",
    "metadata.ablation_parent_config_id",
    "metadata.ablation_scientific_difference",
    "metadata.experiment_role",
    "metadata.log_path",
}
EXPECTED_DIFF_PATHS = IDENTITY_DIFF_PATHS | {SCIENTIFIC_DIFF_PATH}


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
    return [{"path": prefix, "v6_03": left, "v6_04": right}]


def _scientific_payload_without_shape_attention(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(config))
    payload.pop("config_id", None)
    payload.pop("description", None)
    payload.pop("metadata", None)
    payload["export"].pop("output_dir", None)
    payload["export"].pop("run_name", None)
    payload["model"].pop("shape_attention_mode", None)
    return payload


def _tree_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(name: str, item: Any) -> None:
        digest.update(name.encode("utf-8"))
        if is_dataclass(item):
            for field in fields(item):
                visit(f"{name}.{field.name}", getattr(item, field.name))
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(f"{name}.{key}", item[key])
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(f"{name}[{index}]", child)
        else:
            array = np.asarray(item)
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(tuple(array.shape)).encode("utf-8"))
            digest.update(np.ascontiguousarray(array).tobytes())

    visit("graph", value)
    return digest.hexdigest()


def _runner_args(config: Mapping[str, Any]) -> argparse.Namespace:
    command = build_training_command(config, python_executable="python")
    values = list(command[2:])
    wrapper_flags = {
        "--normalization-profile",
        "--condition-feature-transform",
        "--input-feature-schema",
        "--coord-policy",
        "--extent-feature-policy",
    }
    cleaned: list[str] = []
    index = 0
    while index < len(values):
        if values[index] in wrapper_flags:
            index += 2
        else:
            cleaned.append(values[index])
            index += 1
    with patch.object(sys, "argv", [CANDIDATE_PATH.name, *cleaned]):
        return runner.parse_args()


def _runtime_graph_sha(config: Mapping[str, Any], coords: np.ndarray) -> str:
    args = _runner_args(config)
    graph_config = runner._graph_config_from_args(args)
    builder = Heat3DGraphBuilder(**graph_config)
    metadata = builder.build_metadata(coords, key=runner._metadata_key(args.graph_seed))
    return _tree_hash(metadata)


def _dry_run_command(config: Mapping[str, Any]) -> list[str]:
    command = build_training_command(config, python_executable="python")

    def value(flag: str) -> str:
        assert flag in command, f"dry-run command missing {flag}"
        return command[command.index(flag) + 1]

    assert command[:2] == ["python", "scripts/run_heat3d_v4_controlled_training.py"]
    assert value("--shape-attention-mode") == "physics_gate"
    assert value("--scale-attention-mode") == "physics_gate"
    assert value("--subset") == "data/heat3d_v6_p1h_shared_support1024_v0"
    assert value("--dataset-manifest") == (
        "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json"
    )
    assert value("--epochs") == "600"
    assert value("--batch-size") == value("--micro-batch-size") == "24"
    assert value("--seed") == value("--model-seed") == "0"
    assert "--epoch-wise-batch-regrouping" not in command
    return command


def _report(dataset_root: Path) -> dict[str, Any]:
    base = _resolved(BASE_PATH)
    candidate = _resolved(CANDIDATE_PATH)
    validate_v2_config(base, config_path=BASE_PATH)
    validate_v2_config(candidate, config_path=CANDIDATE_PATH)

    diffs = _leaf_diffs(base, candidate)
    paths = {row["path"] for row in diffs}
    unexpected = sorted(paths - EXPECTED_DIFF_PATHS)
    missing = sorted(EXPECTED_DIFF_PATHS - paths)
    assert not unexpected, f"unexpected V6_03/V6_04 resolved diffs: {unexpected}"
    assert not missing, f"expected V6_03/V6_04 resolved diffs missing: {missing}"
    scientific_paths = sorted(paths - IDENTITY_DIFF_PATHS)
    assert scientific_paths == [SCIENTIFIC_DIFF_PATH]
    assert _scientific_payload_without_shape_attention(base) == (
        _scientific_payload_without_shape_attention(candidate)
    )

    assert base["model"]["shape_attention_mode"] == "none"
    assert candidate["model"]["shape_attention_mode"] == "physics_gate"
    assert base["model"]["scale_attention_mode"] == "physics_gate"
    assert candidate["model"]["scale_attention_mode"] == "physics_gate"
    base_model = copy.deepcopy(base["model"])
    candidate_model = copy.deepcopy(candidate["model"])
    base_model.pop("shape_attention_mode")
    candidate_model.pop("shape_attention_mode")
    assert base_model == candidate_model
    for section in ("dataset", "graph", "loss", "optimizer", "run", "diagnostics"):
        assert candidate[section] == base[section], f"{section} drift"
    assert candidate.get("baseline_reference") == base.get("baseline_reference")
    assert candidate["export"]["selection_metric"] == base["export"]["selection_metric"]
    assert candidate["run"]["epochs"] == 600
    assert candidate["run"]["batch_size"] == candidate["run"]["micro_batch_size"] == 24
    assert candidate["run"]["epoch_wise_batch_regrouping"] is False
    assert candidate["metadata"]["training_started"] is False
    assert candidate["metadata"]["canonical_dataset_id"] == CANONICAL_V6_DATASET_ID
    assert candidate["metadata"]["candidate_dataset_id"] == SHARED_SUPPORT_V6_DATASET_ID
    assert candidate["metadata"]["dataset_lifecycle_status"] == "canonical_candidate"
    for key in ("execution_host", "training_commit", "runner_pid", "launch_timestamp_utc"):
        assert candidate["metadata"][key] is None, f"historical metadata retained: {key}"

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert _sha256(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256
    assert manifest["full_field_archive"]["sha256"] == EXPECTED_ARCHIVE_SHA256
    assert len({row["point_coordinates_sha256"] for row in manifest["samples"]}) == 1
    assert len({row["graph_sha256"] for row in manifest["samples"]}) == 1

    dataset = Heat3DV6DualRobinDataset(
        dataset_root,
        MANIFEST_PATH,
        include_roles={"train", "valid"},
    )
    assert dataset.materialized_roles == {"train", "valid"}
    assert len(dataset) == 896
    assert {key: len(value) for key, value in dataset.split_ids.items()} == EXPECTED_SPLIT_COUNTS
    assert not any(sample.meta["split_role"] == "test" for sample in dataset.samples)
    coords = np.asarray(dataset.samples[0].condition.coords)
    assert coords.shape == (1024, 3)
    assert all(np.array_equal(coords, sample.condition.coords) for sample in dataset.samples)

    base_graph_sha = _runtime_graph_sha(base, coords)
    candidate_graph_sha = _runtime_graph_sha(candidate, coords)
    assert base_graph_sha == candidate_graph_sha, "V6_03/V6_04 runtime graph drift"

    command = _dry_run_command(candidate)
    return {
        "schema_version": "heat3d_v6_04_dual_attention_resolved_diff_v1",
        "status": "passed",
        "baseline_config_id": base["config_id"],
        "candidate_config_id": candidate["config_id"],
        "resolved_diff_paths": [row["path"] for row in diffs],
        "resolved_diffs": diffs,
        "identity_diff_paths": sorted(IDENTITY_DIFF_PATHS),
        "scientific_diff_paths": scientific_paths,
        "unexpected_diff_paths": unexpected,
        "scientific_payload_equal_after_registered_ablation": True,
        "invariants": {
            "dataset_equal": True,
            "model_equal_except_shape_attention_mode": True,
            "graph_equal": True,
            "loss_equal": True,
            "optimizer_equal": True,
            "lr_schedule_equal": True,
            "run_equal": True,
            "epochs": 600,
            "batch_size": 24,
            "micro_batch_size": 24,
            "scale_attention_mode": "physics_gate",
            "epoch_wise_batch_regrouping": False,
            "training_started": False,
        },
        "dataset": {
            "dataset_id": SHARED_SUPPORT_V6_DATASET_ID,
            "dataset_root": str(dataset_root),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "materialized_roles": sorted(dataset.materialized_roles),
            "test_target_materialized": False,
            "sample_count": 1024,
            "train_count": 768,
            "valid_count": 128,
            "test_count": 128,
            "node_count": 1024,
            "unique_coordinate_hash_count": 1,
            "unique_graph_hash_count": 1,
        },
        "runtime_graph": {
            "jax_backend": jax.default_backend(),
            "v6_03_sha256": base_graph_sha,
            "v6_04_sha256": candidate_graph_sha,
            "equal": True,
            "frozen_manifest_sha256": manifest["shared_graph_sha256"],
            "matches_frozen_manifest": candidate_graph_sha == manifest["shared_graph_sha256"],
        },
        "dry_run": {
            "training_executed": False,
            "optimizer_update_executed": False,
            "command": command,
            "manual_command": (
                "python scripts/run_heat3d_v4_config.py --config "
                "configs/heat3d_v6/V6_04_V5best_P1h_DualAttention.yaml"
            ),
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    rows = "\n".join(
        f"| `{row['path']}` | `{row['v6_03']}` | `{row['v6_04']}` |"
        for row in report["resolved_diffs"]
    )
    graph = report["runtime_graph"]
    return f"""# V6_04 P1h DualAttention resolved-config diff

Status: **{report['status']}**. The sole scientific difference from
`V6_03_V5best_P1h` is `model.shape_attention_mode: none -> physics_gate`.
Scale attention remains `physics_gate`.

## Resolved leaf differences

| path | V6_03 | V6_04 |
|---|---|---|
{rows}

## Frozen invariants

- scientific diff paths: `{report['scientific_diff_paths']}`
- dataset / graph / loss / optimizer / LR / seed / B24 / e600: unchanged
- runtime graph backend/hash: `{graph['jax_backend']}` / `{graph['v6_04_sha256']}`
- V6_03/V6_04 runtime graph equal: `{graph['equal']}`
- train+valid materialized; test target materialized: false
- training or optimizer update executed: false / false

## Manual command

```bash
{report['dry_run']['manual_command']}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--write-artifacts", action="store_true")
    args = parser.parse_args()
    report = _report(args.dataset.resolve())
    if args.write_artifacts:
        RESOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESOLVED_PATH.write_text(
            yaml.safe_dump(_resolved(CANDIDATE_PATH), sort_keys=False),
            encoding="utf-8",
        )
        DIFF_JSON.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        DIFF_MD.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
