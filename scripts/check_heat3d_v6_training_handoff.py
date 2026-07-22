#!/usr/bin/env python3
"""Check the canonical V6 P1g-v0 training handoff without training."""

from __future__ import annotations

import argparse
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
from rigno.heat3d_v6_dataset import (  # noqa: E402
    CANONICAL_V6_DATASET_ID,
    EXPECTED_SPLIT_COUNTS,
    Heat3DV6DualRobinDataset,
    V6_DUAL_ROBIN_CONDITION_FEATURES,
)
from rigno.heat3d_v6_global_context import (  # noqa: E402
    GLOBAL_CONTEXT_FEATURES_V6,
    global_context_from_v6_inputs,
)


CONFIGS = {
    "V6_01_V4best": (
        ROOT / "configs/heat3d_v6/V6_01_V4best.yaml",
        ROOT / "configs/heat3d_v4/generated/V4P5_02_clean_baseline_raw_B28_e600.yaml",
    ),
    "V6_02_V5best": (
        ROOT / "configs/heat3d_v6/V6_02_V5best.yaml",
        ROOT / "configs/heat3d_v5/V4P5_42_canonical.yaml",
    ),
}
DATA_ROOT = ROOT / "data" / CANONICAL_V6_DATASET_ID
MANIFEST = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json"
AMENDMENT = ROOT / "configs/heat3d_v6/v6_p1g_v0_acceptance_amendment.json"
LIFECYCLE = ROOT / "configs/heat3d_v6/v6_training_dataset_lifecycle.csv"


def resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = (
        resolve_inherited_yaml(payload, path)
        if payload.get("schema_version") == "heat3d_v4_inherited_config_v0"
        else dict(payload)
    )
    value["config_id"] = payload.get("config_id", path.stem)
    return value


def _leaf_diffs(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_leaf_diffs(left.get(key), right.get(key), path))
        return result
    return [] if left == right else [prefix]


def _check_baseline_diff(config_id: str, candidate: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    diff_paths = _leaf_diffs(candidate, base)
    allowed_prefixes = (
        "config_id",
        "description",
        "dataset.",
        "metadata.",
        "model.global_context_feature_names",
        "run.micro_batch_size",
        "run.validation_batch_size",
        "run.prediction_batch_size",
        "run.init_checkpoint",
        "run.final_probe_eval_after_training",
        "run.final_probe_output_dir",
        "run.post_training_diagnostics",
        "run.post_training_diagnostics_output_dir",
        "export.output_dir",
        "export.run_name",
        "diagnostics.run_baseline_comparison",
        "baseline_reference",
    )
    unexpected = [
        path for path in diff_paths
        if not any(path == prefix or path.startswith(prefix) for prefix in allowed_prefixes)
    ]
    assert not unexpected, f"{config_id}: unexpected resolved scientific diffs {unexpected}"
    for section in ("loss", "optimizer", "graph"):
        assert candidate.get(section) == base.get(section), f"{config_id}: {section} drift"
    assert candidate["run"]["epochs"] == base["run"]["epochs"] == 600
    assert candidate["export"]["selection_metric"] == base["export"]["selection_metric"]
    for key in ("architecture", "node_latent_size", "edge_latent_size", "processor_steps", "mlp_hidden_layers"):
        assert candidate["model"].get(key) == base["model"].get(key), f"{config_id}: model.{key} drift"
    model_diffs = {
        key for key in set(candidate["model"]) | set(base["model"])
        if candidate["model"].get(key) != base["model"].get(key)
    }
    allowed_model = {"global_context_feature_names"} if config_id == "V6_02_V5best" else set()
    assert model_diffs == allowed_model, f"{config_id}: unexpected model diff {sorted(model_diffs)}"
    assert candidate["dataset"]["name"] == CANONICAL_V6_DATASET_ID
    assert candidate["dataset"]["loader"] == "v6_dual_robin_manifest_v1"
    assert candidate["dataset"]["split_map_path"] is None
    assert candidate["run"]["batch_size"] == 28
    assert candidate["run"]["micro_batch_size"] == 8
    assert candidate["run"]["validation_batch_size"] == 32
    assert candidate["run"]["prediction_batch_size"] == 32
    assert candidate["run"]["drop_last"] is False
    assert candidate["run"]["init_checkpoint"] is None
    assert candidate["run"]["final_probe_eval_after_training"] is False
    assert candidate["diagnostics"]["run_baseline_comparison"] is False
    assert candidate["metadata"]["training_started"] is False
    assert candidate["metadata"]["effective_batch_size"] == 28
    assert candidate["metadata"]["optimizer_updates_per_epoch"] == 28
    assert candidate["metadata"]["final_partial_effective_batch_size"] == 12
    if config_id == "V6_02_V5best":
        assert candidate["export"]["selection_metric"] == "valid_rel_rmse_v4_pct"
    validate_v2_config(candidate, config_path=CONFIGS[config_id][0])
    return {
        "config_id": config_id,
        "baseline": CONFIGS[config_id][1].name,
        "model_diff_keys": sorted(model_diffs),
        "loss_equal": True,
        "optimizer_equal": True,
        "graph_equal": True,
        "epochs_equal": True,
        "selection_metric_equal": True,
        "effective_batch_size": 28,
        "micro_batch_size": 8,
        "optimizer_updates_per_epoch": 28,
        "tail_effective_batch_size": 12,
        "resolved_diff_paths": diff_paths,
        "unexpected_scientific_diff_paths": unexpected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-resolved", action="store_true")
    args = parser.parse_args()
    reports = []
    resolved_dir = ROOT / "configs/heat3d_v6/resolved"
    if args.write_resolved:
        resolved_dir.mkdir(parents=True, exist_ok=True)
    for config_id, (path, base_path) in CONFIGS.items():
        candidate = resolved(path)
        base = resolved(base_path)
        reports.append(_check_baseline_diff(config_id, candidate, base))
        resolved_path = resolved_dir / f"{config_id}.resolved.yaml"
        if args.write_resolved:
            resolved_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        assert yaml.safe_load(resolved_path.read_text(encoding="utf-8")) == candidate

    dataset = Heat3DV6DualRobinDataset(DATA_ROOT, MANIFEST)
    assert {key: len(value) for key, value in dataset.split_ids.items()} == EXPECTED_SPLIT_COUNTS
    assert not any("bottom_T_fixed" in name for name in V6_DUAL_ROBIN_CONDITION_FEATURES)
    assert tuple(resolved(CONFIGS["V6_02_V5best"][0])["model"]["global_context_feature_names"]) == GLOBAL_CONTEXT_FEATURES_V6
    example = dataset[0]
    bridge = example.build_temperature_rise_legacy_inputs_from_relative_features()
    assert bridge.legacy_inputs.c.shape == (1, 1, 1024, 11)
    assert bridge.target_delta_u.shape == (1, 1, 1024, 1)
    assert example.meta["v6_adapter"]["bottom_boundary_semantics"] == "robin_not_dirichlet"
    context = global_context_from_v6_inputs(**example.v6_global_context_inputs())
    assert tuple(context) == GLOBAL_CONTEXT_FEATURES_V6
    assert np.all(np.isfinite(np.asarray(list(context.values()), dtype=np.float64)))

    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert amendment["original_qualification_unchanged"] is True
    qualification = ROOT / amendment["original_qualification_path"]
    assert hashlib.sha256(qualification.read_bytes()).hexdigest() == amendment["original_qualification_sha256"]
    assert amendment["passed"] is True
    observed = amendment["observed"]
    assert observed["peak_deltaT_below_29K_count"] == 0
    assert observed["peak_deltaT_below_30K_fraction"] <= 0.02
    assert observed["peak_deltaT_30_to_80K_fraction"] >= 0.95
    assert observed["peak_deltaT_above_80K_fraction"] <= 0.05
    assert observed["peak_deltaT_above_100K_count"] == 0

    with LIFECYCLE.open(newline="", encoding="utf-8") as handle:
        lifecycle = list(csv.DictReader(handle))
    assert all(row["lifecycle_status"] == "archived" for row in lifecycle if row["phase"] in {f"P1{letter}" for letter in "abcdef"})
    canonical = [row for row in lifecycle if row["lifecycle_status"] == "canonical"]
    assert len(canonical) == 1 and canonical[0]["dataset_id"] == CANONICAL_V6_DATASET_ID
    smoke = json.loads((ROOT / "configs/heat3d_v6/v6_training_handoff_smoke.json").read_text(encoding="utf-8"))
    assert smoke["training_started"] is False and smoke["checkpoint_saved"] is False
    assert all(row["finite_forward_backward"] and row["adamw_update_finite"] for row in smoke["configs"])

    report = {
        "status": "passed",
        "canonical_dataset_id": CANONICAL_V6_DATASET_ID,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "group_locked": True,
        "loader_smoke": {
            "sample_count": len(dataset),
            "node_count": int(example.condition.coords.shape[0]),
            "input_width": int(example.condition.condition_features.shape[1]),
            "bottom_robin_not_dirichlet": True,
            "global_context_dim": len(context),
            "target_or_label_derived_global_inputs": False,
        },
        "baseline_diffs": reports,
        "training_started": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
