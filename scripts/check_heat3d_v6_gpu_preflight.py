#!/usr/bin/env python3
"""Validate the frozen two-host V6 e1 GPU preflight closeout."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


REPORT = ROOT / "configs/heat3d_v6/v6_training_handoff_gpu_preflight.json"
MANIFEST = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json"
CONFIG_IDS = {"V6_01_V4best", "V6_02_V5best"}
EXPECTED_HOSTS = {"V6_01_V4best": "devbox", "V6_02_V5best": "wsl2"}
EXPECTED_CHECKPOINT_LABELS = {
    "V6_01_V4best": {"final", "legacy_best"},
    "V6_02_V5best": {
        "final",
        "legacy_best",
        "point_global_best",
        "base_mse_best",
        "sample_first_best",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0


def main() -> int:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "heat3d_v6_two_host_gpu_preflight_v1"
    assert payload["status"] == "passed"
    assert payload["formal_training_started"] is False
    assert payload["e600_started"] is False
    assert payload["roles_materialized"] == ["train", "valid_iid"]
    assert payload["forbidden_roles_materialized"] == []

    dataset = payload["dataset"]
    assert dataset["id"] == "heat3d_v6_p1g_geometry_deconfounded1024_v0"
    assert dataset["manifest_sha256"] == _sha256(MANIFEST)
    assert dataset["split_counts"] == {"test_iid": 128, "train": 768, "valid_iid": 128}
    assert dataset["group_locked"] is True
    assert dataset["node_count"] == 1024
    assert dataset["bottom_boundary_semantics"] == "robin_not_fixed_temperature"

    batching = payload["effective_batch"]
    assert batching == {
        "configured": 28,
        "effective": 28,
        "micro_max": 8,
        "updates_per_epoch": 28,
        "tail": 12,
    }

    runs = payload["runs"]
    assert set(runs) == CONFIG_IDS
    for config_id in sorted(CONFIG_IDS):
        run = runs[config_id]
        assert run["status"] in {"passed", "passed_recovered_post_export"}
        assert run["host"] == EXPECTED_HOSTS[config_id]
        assert run["epochs"] == 1
        assert run["random_initialization"] is True
        assert run["node_count"] == 1024
        assert run["sample_counts"] == {"train": 768, "valid_iid": 128}
        assert run["group_counts"] == {"train": 110, "valid_iid": 16, "test_iid": 0, "all": 0}
        assert run["optimizer_updates"] == 28
        assert run["finite_loss_gradient_update"] is True
        assert run["oom_nan_inf"] is False
        assert run["device"]["platform"] == "gpu"
        assert _finite_positive(run["device"]["peak_memory_mb"])
        assert run["device"]["peak_memory_mb"] < run["device"]["memory_limit_mb"]
        for key in ("group_build_s", "initial_loss_s", "epoch_s", "first_update_s", "steady_update_median_s"):
            assert _finite_positive(run["timing"][key]), f"{config_id}: invalid timing {key}"

        artifacts = run["artifacts"]
        assert set(artifacts) == EXPECTED_CHECKPOINT_LABELS[config_id]
        for record in artifacts.values():
            assert len(record["checkpoint_sha256"]) == 64
            assert len(record["predictions_sha256"]) == 64

        reload_audit = run["checkpoint_prediction_reload_audit"]
        assert reload_audit["status"] == "passed"
        assert {entry["label"] for entry in reload_audit["entries"]} == set(artifacts)
        for entry in reload_audit["entries"]:
            assert entry["passed"] is True
            assert entry["parameter_reload_max_abs_error"] == 0.0
            assert entry["npz_reload_max_abs_error_K"] == 0.0
            assert entry["checkpoint_reload_max_abs_error_K"] <= entry["max_abs_tolerance_K"]
            assert entry["checkpoint_reload_rmse_K"] <= entry["rmse_tolerance_K"]

        if config_id == "V6_01_V4best":
            assert run["global_context"] == {"enabled": False, "mode": "none"}
        else:
            context = run["global_context"]
            assert context["enabled"] is True and context["dimension"] == 24
            assert context["fit_population"] == "train_only"
            assert context["fit_sample_count"] == 768
            assert context["target_or_label_derived_inputs"] is False

        assert run["failed_attempts_preserved"]
        assert all(attempt["preserved"] for attempt in run["failed_attempts_preserved"])

    print(json.dumps({"status": "passed", "report": str(REPORT.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
