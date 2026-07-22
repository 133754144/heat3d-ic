#!/usr/bin/env python3
"""Validate the frozen two-host effective-B24 V6 e1 preflight closeout."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


REPORT = ROOT / "configs/heat3d_v6/v6_training_handoff_b24_gpu_preflight.json"
B28_REPORT = ROOT / "configs/heat3d_v6/v6_training_handoff_gpu_preflight.json"
MANIFEST = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json"
FORMAL_CONFIGS = {
    "V6_01_V4best": ROOT / "configs/heat3d_v6/V6_01_V4best.yaml",
    "V6_02_V5best": ROOT / "configs/heat3d_v6/V6_02_V5best.yaml",
}
EXPECTED_HOSTS = {"V6_01_V4best": "devbox", "V6_02_V5best": "wsl2"}
EXPECTED_LABELS = {
    "V6_01_V4best": {"final", "best"},
    "V6_02_V5best": {
        "final", "best", "point_global_best", "base_mse_best", "sample_first_best"
    },
}


def _resolved(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return resolve_inherited_yaml(payload, path)


def _finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0


def main() -> int:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    b28 = json.loads(B28_REPORT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "heat3d_v6_two_host_b24_gpu_preflight_v1"
    assert payload["status"] == "passed"
    assert payload["formal_training_started"] is False
    assert payload["e600_started"] is False
    assert payload["roles_materialized"] == ["train", "valid_iid"]
    assert payload["forbidden_roles_materialized"] == []
    assert payload["dataset"]["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert payload["effective_batch"] == {
        "configured": 24,
        "effective": 24,
        "micro_batch_size": 8,
        "micro_batches_per_epoch": 96,
        "micro_batch_sample_counts_unique": [8],
        "micro_batches_per_update": 3,
        "updates_per_epoch": 32,
        "effective_batch_sample_counts_unique": [24],
        "tail": None,
        "geometry_split": False,
        "padding_policy": "repeat_existing_dummy_edges_only",
    }

    for config_id, path in FORMAL_CONFIGS.items():
        formal = _resolved(path)
        assert formal["run"]["epochs"] == 600
        assert formal["run"]["batch_size"] == 24
        # The report is the immutable historical 3xB8 preflight.  The current
        # formal contract has since moved to one native B24 forward/backward.
        assert formal["run"]["micro_batch_size"] == 24
        assert formal["run"]["validation_batch_size"] == 32
        assert formal["run"]["prediction_batch_size"] == 32
        assert formal["run"]["drop_last"] is False
        assert formal["metadata"]["training_started"] is False

        run = payload["runs"][config_id]
        assert run["status"] == "passed" and run["host"] == EXPECTED_HOSTS[config_id]
        assert run["training_commit"] == payload["preflight_commit"][:7]
        assert run["epochs"] == 1 and run["random_initialization"] is True
        assert run["sample_counts"] == {"train": 768, "valid_iid": 128}
        assert run["group_counts"] == {
            "train_micro_batches": 96, "valid_iid_batches": 4, "test_iid": 0, "all": 0
        }
        assert run["optimizer_updates"] == 32
        assert run["finite_loss_gradient_update"] is True and run["oom_nan_inf"] is False
        assert run["device"]["platform"] == "gpu"
        assert _finite_positive(run["device"]["peak_memory_mb"])
        assert run["device"]["peak_memory_mb"] < run["device"]["memory_limit_mb"]
        assert all(_finite_positive(value) for value in run["timing"].values())

        reload = run["checkpoint_prediction_reload_audit"]
        assert reload["status"] == "passed"
        assert {entry["label"] for entry in reload["entries"]} == EXPECTED_LABELS[config_id]
        for entry in reload["entries"]:
            assert entry["passed"] is True
            assert entry["parameter_reload_max_abs_error"] == 0.0
            assert entry["npz_reload_max_abs_error_K"] == 0.0
            assert entry["checkpoint_reload_max_abs_error_K"] <= 0.1
            assert entry["checkpoint_reload_rmse_K"] <= 0.01
        assert all(len(value) == 64 for value in run["artifact_sha256"].values())
        assert all(len(value) == 64 for value in run["evidence_sha256"].values())

        reference = b28["runs"][config_id]
        expected_epoch = 100.0 * (run["timing"]["epoch_s"] / reference["timing"]["epoch_s"] - 1.0)
        expected_memory = 100.0 * (
            run["device"]["peak_memory_mb"] / reference["device"]["peak_memory_mb"] - 1.0
        )
        assert math.isclose(run["b28_relative_change_pct"]["epoch_s"], expected_epoch)
        assert math.isclose(run["b28_relative_change_pct"]["peak_memory_mb"], expected_memory)

    context = payload["runs"]["V6_02_V5best"]["global_context"]
    assert context["enabled"] is True and context["dimension"] == 24
    assert context["fit_population"] == "train_only" and context["fit_sample_count"] == 768
    assert context["target_or_label_derived_inputs"] is False
    print(json.dumps({"status": "passed", "report": str(REPORT.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
