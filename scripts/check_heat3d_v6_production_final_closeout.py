#!/usr/bin/env python3
"""Deterministic checks for the frozen V6 production inference closeout."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6/v6_production_highres_inference.yaml"
FINAL = ROOT / "configs/heat3d_v6/v6_production_final_closeout.json"
GRAPH_MANIFEST = (
    ROOT / "configs/heat3d_v6/v6_production_graph_cache_manifest_v2.json"
)
BUNDLE_MANIFEST = ROOT / "configs/heat3d_v6/v6_production_bundle_manifest.json"
METRICS = ROOT / "configs/heat3d_v6/v6_production_multiseed_fullfield_metrics.csv"
TIMING = ROOT / "configs/heat3d_v6/v6_production_stage_timing.csv"
SPEEDUP = ROOT / "configs/heat3d_v6/v6_production_solver_speedup.csv"
ENVIRONMENT = ROOT / "configs/heat3d_v6/v6_production_environment.json"

FINGERPRINT = "270a55d098ca2591589c2391362f08ffdc9b56dc45f78fb18f30eaca061f52c6"
EVALUATOR_COMMIT = "f074f1bab6757bd528f134d2cda81af56d247881"
RESOLUTIONS = {1024, 2048, 4096, 8192, 16384, 32768}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_csv(rows, ignored):
    for row in rows:
        for key, value in row.items():
            if key in ignored or value in ("", None):
                continue
            assert math.isfinite(float(value)), (key, value)


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["evaluation_commit"] == EVALUATOR_COMMIT
    assert config["training_executed"] is False
    assert config["checkpoint_modified"] is False
    assert config["role_policy"] == {
        "allowed": ["valid_iid"],
        "forbidden": ["test_iid", "hard"],
    }
    assert config["graph"]["backend"] == "sparse_kdtree_v1"
    assert config["graph"]["graph_builder_code_fingerprint"] == FINGERPRINT
    assert config["graph"]["cache_key_fields"] == [
        "support_hash",
        "resolved_graph_config",
        "graph_seed",
        "graph_builder_code_fingerprint",
    ]
    assert config["resolution"] == {
        "default": 4096,
        "full_field_mode": 8192,
        "maximum_production_verified": 16384,
        "experimental_verified": [32768],
    }

    graph_manifest = _load(GRAPH_MANIFEST)
    assert graph_manifest["schema_version"] == "heat3d_v6_graph_cache_manifest_v2"
    assert graph_manifest["status"] == "frozen"
    assert len(graph_manifest["entries"]) == 6
    assert {int(row["resolution"]) for row in graph_manifest["entries"]} == RESOLUTIONS
    for row in graph_manifest["entries"]:
        payload = row["cache_key_payload"]
        assert set(payload) == {
            "schema_version",
            "support_hash",
            "graph_config",
            "graph_seed",
            "graph_builder_fingerprint",
        }
        assert "commit" not in payload
        assert payload["schema_version"] == "heat3d_graph_cache_key_v2"
        assert payload["graph_builder_fingerprint"] == FINGERPRINT
        assert payload["graph_config"]["discrete_graph_backend"] == "sparse_kdtree_v1"
        assert len(row["metadata_hash"]) == len(row["graph_hash"]) == 64

    final = _load(FINAL)
    assert final["status"] == "passed"
    assert final["evaluation_commit"] == EVALUATOR_COMMIT
    assert final["graph_builder_code_fingerprint"] == FINGERPRINT
    assert final["evaluation_role"] == "valid_iid"
    assert final["test_hard_accessed"] is False
    assert final["training_executed"] is False
    assert final["checkpoint_modified"] is False
    assert final["nonmatched_dof"] is True
    assert final["decision"]["default"] == 4096
    assert final["decision"]["full_field_mode"] == 8192
    assert final["decision"]["high_accuracy_limit"] == 16384
    assert final["decision"]["experimental"] == 32768
    assert set(final["multiseed"]) == {"seed0", "seed1", "seed2"}
    assert all(
        set(seed_rows) == {"4096", "8192", "16384"}
        for seed_rows in final["multiseed"].values()
    )
    assert final["preflight"]["status"] == "passed"
    assert final["runner_reuse_smoke"]["status"] == "passed"
    assert final["runner_reuse_smoke"]["legacy_metadata_build_calls"] == 8
    assert final["runner_reuse_smoke"]["reused_metadata_build_calls"] == 1
    assert final["runner_reuse_smoke"]["forward_max_abs_error_K"] == 0.0

    bundle = _load(BUNDLE_MANIFEST)
    assert bundle["status"] == "archived"
    assert bundle["training_executed"] is False
    assert bundle["test_hard_accessed"] is False
    assert bundle["graph_builder_code_fingerprint"] == FINGERPRINT
    assert len(bundle["files"]) == 16
    assert Path(bundle["archive_path"]).is_dir()

    metrics = _rows(METRICS)
    timing = _rows(TIMING)
    speedup = _rows(SPEEDUP)
    assert len(metrics) == 15
    assert len(timing) == 18
    assert len(speedup) == 6
    _finite_csv(metrics, {"row_type", "seed"})
    _finite_csv(timing, {"platform", "mode", "seed", "device_peak_memory_GB"})
    _finite_csv(speedup, {"nonmatched_dof"})
    assert {int(row["resolution"]) for row in speedup} == RESOLUTIONS
    assert all(row["nonmatched_dof"] == "True" for row in speedup)

    environment = _load(ENVIRONMENT)
    assert environment["local_cpu"]["device"]
    assert environment["gpu"]["device"] == "cuda:0"
    assert "RTX 5070" in environment["gpu"]["device_kind"]

    print(
        json.dumps(
            {
                "status": "passed",
                "evaluation_role": "valid_iid",
                "test_hard_accessed": False,
                "training_executed": False,
                "checkpoint_modified": False,
                "cache_key_commit_bound": False,
                "graph_cache_entries": len(graph_manifest["entries"]),
                "metric_rows": len(metrics),
                "timing_rows": len(timing),
                "speedup_rows": len(speedup),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
