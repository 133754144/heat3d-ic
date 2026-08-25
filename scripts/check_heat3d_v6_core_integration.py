#!/usr/bin/env python3
"""Validate the allowlisted V6 core integration on a clean main checkout."""

from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_graph_cache import graph_hash, metadata_hash  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v6_dataset import (  # noqa: E402
    V6_DUAL_ROBIN_CONDITION_FEATURES,
)
from rigno.heat3d_v6_global_context import (  # noqa: E402
    GLOBAL_CONTEXT_FEATURES_V6,
)
from run_heat3d_v4_config import (  # noqa: E402
    DEFAULT_TRAINING_PROFILE,
    _load_config,
    _selected_config_path,
)


CONFIG = ROOT / "configs/heat3d_v6"
MANIFEST_PATH = CONFIG / "v6_core_integration_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scientific_payload(config: dict) -> dict:
    payload = copy.deepcopy(config)
    payload.pop("config_id", None)
    payload.pop("description", None)
    payload.pop("metadata", None)
    payload.get("export", {}).pop("output_dir", None)
    payload.get("export", {}).pop("run_name", None)
    return payload


def _listed_paths(manifest: dict) -> set[str]:
    paths = {MANIFEST_PATH.relative_to(ROOT).as_posix()}
    for rows in manifest["include"].values():
        paths.update(rows)
    paths.add("docs/v6_core_integration.md")
    return paths


def _check_graph_backend_equivalence() -> dict[str, str]:
    x = np.linspace(0.0, 1.0, 8)
    y = np.linspace(0.0, 1.0, 8)
    z = np.linspace(0.0, 0.2, 4)
    coords = np.stack(
        np.meshgrid(x, y, z, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    hashes = {}
    for backend in ("dense_reference", "sparse_kdtree_v1"):
        builder = Heat3DGraphBuilder(
            rmesh_levels=2,
            subsample_factor=4,
            discrete_graph_backend=backend,
        )
        metadata = builder.build_metadata(
            coords, key=jax.random.PRNGKey(0)
        )
        graphs = builder.build_graphs(metadata)
        hashes[backend] = (
            f"{metadata_hash(metadata)}:{graph_hash(graphs)}"
        )
    assert hashes["dense_reference"] == hashes["sparse_kdtree_v1"]
    return hashes


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "validated_ready_for_pr"
    clean_validation = manifest["clean_checkout_validation"]
    assert clean_validation["v5_checker_count"] == 6
    assert clean_validation["v5_canonical_hashes_unchanged"] is True
    assert clean_validation["v6_core_checker"] == "passed"
    assert clean_validation["v6_production_preflight"] == "passed"
    assert clean_validation["v6_graph_resolutions_checked"] == [
        1024,
        4096,
        8192,
        16384,
    ]
    assert clean_validation["training_executed"] is False
    assert clean_validation["test_hard_accessed"] is False
    assert manifest["integration_policy"] == {
        "full_research_history_merged": False,
        "stable_allowlist_only": True,
        "training_executed": False,
    }
    assert manifest["base_main_commit"] == (
        "159d3490be661bda9dcabbd2fce7a20de7ebb734"
    )
    assert manifest["research_evidence_archive"]["commit"] == (
        "d7f72f157ecfad0db63658755c8f72b296a16674"
    )

    included = _listed_paths(manifest)
    missing = sorted(path for path in included if not (ROOT / path).is_file())
    assert not missing, missing
    actual_configs = {
        path.relative_to(ROOT).as_posix()
        for path in CONFIG.iterdir()
        if path.is_file()
    }
    allowed_configs = {
        path for path in included if path.startswith("configs/heat3d_v6/")
    }
    assert actual_configs == allowed_configs, {
        "unexpected": sorted(actual_configs - allowed_configs),
        "missing": sorted(allowed_configs - actual_configs),
    }
    assert not any(
        (ROOT / name).exists()
        for name in (
            "data/heat3d_v6_p1h_shared_support1024_v0",
            "output/heat3d_v6_runs",
            "checkpoints",
            "logs",
        )
    )
    assert not list(CONFIG.glob("V6_0[124]*"))
    assert not (CONFIG / "preflight").exists()
    assert not (CONFIG / "resolved").exists()

    config_path = ROOT / manifest["canonical_config"]["path"]
    config = _load_config(config_path)
    scientific_sha = _canonical_json_sha(_scientific_payload(config))
    assert scientific_sha == manifest["canonical_config"][
        "scientific_payload_sha256"
    ]
    command = build_training_command(config, python_executable="python")
    assert _canonical_json_sha(command) == manifest["canonical_config"][
        "command_sha256"
    ]
    assert "--dataset-loader" in command
    assert "v6_dual_robin_manifest_v1" in command
    assert "--dataset-manifest" in command
    assert "--micro-batch-size" in command
    assert command[command.index("--micro-batch-size") + 1] == "24"
    assert command[command.index("--batch-size") + 1] == "24"
    assert command[command.index("--epochs") + 1] == "600"
    assert config["run"]["init_checkpoint"] is None
    assert config["metadata"]["training_started"] is False
    assert _selected_config_path(None) == DEFAULT_TRAINING_PROFILE

    dataset_manifest = json.loads(
        (ROOT / manifest["canonical_dataset"]["manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert _sha256(
        ROOT / manifest["canonical_dataset"]["manifest_path"]
    ) == manifest["canonical_dataset"]["manifest_sha256"]
    assert dataset_manifest["sample_count"] == 1024
    assert dataset_manifest["group_count"] == 128
    assert Counter(
        row["split_role"] for row in dataset_manifest["samples"]
    ) == {
        "train": 768,
        "valid": 128,
        "test": 128,
    }
    assert len(
        {row["point_coordinates_sha256"] for row in dataset_manifest["samples"]}
    ) == 1
    assert len({row["graph_sha256"] for row in dataset_manifest["samples"]}) == 1
    assert tuple(config["dataset"]["input_feature_contract"]) == (
        V6_DUAL_ROBIN_CONDITION_FEATURES
    )
    assert tuple(config["model"]["global_context_feature_names"]) == (
        GLOBAL_CONTEXT_FEATURES_V6
    )

    phase = json.loads(
        (CONFIG / "v6_phase_index.json").read_text(encoding="utf-8")
    )
    expected_terms = dict(manifest["governance"])
    expected_terms.pop("true_ood_available")
    expected_terms.update({
        "resolution_16384": (
            "frozen P1i reference operating point selected by preregistered "
            "non-inferiority plus latency-Pareto criteria"
        ),
        "resolution_32768": (
            "exploratory only; marginal point-global improvement with no "
            "consistent source, peak, latency, or memory benefit"
        ),
        "sealed_iid": (
            "post-development final confirmation; ungenerated and unopened"
        ),
    })
    assert phase["governance_terms"] == expected_terms
    assert manifest["governance"]["true_ood_available"] is False
    assert phase["test_and_hard"]["hard_used_for_selection"] is False
    assert phase["test_and_hard"]["test_used_for_selection"] is False
    graph_hashes = _check_graph_backend_equivalence()

    print(
        json.dumps(
            {
                "status": "passed",
                "included_files": len(included),
                "scientific_payload_sha256": scientific_sha,
                "command_sha256": _canonical_json_sha(command),
                "dataset_manifest_sha256": _sha256(
                    ROOT / manifest["canonical_dataset"]["manifest_path"]
                ),
                "graph_backend_equivalent": True,
                "graph_fixture_hash": graph_hashes["dense_reference"],
                "v5_default_unchanged": True,
                "training_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
