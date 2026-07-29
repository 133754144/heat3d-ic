#!/usr/bin/env python3
"""Build the authoritative V6 governance report and machine-readable manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6"
DOCS = ROOT / "docs"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _last_commit(path: Path) -> str:
    relative = _relative(path)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        return "commit_containing_this_file"
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "commit_containing_this_file"


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "last_committed_at": _last_commit(path),
    }


def main() -> int:
    acceptance_path = CONFIG / "v6_p1h_shared_support1024_acceptance.json"
    manifest_path = CONFIG / "v6_p1h_shared_support1024_manifest.json"
    ladder_path = CONFIG / "v6_source_aware_resolution_ladder.json"
    model_path = CONFIG / "v6_model_closeout_anchored_resolution.json"
    production_path = CONFIG / "v6_production_final_closeout.json"
    performance_path = CONFIG / "v6_final_performance_closeout.json"
    bundle_path = CONFIG / "v6_production_bundle_manifest.json"
    prereg_path = CONFIG / "v6_hard_ood_preregistration.json"
    role_path = CONFIG / "v6_hard_input_stress_role.json"
    preflight_path = CONFIG / "v6_hard_ood_preflight.json"
    adapter_path = CONFIG / "v6_hard_ood_evaluator_adapter.json"
    hard_metrics_path = CONFIG / "v6_hard_ood_metrics.csv"
    hard_doc_path = DOCS / "v6_hard_ood_closeout.md"
    training_path = CONFIG / "v6_latest_training_results.json"

    acceptance = _load(acceptance_path)
    ladder = _load(ladder_path)
    model = _load(model_path)
    production = _load(production_path)
    performance = _load(performance_path)
    bundle = _load(bundle_path)
    prereg = _load(prereg_path)
    training = _load(training_path)
    hard_path = CONFIG / "v6_hard_ood_closeout.json"
    hard = _load(hard_path) if hard_path.is_file() else None

    v604 = next(
        row
        for row in training["checkpoint_rows"]
        if row["config_id"] == "V6_04_V5best_P1h_DualAttention"
        and row["checkpoint_kind"] == "point_global_best"
    )
    three_seed = {
        seed: {
            str(resolution): production["multiseed"][seed][str(resolution)]
            for resolution in (4096, 8192, 16384)
        }
        for seed in ("seed0", "seed1", "seed2")
    }
    key_artifacts = [
        acceptance_path,
        manifest_path,
        ladder_path,
        model_path,
        production_path,
        performance_path,
        bundle_path,
        CONFIG / "v6_production_graph_cache_manifest_v2.json",
        CONFIG / "v6_final_performance_timing.csv",
        CONFIG / "v6_final_persistent_gpu.csv",
        CONFIG / "v6_final_solver_inference_comparison.csv",
        CONFIG / "v6_final_corrected_confirmatory_holdout_metrics.csv",
        prereg_path,
        role_path,
        preflight_path,
        adapter_path,
    ]
    if hard_path.is_file():
        key_artifacts.extend([hard_path, hard_metrics_path, hard_doc_path])

    machine_manifest = {
        "schema_version": "heat3d_v6_total_governance_manifest_v1",
        "status": "closed" if hard is not None else "preregistered_pending_hard_stress",
        "governance_commit_binding": "commit_containing_this_file",
        "canonical_dataset": {
            "dataset_id": acceptance["dataset_id"],
            "sample_count": acceptance["sample_count"],
            "group_count": acceptance["group_count"],
            "split_counts": acceptance["split_counts"],
            "manifest_sha256": _sha256(manifest_path),
            "full_field_archive_sha256": acceptance[
                "full_field_archive_sha256"
            ],
            "shared_coordinate_sha256": acceptance[
                "shared_coordinate_sha256"
            ],
            "shared_graph_sha256": acceptance["shared_graph_sha256"],
            "shared_support_index_sha256": acceptance[
                "shared_support_index_sha256"
            ],
            "repository_binding": "data/heat3d_v6_p1h_shared_support1024_v0",
            "remote_archive": (
                "hf://datasets/133754144X/heat3d-thermal-simulation/"
                "subsets/heat3d_v6_p1h_shared_support1024_v0"
            ),
        },
        "canonical_model": {
            "config_id": "V6_03_V5best_P1h",
            "reference_seed": 0,
            "reference_checkpoint_kind": "point_global_best",
            "reference_checkpoint_epoch": 111,
            "reference_checkpoint_sha256": (
                "3ad58c2b34a46481acb74722c80bdcadb"
                "f55a0d613bc25c4fe2d7646b91aa1f2"
            ),
            "replication_seeds": [1, 2],
            "ablation": {
                "config_id": "V6_04_V5best_P1h_DualAttention",
                "checkpoint_epoch": int(v604["checkpoint_epoch"]),
                "checkpoint_sha256": v604["checkpoint_sha256"],
                "status": "registered_ablation_not_canonical",
            },
            "applicability": "P1h source-aware support family only",
        },
        "three_seed_freeze": {
            "checkpoint_prediction_pair_count": model[
                "three_seed_artifact_freeze"
            ]["checkpoint_prediction_pair_count"],
            "artifacts": model["three_seed_artifact_freeze"]["artifacts"],
            "high_resolution_results": three_seed,
        },
        "source_aware_ladder": {
            "path": _relative(ladder_path),
            "sha256": _sha256(ladder_path),
            "resolutions": [4096, 8192, 16384],
            "role_names": {
                "4096": "default_hotspot_oriented",
                "8192": "balanced_full_field",
                "16384": "maximum_full_field_accuracy",
            },
            "32768": "experimental_excluded",
            "probe_hashes": {
                str(resolution): {
                    "coordinate_sha256": ladder["probes"][str(resolution)][
                        "coordinate_sha256"
                    ],
                    "indices_sha256": ladder["probes"][str(resolution)][
                        "indices_sha256"
                    ],
                }
                for resolution in (4096, 8192, 16384)
            },
        },
        "production_workflow": {
            "name": "anchor_derived_source_aware_v1",
            "conditioning_support": "1024 source-aware anchors",
            "query_support": "4096/8192/16384 source-aware solver nodes",
            "global_context_and_scale": "anchor-derived",
            "reconstruction": (
                "layer_interface_knn_inverse_distance_v1_to_240825_nodes"
            ),
            "graph_backend": "sparse_kdtree_v1_edge_list_graph_cache",
            "bundle": {
                "status": bundle["status"],
                "repository_location": "output/v6_production_inference_bundle_f074f1b",
                "manifest_sha256": _sha256(bundle_path),
            },
        },
        "governance": {
            "confirmatory_holdout_classification": (
                "corrected_confirmatory_holdout"
            ),
            "protocol_deviation": performance["protocol_deviation"],
            "legal_structured_fvm_mesh_sensitivity": {
                "status": performance[
                    "legal_structured_fvm_mesh_sensitivity"
                ]["status"],
                "nonmatched_dof": True,
            },
            "hard_ood_preregistration": _artifact(prereg_path),
            "hard_ood_preflight": _artifact(preflight_path),
            "hard_ood_evaluator_adapter": _artifact(adapter_path),
            "hard_stress_result": (
                _artifact(hard_path) if hard_path.is_file() else None
            ),
            "canonical_ood_status": "not_available",
        },
        "hardware_scope": {
            "cpu": {
                "model": "Apple M4",
                "physical_cores": 10,
                "core_configuration": "4 performance + 6 efficiency",
                "memory_GB": 16,
            },
            "gpu": {
                "model": "NVIDIA GeForce RTX 5070",
                "memory_limit_GB": 9.61536,
                "hosts": ["devbox", "wsl2"],
            },
            "speedup_semantics": {
                "CPU_to_CPU": (
                    "local Apple-M4 CPU production latency divided into "
                    "240825-node FVM CPU cold/warm latency"
                ),
                "GPU_to_CPU": (
                    "RTX-5070 GPU production latency divided into 240825-node "
                    "FVM CPU cold/warm latency"
                ),
                "nonmatched_DOF": True,
            },
        },
        "applicability_and_limits": [
            "Only the P1h source-aware support family is covered.",
            "Added query nodes still participate in the frozen joint encoder/processor path.",
            "The corrected confirmatory holdout is not a fresh untouched test set.",
            "Canonical P1h has no registered distribution-shift OOD role.",
            "FVM/model comparisons use nonmatched degrees of freedom and hardware.",
            "No uncertainty calibration or real-package experimental validation is claimed.",
        ],
        "official_artifacts": [_artifact(path) for path in key_artifacts],
        "local_absolute_paths_allowed": False,
        "training_executed_by_governance_closeout": False,
        "checkpoint_sampling_graph_reconstruction_modified": False,
    }
    manifest_output = CONFIG / "v6_total_governance_manifest.json"
    manifest_output.write_text(
        json.dumps(machine_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    amendment = {
        "schema_version": "heat3d_v6_governance_amendment_v1",
        "status": "effective",
        "historical_results_unchanged": True,
        "terminology": {
            "test_iid": "corrected_confirmatory_holdout",
            "matched_accuracy_fvm": (
                "legal_structured_FVM_mesh_sensitivity"
            ),
            "4096": "default_hotspot_oriented",
            "8192": "balanced_full_field",
            "16384": "maximum_full_field_accuracy",
        },
        "protocol_deviation": performance["protocol_deviation"],
        "speedup_semantics": machine_manifest["hardware_scope"][
            "speedup_semantics"
        ],
        "cpu_hardware": machine_manifest["hardware_scope"]["cpu"],
        "canonical_ood_status": "not_available",
        "governance_manifest": _relative(manifest_output),
        "local_absolute_paths_allowed_in_new_governance_outputs": False,
    }
    amendment_output = CONFIG / "v6_governance_amendment.json"
    amendment_output.write_text(
        json.dumps(amendment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    perf = production["mean_std"]
    lines = [
        "# V6 total closeout",
        "",
        f"Status: **{machine_manifest['status']}**.",
        "",
        "This is the authoritative V6 summary. Earlier phase reports remain "
        "evidence records and are not rewritten.",
        "",
        "## Canonical dataset and model",
        "",
        "- Dataset: `heat3d_v6_p1h_shared_support1024_v0`, 1024 cases, "
        "128 geometry groups, group-locked 768/128/128 splits.",
        "- Model: `V6_03_V5best_P1h`; seed0 e111 point-global checkpoint is "
        "the reference. Seeds 1/2 are replications.",
        "- `V6_04_V5best_P1h_DualAttention` remains a registered ablation.",
        "- Applicability is limited to the P1h source-aware support family.",
        "",
        "## Frozen production workflow",
        "",
        "1024 source-aware conditioning anchors provide Global Context and "
        "scale. Added 4096/8192/16384 source-aware query nodes use the frozen "
        "Anchor-derived workflow, sparse KD-tree edge-list graph cache, and "
        "layer/interface-aware reconstruction to 240825 solver nodes.",
        "",
        "- 4096: default/hotspot-oriented.",
        "- 8192: balanced full-field.",
        "- 16384: maximum full-field accuracy.",
        "- 32768: experimental and excluded from formal holdout/hard tables.",
        "",
        "## Three-seed valid_iid full-field performance",
        "",
        "| Mode | Full RMSE mean±std K | Point-global mean±std % |",
        "|---:|---:|---:|",
    ]
    for resolution in (4096, 8192, 16384):
        row = perf[str(resolution)]
        lines.append(
            f"| {resolution} | {row['mean']['full_raw_cv_rmse_K']:.4f}±"
            f"{row['std']['full_raw_cv_rmse_K']:.4f} | "
            f"{row['mean']['full_point_global_pct']:.4f}±"
            f"{row['std']['full_point_global_pct']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Governance",
            "",
            "- The previously opened test split is a `corrected confirmatory "
            "holdout`, not a pristine test set.",
            "- The wrong-ladder temporary outputs are excluded by SHA and were "
            "never used for selection.",
            "- FVM results are `legal structured-FVM mesh sensitivity`, not a "
            "matched-accuracy claim.",
            "- CPU→CPU and GPU→CPU speedups use the same 240825-node CPU FVM "
            "cold/warm denominator and remain nonmatched-DOF.",
            "- Canonical P1h contains no true OOD role. The preregistered hard "
            "corner is an input-defined in-distribution stress subset.",
        ]
    )
    if hard is not None:
        hard_rows = {
            int(row["resolution"]): row
            for row in hard["metric_rows"]
            if row["role"] == "hard_input_stress"
        }
        lines.extend(
            [
                "- The hard-stress subgroup was opened once after the "
                "preregistration and label-free preflight; it was not used "
                "for selection or tuning.",
                "",
                "## Frozen hard-stress descriptive result",
                "",
                "| Mode | Full RMSE K | Point-global % | Source RMSE K | "
                "Bottom RMSE K |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for resolution in (4096, 8192, 16384):
            row = hard_rows[resolution]
            lines.append(
                f"| {resolution} | {row['full_raw_cv_rmse_K']:.4f} | "
                f"{row['full_point_global_pct']:.4f} | "
                f"{row['full_source_rmse_K']:.4f} | "
                f"{row['full_bottom_rmse_K']:.4f} |"
            )
        lines.extend(
            [
                "",
                "Lower hard-stress relative errors partly reflect the larger "
                "target-energy denominator of high-power, weak-cooling cases; "
                "they do not establish that the subgroup is intrinsically "
                "easier. Canonical distribution-shift OOD remains unavailable "
                "and was not run.",
                "",
            ]
        )
    lines.extend(
        [
            "## Remaining limits",
            "",
        ]
    )
    lines.extend(
        f"- {item}" for item in machine_manifest["applicability_and_limits"]
    )
    lines.append("")
    (DOCS / "v6_total_closeout.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    amendment_lines = [
        "# V6 governance amendment",
        "",
        "- `test_iid` is classified as a corrected confirmatory holdout.",
        "- The wrong-ladder protocol deviation, temporary-result hashes, "
        "exclusion, and non-selection evidence are bound in the JSON amendment.",
        "- `matched-accuracy FVM` is replaced by `legal structured-FVM mesh "
        "sensitivity`.",
        "- 4096/8192/16384 mean default/hotspot-oriented, balanced full-field, "
        "and maximum full-field accuracy.",
        "- CPU→CPU and GPU→CPU speedups are distinct and nonmatched-DOF.",
        "- Local CPU: Apple M4, 10 cores, 16 GB memory.",
        "",
    ]
    (DOCS / "v6_governance_amendment.md").write_text(
        "\n".join(amendment_lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": machine_manifest["status"],
                "official_artifact_count": len(key_artifacts),
                "hard_result_present": hard is not None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
