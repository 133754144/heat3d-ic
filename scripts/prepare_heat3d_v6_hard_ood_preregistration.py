#!/usr/bin/env python3
"""Freeze the label-independent V6 hard-stress role and evaluation contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6"
P1G_CONFIG = CONFIG / "v6_p1g_geometry_deconfounded1024.yaml"
P1H_MANIFEST = CONFIG / "v6_p1h_shared_support1024_manifest.json"
P1H_ACCEPTANCE = CONFIG / "v6_p1h_shared_support1024_acceptance.json"
LADDER = CONFIG / "v6_source_aware_resolution_ladder.json"
MODEL_CLOSEOUT = CONFIG / "v6_model_closeout_anchored_resolution.json"
EVALUATOR = ROOT / "scripts/run_heat3d_v6_final_performance.py"
ROLE_OUTPUT = CONFIG / "v6_hard_input_stress_role.json"
PREREG_OUTPUT = CONFIG / "v6_hard_ood_preregistration.json"
MARKDOWN_OUTPUT = ROOT / "docs/v6_hard_ood_preregistration.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    config = yaml.safe_load(P1G_CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(P1H_MANIFEST.read_text(encoding="utf-8"))
    acceptance = json.loads(P1H_ACCEPTANCE.read_text(encoding="utf-8"))
    ladder = json.loads(LADDER.read_text(encoding="utf-8"))
    model_closeout = json.loads(MODEL_CLOSEOUT.read_text(encoding="utf-8"))

    manifest_test = [
        row for row in manifest["samples"] if row["split_role"] == "test"
    ]
    manifest_test_ids = {str(row["sample_id"]) for row in manifest_test}
    cases = {
        str(row["id"]): row
        for row in config["cases"]
        if row["split_role"] == "test"
    }
    selected = [
        str(row["sample_id"])
        for row in manifest_test
        if (
            float(cases[str(row["sample_id"])]["top_h_W_m2K"]) == 1000.0
            and float(cases[str(row["sample_id"])]["bottom_h_W_m2K"]) == 20.0
            and float(cases[str(row["sample_id"])]["package_total_power_W"])
            == 6.0
        )
    ]
    if len(selected) != 16 or len(set(selected)) != 16:
        raise RuntimeError("hard input-stress role must contain 16 unique cases")
    if not set(selected).issubset(manifest_test_ids):
        raise RuntimeError("hard input-stress role escaped the test holdout")
    selected_groups = [
        str(cases[sample_id]["group_id"]) for sample_id in selected
    ]
    if len(set(selected_groups)) != 16:
        raise RuntimeError("hard input-stress role must use one case per test group")

    role = {
        "schema_version": "heat3d_v6_hard_input_stress_role_v1",
        "status": "frozen_before_hard_specific_metric_open",
        "role_id": "hard_input_stress_corner_v1",
        "role_classification": "input_defined_in_distribution_stress_subset",
        "parent_split_role": "test",
        "sample_count": 16,
        "sample_ids": selected,
        "group_ids": selected_groups,
        "selection_definition": {
            "top_h_W_m2K": 1000.0,
            "bottom_h_W_m2K": 20.0,
            "package_total_power_W": 6.0,
            "interpretation": "weakest cooling plus highest frozen package power",
        },
        "selection_sources": [
            _relative(P1G_CONFIG),
            _relative(P1H_MANIFEST),
        ],
        "selection_source_sha256": {
            _relative(P1G_CONFIG): _sha256(P1G_CONFIG),
            _relative(P1H_MANIFEST): _sha256(P1H_MANIFEST),
        },
        "selection_uses_target_labels": False,
        "selection_uses_model_errors": False,
        "is_distribution_shift_ood": False,
        "same_physical_labels_previously_included_in_corrected_holdout": True,
        "hard_specific_subgroup_metrics_computed_before_preregistration": False,
    }
    ROLE_OUTPUT.write_text(
        json.dumps(role, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    reference = next(
        row
        for row in model_closeout["three_seed_artifact_freeze"]["artifacts"]
        if row["seed"] == "0" and row["checkpoint_kind"] == "point_global_best"
    )
    commands = []
    for resolution in (4096, 8192, 16384):
        commands.append(
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
            "python scripts/run_heat3d_v6_final_performance.py "
            "--dataset data/heat3d_v6_p1h_shared_support1024_v0 "
            "--manifest configs/heat3d_v6/"
            "v6_p1h_shared_support1024_manifest.json "
            "--ladder configs/heat3d_v6/"
            "v6_source_aware_resolution_ladder.json "
            "--run-dir output/heat3d_v6_runs/V6_03_V5best_P1h "
            f"--resolution {resolution} --role hard_input_stress "
            "--role-manifest configs/heat3d_v6/"
            "v6_hard_input_stress_role.json --mode cached --batch-size 8 "
            "--graph-cache output/heat3d_v6_governance/graph_cache "
            "--reconstruction-cache "
            "output/heat3d_v6_governance/reconstruction_cache "
            "--serialization-dir output/heat3d_v6_governance/serialization "
            "--platform cpu --output output/heat3d_v6_governance/"
            f"hard_input_stress_{resolution}.json"
        )

    prereg = {
        "schema_version": "heat3d_v6_hard_ood_preregistration_v1",
        "status": "frozen_before_hard_specific_metric_open",
        "commit_binding": "commit_containing_this_file_must_be_pushed_before_run",
        "dataset": {
            "dataset_id": acceptance["dataset_id"],
            "manifest_path": _relative(P1H_MANIFEST),
            "manifest_sha256": _sha256(P1H_MANIFEST),
            "full_field_archive_sha256": acceptance[
                "full_field_archive_sha256"
            ],
            "shared_coordinate_sha256": acceptance[
                "shared_coordinate_sha256"
            ],
            "shared_graph_sha256": acceptance["shared_graph_sha256"],
        },
        "checkpoint": {
            "config_id": "V6_03_V5best_P1h",
            "seed": 0,
            "kind": "point_global_best",
            "epoch": int(reference["checkpoint_epoch"]),
            "sha256": reference["checkpoint_sha256"],
            "modified": False,
        },
        "workflow": {
            "name": "frozen_anchor_derived_source_aware",
            "steps": [
                "1024_anchor_forward",
                "anchor_derived_global_context_and_scale",
                "N_node_source_aware_forward",
                "anchor_scale_reconstruction",
                "layer_interface_knn_inverse_distance_v1_full_field_reconstruction",
            ],
            "ladder_path": _relative(LADDER),
            "ladder_sha256": _sha256(LADDER),
            "evaluator_path": _relative(EVALUATOR),
            "evaluator_sha256": _sha256(EVALUATOR),
            "resolutions": {
                "4096": "default_hotspot_oriented",
                "8192": "balanced_full_field",
                "16384": "maximum_full_field_accuracy",
            },
            "excluded_resolution": 32768,
        },
        "primary_metrics": [
            "support_point_global_cv_relative_rmse_pct",
            "support_sample_first_cv_relative_rmse_pct",
            "support_raw_cv_weighted_rmse_K",
            "support_field_cv_weighted_bias_K",
            "full_field_point_global_cv_relative_rmse_pct",
            "full_field_sample_first_cv_relative_rmse_pct",
            "full_field_raw_cv_weighted_rmse_K",
            "full_field_peak_error_rmse_K",
            "full_field_source_cv_weighted_rmse_K",
            "full_field_layer_cv_weighted_rmse_K",
            "full_field_interface_cv_weighted_rmse_K",
            "full_field_top_cv_weighted_rmse_K",
            "full_field_bottom_cv_weighted_rmse_K",
        ],
        "roles": {
            "valid_iid": "frozen_existing_result_only",
            "corrected_confirmatory_holdout": "frozen_existing_result_only",
            "hard_input_stress": {
                "role_manifest": _relative(ROLE_OUTPUT),
                "role_manifest_sha256": _sha256(ROLE_OUTPUT),
                "sample_count": 16,
                "run_once_after_preregistration_push": True,
            },
            "canonical_ood": {
                "status": "not_available",
                "evidence": (
                    "configs/heat3d_v6/"
                    "v6_p1g_geometry_deconfounded1024.yaml#"
                    "split_contract.OOD_roles=[]"
                ),
                "reason": (
                    "P1h inherits P1g train/valid/test only. Archived P1e OOD "
                    "uses a different dataset/support and is outside the "
                    "canonical checkpoint applicability boundary."
                ),
                "labels_must_not_be_accessed": True,
            },
        },
        "prior_holdout_governance": {
            "classification": "corrected_confirmatory_holdout",
            "labels_already_opened_before_this_preregistration": True,
            "hard_specific_subgroup_metrics_previously_computed": False,
            "protocol_deviation_record": (
                "configs/heat3d_v6/v6_final_performance_closeout.json#"
                "test_opening_audit"
            ),
        },
        "selection_policy": {
            "hard_or_ood_used_for_model_checkpoint_resolution_selection": False,
            "posthoc_reselection_or_tuning_allowed": False,
            "results_are_descriptive_only": True,
        },
        "command_plan": commands,
        "training_allowed": False,
        "checkpoint_sampling_graph_reconstruction_changes_allowed": False,
        "hard_labels_read_by_preregistration_generator": False,
        "ood_labels_read_by_preregistration_generator": False,
    }
    PREREG_OUTPUT.write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# V6 hard/OOD preregistration",
        "",
        "Status: frozen before hard-specific subgroup metrics.",
        "",
        "The canonical P1h dataset has no registered OOD role. Earlier P1e OOD "
        "roles use a different dataset and support, so they are not silently "
        "reused here.",
        "",
        "The only executable stress role is `hard_input_stress_corner_v1`: "
        "16 test-holdout cases with top h=1000 W/(m²·K), bottom h=20 "
        "W/(m²·K), and package power=6 W. Selection uses only the frozen "
        "input case table and manifest, never target temperature or model error.",
        "",
        "The underlying test labels were already opened in the corrected "
        "confirmatory holdout. This gate freezes the subgroup before any "
        "hard-specific metric is computed; it cannot retroactively claim that "
        "the physical labels were never read.",
        "",
        "4096 is default/hotspot-oriented, 8192 is balanced full-field, and "
        "16384 is the maximum full-field accuracy mode. 32768 is excluded.",
        "",
        "No hard/OOD result may change model, checkpoint, resolution, graph, "
        "reconstruction, or any later tuning decision.",
        "",
    ]
    MARKDOWN_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "hard_sample_count": len(selected),
                "canonical_ood_status": "not_available",
                "labels_read": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
