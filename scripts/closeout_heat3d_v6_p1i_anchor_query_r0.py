#!/usr/bin/env python3
"""Close out three-seed R0 and freeze high-N binding only after all pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def r0_report(payload: dict[str, Any]) -> str:
    lines = [
        "# P1i three-seed 1024 R0 equivalence gate",
        "",
        f"Status: **{payload['status']}**.",
        "",
        "The hard gate is prediction-level. Aggregate metric agreement is only a secondary consistency check.",
        "",
        "| seed | epoch | adapter-reference max K | archived RMSE K | full-field archived RMSE K | support PG % | full PG % |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seeds"]:
        lines.append(
            f"| {row['seed']} | {row['checkpoint_epoch']} | "
            f"{row['adapter_reference_max_abs_K']:.9g} | {row['archived_rmse_K']:.9g} | "
            f"{row['archived_full_rmse_K']:.9g} | {row['support_point_global_pct']:.6f} | "
            f"{row['full_point_global_pct']:.6f} |"
        )
    lines += [
        "",
        "All original support/order/features, real-edge graph semantics, anchor-derived context/native scale/QK inputs, predictions, and 240825-node reconstructions passed the frozen checks.",
        "",
        "Roles: train inputs were used only to replay the frozen train-only standardizer; valid_iid was evaluated. test/sealed were not accessed. No training, tuning, checkpoint mutation, or high-N inference occurred.",
    ]
    return "\n".join(lines) + "\n"


def binding_report(payload: dict[str, Any]) -> str:
    subset = payload["development_subset"]
    return "\n".join([
        "# P1i high-N implementation binding",
        "",
        f"Status: **{payload['status']}** (released only after all three 1024 R0 gates passed).",
        "",
        "## Frozen resolution and selection",
        "",
        "- mandatory prefixes: 1024, 4096, 8192, 16384; optional 32768 is valid-only and excluded from mandatory ranking.",
        "- every sample keeps its exact ordered 1024 anchors; added nodes are one deterministic solver-index sequence whose prefixes define all resolutions.",
        "- node selection uses no temperature, target, model prediction, or error.",
        "",
        "## Field and measure binding",
        "",
        "- coords/control-volume/layer come directly from the frozen full-field sidecar.",
        "- this sidecar does not persist full k/q; k/q therefore fail closed to deterministic reconstruction from frozen sample metadata and the fingerprinted continuous-field implementation, with power error <=1e-12.",
        "- selected k/q are direct values at solver indices; effective operator CV is the conservative same-layer nearest-node partition of all solver control volumes.",
        "",
        "## Development subset and timing",
        "",
        f"- fixed valid_iid subset: {subset['count']} IDs selected by SHA256(sample_id), independent of labels and model error.",
        "- R0 validation cost includes duplicate reference/adapter graph and forward work plus array/reconstruction audits; it is not production timing.",
        "- no 4096/8192/16384/32768 inference was executed in this closeout.",
    ]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-result", action="append", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-r0-json", type=Path, required=True)
    parser.add_argument("--output-r0-md", type=Path, required=True)
    parser.add_argument("--output-binding-json", type=Path, required=True)
    parser.add_argument("--output-binding-md", type=Path, required=True)
    args = parser.parse_args()

    raw = [json.loads(path.read_text()) for path in args.seed_result]
    if {int(row["seed"]) for row in raw} != {0, 1, 2} or len(raw) != 3:
        raise RuntimeError("R0 closeout requires exactly seeds 0/1/2")
    if any(row["status"] != "passed" or not all(row["checks"].values()) for row in raw):
        raise RuntimeError("Stage B is blocked because at least one R0 gate failed")
    for row in raw:
        roles = row["role_contract"]
        if roles["test_accessed"] or roles["sealed_accessed"] or roles["training_executed"] or roles["high_n_inference_executed"]:
            raise RuntimeError("R0 role contract failed")

    seeds = []
    for path, row in sorted(zip(args.seed_result, raw, strict=True), key=lambda pair: int(pair[1]["seed"])):
        metrics = row["metrics_secondary"]
        seeds.append({
            "seed": int(row["seed"]),
            "config_id": row["config_id"],
            "checkpoint_epoch": int(row["checkpoint"]["epoch"]),
            "checkpoint_sha256": row["checkpoint"]["sha256"],
            "raw_result_path": str(path),
            "raw_result_sha256": sha256(path),
            "all_checks_passed": all(row["checks"].values()),
            "adapter_reference_max_abs_K": row["prediction_equivalence"]["adapter_vs_reference"]["max_abs_error_K"],
            "adapter_reference_scale_max_abs": row["feature_and_scale_equivalence"]["predicted_scale"]["max_abs_error"],
            "adapter_reference_full_max_abs_K": row["full_field_reconstruction_equivalence"]["adapter_vs_reference"]["max_abs_error_K"],
            "archived_rmse_K": row["prediction_equivalence"]["adapter_vs_archived"]["rmse_K"],
            "archived_full_rmse_K": row["full_field_reconstruction_equivalence"]["adapter_vs_archived"]["rmse_K"],
            "support_point_global_pct": metrics["support"]["point_global_true_rms_relative_rmse_pct"],
            "support_sample_first_pct": metrics["support"]["sample_first_cv_relative_rmse_pct"],
            "support_raw_cv_rmse_K": metrics["support"]["raw_cv_weighted_rmse_K"],
            "full_point_global_pct": metrics["full_240825"]["point_global_true_rms_relative_rmse_pct"],
            "full_sample_first_pct": metrics["full_240825"]["sample_first_cv_relative_rmse_pct"],
            "full_raw_cv_rmse_K": metrics["full_240825"]["raw_cv_weighted_rmse_K"],
            "input_sample_count": row["input_equivalence"]["sample_count"],
            "graph_sample_count": row["graph_equivalence"]["sample_count"],
            "mapping_sample_count": row["full_field_reconstruction_equivalence"]["mapping_sample_count"],
        })
    r0 = {
        "schema_version": "heat3d_v6_p1i_anchor_query_r0_closeout_v1",
        "status": "passed_three_seed_prediction_level_equivalence",
        "gate_contract_path": str(args.contract),
        "gate_contract_sha256": sha256(args.contract),
        "manifest_sha256": sha256(args.manifest),
        "seed_count": 3,
        "valid_iid_count_per_seed": 128,
        "seeds": seeds,
        "stage_b_released": True,
        "aggregate_metrics_are_secondary": True,
        "role_contract": {
            "accessed_roles": ["train_inputs_for_frozen_standardizer", "valid_iid"],
            "test_accessed": False, "sealed_accessed": False,
            "training_executed": False, "checkpoint_modified": False,
            "high_n_inference_executed": False,
        },
    }
    write_json(args.output_r0_json, r0)
    args.output_r0_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_r0_md.write_text(r0_report(r0), encoding="utf-8")

    manifest = json.loads(args.manifest.read_text())
    valid_ids = [str(row["sample_id"]) for row in manifest["samples"] if row["split_role"] == "valid_iid"]
    development_ids = sorted(valid_ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:32]
    repo = Path(__file__).resolve().parents[1]
    code_paths = {
        "adapter_and_selector": repo / "rigno/heat3d_v6_p1i_anchor_query.py",
        "graph_builder": repo / "rigno/graphBuilder_Heat3D.py",
        "reconstruction": repo / "rigno/heat3d_v6_full_field.py",
        "full_kq_reconstruction": repo / "scripts/benchmark_heat3d_v6_p1i_resolution.py",
        "mesh_core": repo / "scripts/heat3d_v6_randomblock_core.py",
    }
    binding = {
        "schema_version": "heat3d_v6_p1i_high_n_implementation_binding_v1",
        "status": "frozen_after_three_seed_r0_pass",
        "r0_closeout_path": str(args.output_r0_json),
        "r0_closeout_sha256": sha256(args.output_r0_json),
        "prior_protocol_path": str(args.protocol),
        "prior_protocol_sha256": sha256(args.protocol),
        "code_fingerprints": {name: {"path": str(path.relative_to(repo)), "sha256": sha256(path)} for name, path in code_paths.items()},
        "resolutions": {"mandatory": [1024, 4096, 8192, 16384],
                        "optional_valid_only": 32768, "optional_enters_mandatory_ranking": False},
        "nested_support": {
            "implementation": "rigno.heat3d_v6_p1i_anchor_query.deterministic_nested_query_order",
            "algorithm": "anchored_stratified_deficit_round_robin_v1",
            "selection_seed": 20260808,
            "ordering": "exact original 1024 anchor indices in original order, then one deterministic added-node sequence; N uses the first N indices",
            "stratum_fractions": {"source": 0.35, "interface": 0.15, "robin": 0.10, "volume": 0.40},
            "within_stratum": "per-layer SHA256(seed:sample_id:solver_index), volume-weighted deficit interleave",
            "fallback": "exhausted strata/layers removed; active weights renormalized; no replacement or target-dependent retry",
            "target_temperature_prediction_or_error_used": False,
        },
        "high_n_fields": {
            "coords_control_volume_layer": "direct shared/coords_m, shared/control_volume_m3, shared/layer_id from frozen full_fields.h5",
            "full_field_sidecar_persists_k_q": False,
            "k_q_fallback": "deterministically reconstruct full solver-node k/q from frozen sample_meta physics using fingerprinted mesh and _continuous_fields; fail closed on fingerprint or power-audit drift",
            "selected_k_q": "direct values at selected solver indices; no interpolation from old 1024 support",
            "effective_operator_cv": "rigno.heat3d_v6_p1i_anchor_query.conservative_selected_control_volume; same-layer nearest partition of all solver CV",
            "anchor_context_and_scale": "always exact original 1024 anchors with frozen train-only normalization",
            "temperature_label_used": False,
        },
        "graph_cache_contract": {
            "backend": "sparse_kdtree_v1",
            "graph_config_and_seed": "from each frozen checkpoint run_config",
            "cache_key": ["ordered_support_hash", "resolved_graph_config_hash", "graph_seed", "graph_builder_code_sha256"],
            "cached_uncached_real_edge_hash_must_match": True,
        },
        "reconstruction_contract": {
            "method": "layer/interface-aware inverse-distance reconstruction with per-domain same-layer fallback",
            "cache_key": ["full_coords_hash", "ordered_support_hash", "layer_hash", "interface_definition_hash", "reconstruction_code_sha256"],
            "cached_uncached_map_and_prediction_must_match": True,
        },
        "numeric_tolerances": {
            "R0_adapter_vs_reference_prediction_max_abs_K": 0.0,
            "R0_adapter_vs_reference_scale_max_abs": 0.0,
            "R0_adapter_vs_reference_full_field_max_abs_K": 0.0,
            "anchor_to_solver_coordinate_max_distance_m": 1e-14,
            "operator_volume_relative_error": 1e-12,
            "full_kq_power_relative_error": 1e-12,
            "cached_uncached_prediction_max_abs_K": 1e-6,
            "all_outputs_finite": True,
        },
        "development_subset": {
            "role": "valid_iid", "count": 32,
            "selection_rule": "ascending SHA256(sample_id), first 32",
            "selection_seed": None,
            "sample_ids": development_ids,
            "model_error_or_temperature_used": False,
        },
        "timing_contract": {
            "R0_validation_cost": "duplicate reference+adapter graphs/forwards plus hashes and 240825 reconstruction audits; validation only",
            "production_timing": "future single workflow wall clock with graph/cache and reconstruction boundaries explicitly separated",
            "R0_may_be_reported_as_production_timing": False,
        },
        "execution_contract": {
            "high_n_inference_executed_this_closeout": False,
            "training_executed": False, "test_accessed": False, "sealed_accessed": False,
        },
    }
    write_json(args.output_binding_json, binding)
    args.output_binding_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_binding_md.write_text(binding_report(binding), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
