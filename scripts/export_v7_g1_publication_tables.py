#!/usr/bin/env python3
"""Export publication tables from frozen V7 G1 JSON evidence.

This exporter is deliberately copy-only: it reads JSON summaries already
produced by the frozen analyses, never opens checkpoints/NPZ/HDF5/prediction
payloads, and performs no new metric calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


H2_PRIMARY_ROUTE = "U_v2_16384_reconstruction"
H2_ROBUSTNESS_ROUTE = "U_v2_direct240825"
H2_PRIMARY_METRIC = "source_region_RMSE_K"
H2_VARIANT_TO_CONTRAST = {
    "layout_agnostic_stratified_support": ("H2a", "Full vs generic support"),
    "cv_only_support": ("H2b", "Full vs CV-only support"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"publication exporter accepts JSON only: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _source(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}


def _find_one(rows: Sequence[Mapping[str, Any]], **criteria: Any) -> Mapping[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {criteria}, got {len(matches)}")
    return matches[0]


def _native_summary(
    rows: Sequence[Mapping[str, Any]], variant: str, metric: str
) -> dict[str, Any]:
    row = _find_one(rows, variant=variant, metric=metric, domain="registered_support_1024")
    return {
        "mean": row["mean"],
        "sample_sd": row["sample_sd"],
        "seed_values": row["seed_values"],
    }


def _h2_summary(
    rows: Sequence[Mapping[str, Any]], route_id: str, variant: str, metric: str
) -> dict[str, Any]:
    row = _find_one(rows, route_id=route_id, variant=variant, metric=metric)
    return {
        "mean": row["mean"],
        "sample_sd": row["sample_sd"],
        "seed_values": row["seed_values"],
    }


def _effect_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    distribution = row.get("paired_sample_distribution")
    paired_median = distribution["median"] if distribution is not None else row["paired_median"]
    paired_p90 = distribution["p90"] if distribution is not None else row["paired_p90"]
    paired_p95 = distribution["p95"] if distribution is not None else row["paired_p95"]
    paired_worst_10_mean = distribution["worst_10_mean"] if distribution is not None else row["paired_worst_10_mean"]
    bootstrap = row.get("bootstrap")
    ci_low = bootstrap["ci_low"] if bootstrap is not None else row["bootstrap_ci_low"]
    ci_high = bootstrap["ci_high"] if bootstrap is not None else row["bootstrap_ci_high"]
    claim_status = row.get("claim_status", row.get("claim_status_under_directional_rule"))
    return {
        "full_pooled_aggregate": row["full_pooled_aggregate"],
        "ablation_pooled_aggregate": row["ablation_pooled_aggregate"],
        "effect_ablation_minus_full": row["effect_ablation_minus_full"],
        "paired_median": paired_median,
        "paired_p90": paired_p90,
        "paired_p95": paired_p95,
        "paired_worst_10_mean": paired_worst_10_mean,
        "per_seed_effects": row["per_seed_effects"],
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "claim_status": claim_status,
    }


def _export_main_table(
    root: Path,
    completion: Mapping[str, Any],
    native_summary_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    specs = (
        ("H1", "Full vs Vanilla", "vanilla_RIGNO", "point_global_relative_rmse_pct"),
        ("H1b", "Full vs capacity-matched Vanilla", "vanilla_RIGNO_capacity_matched", "point_global_relative_rmse_pct"),
        ("H3", "Full vs no FiLM", "no_film", "sample_first_relative_rmse_pct"),
        ("H4", "Full vs physics-scale-only (learned residual scale correction removed)", "physics_scale_only", "raw_K_CV_RMSE_K"),
    )
    effect_rows = completion["hypothesis_effects"]
    rows = []
    for hypothesis, comparison, ablation_variant, metric in specs:
        effect = _find_one(effect_rows, hypothesis=hypothesis, primary_metric=metric)
        rows.append(
            {
                "hypothesis": hypothesis,
                "comparison": comparison,
                "domain": "registered_support_1024",
                "primary_metric": metric,
                "full_seed_summary": _native_summary(native_summary_rows, "Full", metric),
                "ablation_variant": ablation_variant,
                "ablation_seed_summary": _native_summary(native_summary_rows, ablation_variant, metric),
                **_effect_fields(effect),
            }
        )
    sources = [
        root / "docs/v7_g1_formal_completion_receipt.json",
        root / "research_artifacts/v7_g1_formal_archive/analysis_1024/variant_level_native_summary.json",
        root / "research_artifacts/v7_g1_formal_archive/analysis_1024/hypothesis_effect_table.json",
    ]
    payload = {
        "schema_version": "heat3d_v7_g1_publication_main_ablation_table_v1",
        "table_id": "G1_main_ablation",
        "copy_only_from_frozen_evidence": True,
        "scope": "frozen P1i valid_iid native 1024-point training evidence",
        "claim_boundary": "No test/OOD/external-superiority claim; H1/H1b primary remains point_global_relative_rmse_pct.",
        "source_artifacts": [_source(path, root) for path in sources],
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _export_h2_table(
    root: Path,
    h2_effect: Mapping[str, Any],
    h2_summary_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    effect_rows = [
        row for row in h2_effect["rows"]
        if row["route_id"] == H2_PRIMARY_ROUTE and row["metric"] == H2_PRIMARY_METRIC
    ]
    if len(effect_rows) != 2:
        raise ValueError(f"expected two primary H2 contrast rows, got {len(effect_rows)}")
    rows = []
    for effect in effect_rows:
        contrast_id, contrast_label = H2_VARIANT_TO_CONTRAST[effect["ablation_variant"]]
        rows.append(
            {
                "hypothesis_group": "H2",
                "contrast_id": contrast_id,
                "contrast_label": contrast_label,
                "route_id": H2_PRIMARY_ROUTE,
                "route_role": "primary",
                "domain": "heat3d_v6_p1i_full_field_240825",
                "primary_metric": H2_PRIMARY_METRIC,
                "full_seed_summary": _h2_summary(h2_summary_rows, H2_PRIMARY_ROUTE, "Full", H2_PRIMARY_METRIC),
                "ablation_variant": effect["ablation_variant"],
                "ablation_seed_summary": _h2_summary(
                    h2_summary_rows, H2_PRIMARY_ROUTE, effect["ablation_variant"], H2_PRIMARY_METRIC
                ),
                **_effect_fields(effect),
            }
        )
    sources = [
        root / "research_artifacts/v7_g1_formal_archive/h2_fullfield_240825_native/h2_hypothesis_effect_table.json",
        root / "research_artifacts/v7_g1_formal_archive/h2_fullfield_240825_native/h2_variant_route_summary.json",
        root / "research_artifacts/v7_g1_formal_archive/h2_fullfield_240825_native/h2_bootstrap_ci_receipt.json",
    ]
    payload = {
        "schema_version": "heat3d_v7_g1_publication_h2_common_domain_table_v1",
        "table_id": "H2_common_domain_attribution",
        "copy_only_from_frozen_evidence": True,
        "hypothesis_structure": "H2 is one hypothesis group with two preregistered contrasts: H2a and H2b.",
        "route_id": H2_PRIMARY_ROUTE,
        "route_role": "primary",
        "domain": "heat3d_v6_p1i_full_field_240825",
        "primary_metric": H2_PRIMARY_METRIC,
        "bootstrap": {
            "replicates": 10000,
            "random_seed": 20260829,
            "resampling": ["seed_with_replacement", "valid_iid_sample_within_seed_with_replacement"],
            "interval": "percentile_95_percent_CI",
        },
        "source_artifacts": [_source(path, root) for path in sources],
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _export_route_table(root: Path, route_payload: Mapping[str, Any], output: Path) -> None:
    rows = [
        {
            "variant": row["variant"],
            "metric": row["metric"],
            "direct_route_id": H2_ROBUSTNESS_ROUTE,
            "primary_route_id": H2_PRIMARY_ROUTE,
            "direct_seed_values": row["direct_seed_values"],
            "primary_seed_values": row["reconstruction_seed_values"],
            "route_difference_definition": route_payload["definition"],
            "route_difference_mean": row["route_difference_mean"],
            "route_difference_sample_sd": row["route_difference_sample_sd"],
            "route_difference_seed_values": row["route_difference_seed_values"],
        }
        for row in route_payload["rows"]
    ]
    if len(rows) != 9:
        raise ValueError(f"expected 9 H2 route robustness rows, got {len(rows)}")
    source_path = root / "research_artifacts/v7_g1_formal_archive/h2_fullfield_240825_native/h2_route_comparison.json"
    payload = {
        "schema_version": "heat3d_v7_g1_publication_h2_route_robustness_table_v1",
        "table_id": "H2_U_route_robustness",
        "copy_only_from_frozen_evidence": True,
        "scope": "same frozen 240825 common-domain H2 predictions; route sensitivity only",
        "primary_route_id": H2_PRIMARY_ROUTE,
        "robustness_route_id": H2_ROBUSTNESS_ROUTE,
        "source_artifacts": [_source(source_path, root)],
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _export_parameter_table(root: Path, output: Path) -> None:
    source_path = root / "docs/v7_g1_training_results.md"
    payload = {
        "schema_version": "heat3d_v7_g1_publication_parameter_control_table_v1",
        "table_id": "G1_parameter_count_and_control",
        "copy_only_from_frozen_evidence": True,
        "control": {
            "training_epochs": 200,
            "seeds": [0, 1, 2],
            "train_samples": 768,
            "valid_iid_samples": 128,
            "checkpoint_policy": "pre-registered valid_iid sample_first_relative_rmse_pct selected-best checkpoint",
            "model_or_data_changed_during_closeout": False,
        },
        "parameter_count_rows": [
            {"variant_group": "Full", "parameter_count": 892776},
            {"variant_group": "layout_agnostic_stratified_support", "parameter_count": 892776},
            {"variant_group": "cv_only_support", "parameter_count": 892776},
            {"variant_group": "no_film", "parameter_count": 878696},
            {"variant_group": "physics_scale_only", "parameter_count": 845158},
            {"variant_group": "vanilla_RIGNO", "parameter_count": 826277},
            {"variant_group": "vanilla_RIGNO_capacity_matched", "parameter_count": 895905},
        ],
        "source_artifacts": [_source(source_path, root)],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    completion_path = root / "docs/v7_g1_formal_completion_receipt.json"
    native_summary_path = root / "research_artifacts/v7_g1_formal_archive/analysis_1024/variant_level_native_summary.json"
    h2_effect_path = root / "research_artifacts/v7_g1_formal_archive/h2_fullfield_240825_native/h2_hypothesis_effect_table.json"
    h2_summary_path = root / "research_artifacts/v7_g1_formal_archive/h2_fullfield_240825_native/h2_variant_route_summary.json"
    h2_route_path = root / "research_artifacts/v7_g1_formal_archive/h2_fullfield_240825_native/h2_route_comparison.json"
    for path in (completion_path, native_summary_path, h2_effect_path, h2_summary_path, h2_route_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    completion = _load(completion_path)
    native_summary = _load(native_summary_path)
    h2_effect = _load(h2_effect_path)
    h2_summary = _load(h2_summary_path)
    h2_route = _load(h2_route_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _export_main_table(root, completion, native_summary["rows"], output_dir / "g1_main_ablation_table.json")
    _export_h2_table(root, h2_effect, h2_summary["rows"], output_dir / "h2_common_domain_attribution_table.json")
    _export_route_table(root, h2_route, output_dir / "h2_u_route_robustness_table.json")
    _export_parameter_table(root, output_dir / "parameter_count_control_table.json")
    print(json.dumps({"output_dir": str(output_dir), "tables": 4}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
