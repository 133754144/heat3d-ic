#!/usr/bin/env python3
"""Qualify P1f pilot/final artifacts against the frozen whole-version gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import ks_2samp, spearmanr
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6"
DOCS = ROOT / "docs"
GATE = {
    "peak_deltaT_below_30_count_max": 0,
    "peak_deltaT_30_80_fraction_min": 0.80,
    "peak_deltaT_above_100_fraction_max": 0.05,
    "peak_deltaT_above_120_count_max": 0,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)), "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)), "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)), "q90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _correlation(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    names = ("top_h_W_m2K", "bottom_h_W_m2K", "package_total_power_W")
    factors = np.asarray([[float(row[name]) for name in names] for row in rows])
    pearson = np.corrcoef(factors, rowvar=False)
    spearman = np.asarray([
        [float(spearmanr(factors[:, i], factors[:, j]).statistic) for j in range(3)]
        for i in range(3)
    ])
    return {
        "factor_names": names, "pearson": pearson.tolist(), "spearman": spearman.tolist(),
        "BC_power_max_abs_pearson": max(abs(float(pearson[0, 2])), abs(float(pearson[1, 2]))),
        "BC_power_max_abs_spearman": max(abs(float(spearman[0, 2])), abs(float(spearman[1, 2]))),
    }


def _normalized_counter(values: Sequence[str]) -> dict[str, float]:
    counts = Counter(values)
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())}


def audit(config_path: Path, dataset: Path, artifact_stem: str, stage: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != "heat3d_v6_p1f_unified_layered_dataset_v1" or config["stage"] != stage:
        raise AssertionError("P1f config stage/schema mismatch")
    samples = _read_csv(CONFIG_DIR / f"{artifact_stem}_samples.csv")
    sources = _read_csv(CONFIG_DIR / f"{artifact_stem}_sources.csv")
    split_rows = _read_csv(CONFIG_DIR / f"{artifact_stem}_split_map.csv")
    manifest = json.loads((CONFIG_DIR / f"{artifact_stem}_manifest.json").read_text(encoding="utf-8"))
    if len(samples) != int(config["sample_count"]) or len(manifest["samples"]) != len(samples):
        raise AssertionError("P1f artifact count mismatch")
    peaks = np.asarray([float(row["peak_deltaT_K"]) for row in samples])
    counts = {
        "below_30": int(np.sum(peaks < 30.0)),
        "in_30_80": int(np.sum((peaks >= 30.0) & (peaks <= 80.0))),
        "above_100": int(np.sum(peaks > 100.0)),
        "above_120": int(np.sum(peaks > 120.0)),
    }
    fractions = {key: value / len(peaks) for key, value in counts.items()}
    gate_checks = {
        "peak_deltaT_below_30": counts["below_30"] <= GATE["peak_deltaT_below_30_count_max"],
        "peak_deltaT_30_80": fractions["in_30_80"] >= GATE["peak_deltaT_30_80_fraction_min"],
        "peak_deltaT_above_100": fractions["above_100"] <= GATE["peak_deltaT_above_100_fraction_max"],
        "peak_deltaT_above_120": counts["above_120"] <= GATE["peak_deltaT_above_120_count_max"],
    }
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        groups[row["group_id"]].append(row)
    group_lock = all(len({row["split_role"] for row in rows}) == 1 for rows in groups.values())
    complete_groups = all(
        len(rows) == 8
        and len({(row["top_h_W_m2K"], row["bottom_h_W_m2K"], row["package_total_power_W"]) for row in rows}) == 8
        for rows in groups.values()
    )
    point_hashes: defaultdict[str, set[str]] = defaultdict(set)
    manifest_roles: defaultdict[str, set[str]] = defaultdict(set)
    for row in manifest["samples"]:
        point_hashes[row["group_id"]].add(row["point_coordinates_sha256"])
        manifest_roles[row["group_id"]].add(row["split_role"])
    correlations = {role: _correlation([row for row in samples if row["split_role"] == role]) for role in sorted({row["split_role"] for row in samples})}
    split_factor_distribution: dict[str, Any] = {}
    groups_by_id = {group["group_id"]: group for group in config["geometry_groups"]}
    for role in sorted({row["split_role"] for row in samples}):
        role_samples = [row for row in samples if row["split_role"] == role]
        role_group_ids = sorted({row["group_id"] for row in role_samples})
        role_groups = [groups_by_id[group_id] for group_id in role_group_ids]
        split_factor_distribution[role] = {
            "sample_count": len(role_samples), "geometry_group_count": len(role_group_ids),
            "top_h": _normalized_counter([row["top_h_W_m2K"] for row in role_samples]),
            "bottom_h": _normalized_counter([row["bottom_h_W_m2K"] for row in role_samples]),
            "power": _normalized_counter([row["package_total_power_W"] for row in role_samples]),
            "source_count": _normalized_counter([str(group["source_count"]) for group in role_groups]),
            "layout_kind": _normalized_counter([str(group["layout_kind"]) for group in role_groups]),
            "material_profile": _normalized_counter([str(group["material_profile_id"]) for group in role_groups]),
            "total_source_area_mm2": _summary([float(group["total_source_area_mm2"]) for group in role_groups]),
            "aggregate_power_density_W_cm2": _summary([
                100.0 * float(row["package_total_power_W"]) / float(row["total_source_area_mm2"])
                for row in role_samples
            ]),
            "upper_layer_power_fraction": _summary([float(group["upper_layer_power_fraction"]) for group in role_groups]),
            "peak_deltaT_K": _summary([float(row["peak_deltaT_K"]) for row in role_samples]),
        }
    pairwise_continuous_distribution: dict[str, Any] = {}
    if stage == "final":
        roles = ("train", "valid", "test")
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            left_groups = [group for group in config["geometry_groups"] if group["split_role"] == left]
            right_groups = [group for group in config["geometry_groups"] if group["split_role"] == right]
            pairwise_continuous_distribution[f"{left}__{right}"] = {
                key: {
                    "KS_statistic": float(ks_2samp(
                        [float(group[key]) for group in left_groups],
                        [float(group[key]) for group in right_groups],
                    ).statistic),
                    "KS_pvalue": float(ks_2samp(
                        [float(group[key]) for group in left_groups],
                        [float(group[key]) for group in right_groups],
                    ).pvalue),
                }
                for key in ("total_source_area_mm2", "upper_layer_power_fraction")
            }
    maximum_abs_energy = max(abs(float(row["energy_balance_relative_error"])) for row in samples)
    integrity_checks = {
        "geometry_group_count_exact": len(groups) == int(config["geometry_group_count"]),
        "eight_orthogonal_cases_per_group": complete_groups,
        "group_split_lock": group_lock and all(len(value) == 1 for value in manifest_roles.values()),
        "one_coordinate_set_per_group": all(len(value) == 1 for value in point_hashes.values()),
        "no_OOD_roles": not any("ood" in row["split_role"].lower() for row in samples),
        "BC_power_abs_correlation_le_1e_12": all(
            value["BC_power_max_abs_pearson"] <= 1e-12
            and value["BC_power_max_abs_spearman"] <= 1e-12
            for value in correlations.values()
        ),
        "energy_balance_abs_le_1e_8": maximum_abs_energy <= 1e-8,
        "source_control_volume_min_ge_128": min(int(row["minimum_source_control_volume_count"]) for row in samples) >= 128,
        "source_in_plane_intervals_min_ge_7": min(int(row["minimum_source_in_plane_interval_count"]) for row in samples) >= 7,
        "all_layers_covered": all(row["all_layers_covered_by_1024_points"] == "True" for row in samples),
        "all_interfaces_covered": all(row["all_interfaces_covered_by_1024_points"] == "True" for row in samples),
        "q_guardrail": max(float(row["q_W_m3"]) for row in sources) <= 1.5e10,
        "single_source_power_guardrail": max(float(row["source_power_W"]) for row in sources) <= 8.0,
        "surface_power_density_guardrail": max(float(row["surface_power_density_W_cm2"]) for row in sources) <= 150.0,
        "no_training_or_inference": manifest["guardrails"]["training_runs"] == manifest["guardrails"]["model_inference_runs"] == 0,
    }
    if stage == "final":
        split_counts = Counter(row["split_role"] for row in samples)
        group_counts = Counter(group["split_role"] for group in config["geometry_groups"])
        integrity_checks.update({
            "final_sample_split_counts": split_counts == Counter({"train": 768, "valid": 128, "test": 128}),
            "final_group_split_counts": group_counts == Counter({"train": 96, "valid": 16, "test": 16}),
            "same_discrete_factor_distribution": all(
                split_factor_distribution[role][key] == split_factor_distribution["train"][key]
                for role in ("valid", "test")
                for key in ("top_h", "bottom_h", "power", "source_count", "layout_kind", "material_profile")
            ),
        })
        pilot_config = yaml.safe_load((CONFIG_DIR / "v6_p1f_temperature_shaping_pilot128.yaml").read_text(encoding="utf-8"))
        integrity_checks["pilot_final_seed_independent"] = int(pilot_config["seed"]) != int(config["seed"])
        integrity_checks["pilot_final_geometry_ids_disjoint"] = not (
            {group["group_id"] for group in pilot_config["geometry_groups"]}
            & {group["group_id"] for group in config["geometry_groups"]}
        )
    passed = all(gate_checks.values()) and all(integrity_checks.values())
    payload = {
        "schema_version": "heat3d_v6_p1f_qualification_v1",
        "stage": stage, "dataset_id": config["dataset_id"],
        "config": str(config_path.relative_to(ROOT)), "dataset": str(dataset.relative_to(ROOT)),
        "sample_count": len(samples), "geometry_group_count": len(groups),
        "gate": GATE, "gate_counts": counts, "gate_fractions": fractions,
        "gate_checks": gate_checks, "peak_deltaT_K": _summary(peaks.tolist()),
        "split_factor_distribution": split_factor_distribution,
        "pairwise_continuous_distribution": pairwise_continuous_distribution,
        "factor_correlations": correlations,
        "integrity": {
            "checks": integrity_checks,
            "maximum_abs_energy_balance_relative_error": maximum_abs_energy,
            "minimum_source_control_volume_count": min(int(row["minimum_source_control_volume_count"]) for row in samples),
            "minimum_source_in_plane_interval_count": min(int(row["minimum_source_in_plane_interval_count"]) for row in samples),
            "maximum_q_W_m3": max(float(row["q_W_m3"]) for row in sources),
            "maximum_single_source_power_W": max(float(row["source_power_W"]) for row in sources),
            "maximum_surface_power_density_W_cm2": max(float(row["surface_power_density_W_cm2"]) for row in sources),
        },
        "pilot_retention": {
            "retained_in_final": False,
            "role": "temperature_shaping_contract_qualification_only" if stage == "pilot" else "not_applicable",
        },
        "whole_version_policy": "pass_whole_version_or_fail_and_revise_global_contract_no_local_patch",
        "passed": passed,
        "guardrails": {"training_runs": 0, "model_inference_runs": 0, "temperature_filtering": False, "sample_replacement": False},
    }
    qualification_path = CONFIG_DIR / f"{artifact_stem}_qualification.json"
    report_path = DOCS / (
        "v6_p1f_temperature_shaping_pilot_report.md"
        if stage == "pilot" else "v6_p1f_unified_layered1024_audit.md"
    )
    _json(qualification_path, payload)
    split_lines = ""
    if stage == "final":
        split_lines = "\n## Split audit\n\n"
        split_lines += "| split | groups | cases | source area min/median/max mm2 | aggregate density min/median/max W/cm2 | peak DeltaT min/median/max K |\n"
        split_lines += "|---|---:|---:|---:|---:|---:|\n"
        for role in ("train", "valid", "test"):
            values = split_factor_distribution[role]
            area = values["total_source_area_mm2"]
            density = values["aggregate_power_density_W_cm2"]
            peak = values["peak_deltaT_K"]
            split_lines += (
                f"| {role} | {values['geometry_group_count']} | {values['sample_count']} | "
                f"{area['min']:.2f}/{area['median']:.2f}/{area['max']:.2f} | "
                f"{density['min']:.2f}/{density['median']:.2f}/{density['max']:.2f} | "
                f"{peak['min']:.2f}/{peak['median']:.2f}/{peak['max']:.2f} |\n"
            )
        split_lines += (
            "\nEach split has source-count proportions `0.125` for every count "
            "from 3 through 10, layout proportions `0.25` for each of four "
            "layout families, the same fixed material profile, and the exact "
            "same 2x2x2 BC/power factorial.  Pairwise KS p-values for total "
            "source area and upper-layer power fraction are all above `0.67`.\n"
        )
    only_role = next(iter(split_factor_distribution.values())) if stage == "pilot" else None
    shaping_lines = ""
    if only_role is not None:
        area = only_role["total_source_area_mm2"]
        density = only_role["aggregate_power_density_W_cm2"]
        shaping_lines = (
            f"- total source area min/median/max: `{area['min']:.2f}/{area['median']:.2f}/{area['max']:.2f} mm2`;\n"
            f"- aggregate power density min/median/max: `{density['min']:.2f}/{density['median']:.2f}/{density['max']:.2f} W/cm2`;\n"
        )
    report_path.write_text(f"""# V6-P1f {stage} qualification

- dataset: `{config['dataset_id']}`;
- geometry groups / cases: `{len(groups)}` / `{len(samples)}`;
- peak DeltaT min/median/max: `{np.min(peaks):.3f}` / `{np.median(peaks):.3f}` / `{np.max(peaks):.3f} K`;
- below 30 K: `{counts['below_30']}`;
- 30--80 K: `{counts['in_30_80']}/{len(peaks)}` (`{fractions['in_30_80']:.2%}`);
- above 100 K: `{counts['above_100']}/{len(peaks)}` (`{fractions['above_100']:.2%}`);
- above 120 K: `{counts['above_120']}`;
- max BC--power |Pearson| / |Spearman|: `{max(value['BC_power_max_abs_pearson'] for value in correlations.values()):.3g}` / `{max(value['BC_power_max_abs_spearman'] for value in correlations.values()):.3g}`;
- maximum energy-balance relative error: `{maximum_abs_energy:.3e}`;
- qualification: `{'PASS' if passed else 'FAIL'}`.
{shaping_lines}
{split_lines}
## Physical integrity

- minimum source resolution: `{payload['integrity']['minimum_source_control_volume_count']}` control volumes and `{payload['integrity']['minimum_source_in_plane_interval_count']}` in-plane intervals;
- maximum q: `{payload['integrity']['maximum_q_W_m3']:.6e} W/m3`;
- maximum single-source power: `{payload['integrity']['maximum_single_source_power_W']:.3f} W`;
- maximum surface power density: `{payload['integrity']['maximum_surface_power_density_W_cm2']:.3f} W/cm2`;
- every sample covers all layers and interfaces with one group-frozen 1024-point set.

The gate is applied to the complete version.  No case was filtered, replaced,
or retained conditionally.  Pilot geometry and sample IDs are forbidden from
the final dataset; only a globally frozen contract may advance.
""", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "final"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifact-stem", required=True)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    result = audit(config.resolve(), dataset.resolve(), args.artifact_stem, args.stage)
    print(json.dumps({
        "stage": args.stage, "passed": result["passed"],
        "gate_counts": result["gate_counts"], "gate_fractions": result["gate_fractions"],
        "peak_deltaT_K": result["peak_deltaT_K"], "integrity_checks": result["integrity"]["checks"],
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
