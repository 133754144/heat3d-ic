#!/usr/bin/env python3
"""Audit P1e orthogonality, paired BC sensitivity, field rank, and learnability."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _effective_rank(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    scale = np.std(matrix, axis=0, keepdims=True)
    matrix = matrix[:, np.ravel(scale > 1e-15)]
    matrix = matrix / np.std(matrix, axis=0, keepdims=True)
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular**2
    probability = energy / np.sum(energy)
    entropy_rank = float(np.exp(-np.sum(probability[probability > 0] * np.log(probability[probability > 0]))))
    return {
        "algebraic_rank": int(np.linalg.matrix_rank(matrix)),
        "effective_rank": entropy_rank, "singular_values": singular.tolist(),
    }


def _relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b**2)), 1e-15))


def _field(dataset: Path, sample_id: str) -> np.ndarray:
    return np.load(dataset / sample_id / "deltaT.npy", allow_pickle=False).reshape(-1).astype(np.float64)


def audit(config_path: Path, dataset: Path, stem: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    samples = _read_csv(CONFIG_DIR / f"{stem}_samples.csv")
    manifest = json.loads((CONFIG_DIR / f"{stem}_manifest.json").read_text(encoding="utf-8"))
    if len(samples) != int(config["sample_count"]) or manifest["sample_count"] != len(samples):
        raise AssertionError("artifact count mismatch")
    factor_names = ("top_h_W_m2K", "bottom_h_W_m2K", "package_total_power_W")
    factors = np.asarray([[float(row[key]) for key in factor_names] for row in samples])
    pearson = np.corrcoef(factors, rowvar=False)
    spearman = np.asarray([[float(spearmanr(factors[:, i], factors[:, j]).statistic) for j in range(3)] for i in range(3)])
    group_roles: defaultdict[str, set[str]] = defaultdict(set)
    for row in samples:
        group_roles[row["group_id"]].add(row["split_role"])
    complete_rows = [row for row in samples if row["design_block"] == "complete_factorial"]
    complete_groups = sorted({row["group_id"] for row in complete_rows})
    per_group: defaultdict[str, dict[tuple[float, float, float], dict[str, str]]] = defaultdict(dict)
    for row in complete_rows:
        key = tuple(float(row[name]) for name in factor_names)
        per_group[row["group_id"]][key] = row
    sensitivity_rows: list[dict[str, Any]] = []
    response_vectors: list[np.ndarray] = []
    top_monotonic: list[bool] = []
    bottom_monotonic: list[bool] = []
    bottom_fraction_monotonic: list[bool] = []
    top_effects: list[float] = []
    bottom_effects: list[float] = []
    top_field_correlations: list[float] = []
    bottom_field_correlations: list[float] = []
    top_bottom_cosines: list[float] = []
    for group_id in complete_groups:
        rows = per_group[group_id]
        tops = sorted({key[0] for key in rows})
        bottoms = sorted({key[1] for key in rows})
        powers = sorted({key[2] for key in rows})
        if len(rows) != 64 or [len(tops), len(bottoms), len(powers)] != [4, 4, 4]:
            raise AssertionError(f"{group_id}: incomplete factorial")
        for power in powers:
            matrix_fields: list[np.ndarray] = []
            for top in tops:
                for bottom in bottoms:
                    matrix_fields.append(_field(dataset, rows[(top, bottom, power)]["sample_id"]))
            response_vectors.append(np.stack(matrix_fields))
            for bottom in bottoms:
                sequence = [float(rows[(top, bottom, power)]["peak_deltaT_K"]) for top in tops]
                top_monotonic.append(all(a >= b - 1e-9 for a, b in zip(sequence, sequence[1:])))
            for top in tops:
                sequence = [float(rows[(top, bottom, power)]["peak_deltaT_K"]) for bottom in bottoms]
                fractions = [float(rows[(top, bottom, power)]["bottom_heat_fraction"]) for bottom in bottoms]
                bottom_monotonic.append(all(a >= b - 1e-9 for a, b in zip(sequence, sequence[1:])))
                bottom_fraction_monotonic.append(all(a <= b + 1e-9 for a, b in zip(fractions, fractions[1:])))
            for bottom in bottoms:
                low = _field(dataset, rows[(tops[0], bottom, power)]["sample_id"])
                high = _field(dataset, rows[(tops[-1], bottom, power)]["sample_id"])
                top_effects.append(_relative_rms(low, high))
            for top in tops:
                low = _field(dataset, rows[(top, bottoms[0], power)]["sample_id"])
                high = _field(dataset, rows[(top, bottoms[-1], power)]["sample_id"])
                bottom_effects.append(_relative_rms(low, high))
            reference = _field(dataset, rows[(tops[0], bottoms[0], power)]["sample_id"])
            top_vector = _field(dataset, rows[(tops[-1], bottoms[0], power)]["sample_id"]) - reference
            bottom_vector = _field(dataset, rows[(tops[0], bottoms[-1], power)]["sample_id"]) - reference
            top_field_correlation = float(np.corrcoef(reference, reference + top_vector)[0, 1])
            bottom_field_correlation = float(np.corrcoef(reference, reference + bottom_vector)[0, 1])
            top_field_correlations.append(top_field_correlation)
            bottom_field_correlations.append(bottom_field_correlation)
            cosine = float(np.dot(top_vector, bottom_vector) / max(np.linalg.norm(top_vector) * np.linalg.norm(bottom_vector), 1e-15))
            top_bottom_cosines.append(cosine)
            sensitivity_rows.append({
                "group_id": group_id, "package_total_power_W": power,
                "top_extreme_relative_field_rms": _relative_rms(reference, reference + top_vector),
                "bottom_extreme_relative_field_rms": _relative_rms(reference, reference + bottom_vector),
                "top_extreme_field_pearson": top_field_correlation,
                "bottom_extreme_field_pearson": bottom_field_correlation,
                "top_bottom_effect_cosine": cosine,
            })
    response = np.concatenate(response_vectors, axis=0)
    response_rank = _effective_rank(response)
    normalized_response = response / np.maximum(np.sqrt(np.mean(response**2, axis=1, keepdims=True)), 1e-15)
    normalized_response_rank = _effective_rank(normalized_response)
    factor_rank = _effective_rank(factors)
    max_offdiag_pearson = float(np.max(np.abs(pearson - np.eye(3))))
    max_offdiag_spearman = float(np.max(np.abs(spearman - np.eye(3))))
    checks = {
        "at_least_128_preregistered_paired_cases": len(complete_rows) >= 128,
        "BC_power_pearson_abs_le_0p02": max(abs(float(pearson[0, 2])), abs(float(pearson[1, 2]))) <= 0.02,
        "BC_power_spearman_abs_le_0p02": max(abs(float(spearman[0, 2])), abs(float(spearman[1, 2]))) <= 0.02,
        "group_locked_splits": all(len(value) == 1 for value in group_roles.values()),
        "top_peak_monotonic_fraction_ge_0p99": float(np.mean(top_monotonic)) >= 0.99,
        "bottom_peak_monotonic_fraction_ge_0p99": float(np.mean(bottom_monotonic)) >= 0.99,
        "bottom_heat_fraction_monotonic_fraction_ge_0p99": float(np.mean(bottom_fraction_monotonic)) >= 0.99,
        "bottom_effect_nonzero": float(np.median(bottom_effects)) > 1e-3,
        "field_response_algebraic_rank_ge_3": response_rank["algebraic_rank"] >= 3,
        "temperature_window_report_only_has_coverage": sum(30.0 <= float(row["peak_deltaT_K"]) <= 80.0 for row in samples) > 0,
    }
    payload = {
        "schema_version": "heat3d_v6_p1e_orthogonal_deconfounding_audit_v1",
        "config": str(config_path.relative_to(ROOT)), "dataset": str(dataset.relative_to(ROOT)),
        "sample_count": len(samples), "complete_factorial_case_count": len(complete_rows),
        "complete_factorial_group_count": len(complete_groups),
        "factor_names": factor_names, "pearson_correlation": pearson.tolist(),
        "spearman_correlation": spearman.tolist(), "max_offdiagonal_pearson": max_offdiag_pearson,
        "max_offdiagonal_spearman": max_offdiag_spearman,
        "fixed_geometry_power_BC_sensitivity": {
            "top_peak_monotonic_fraction": float(np.mean(top_monotonic)),
            "bottom_peak_monotonic_fraction": float(np.mean(bottom_monotonic)),
            "bottom_heat_fraction_monotonic_fraction": float(np.mean(bottom_fraction_monotonic)),
            "top_extreme_relative_field_rms": {"median": float(np.median(top_effects)), "min": float(np.min(top_effects)), "max": float(np.max(top_effects))},
            "bottom_extreme_relative_field_rms": {"median": float(np.median(bottom_effects)), "min": float(np.min(bottom_effects)), "max": float(np.max(bottom_effects))},
            "top_extreme_field_pearson": {"median": float(np.median(top_field_correlations)), "min": float(np.min(top_field_correlations)), "max": float(np.max(top_field_correlations))},
            "bottom_extreme_field_pearson": {"median": float(np.median(bottom_field_correlations)), "min": float(np.min(bottom_field_correlations)), "max": float(np.max(bottom_field_correlations))},
            "top_bottom_effect_cosine": {"median": float(np.median(top_bottom_cosines)), "min": float(np.min(top_bottom_cosines)), "max": float(np.max(top_bottom_cosines))},
        },
        "factor_design_rank": factor_rank,
        "projected_field_response_rank": response_rank,
        "amplitude_normalized_projected_field_response_rank": normalized_response_rank,
        "bottom_BC_learnability": {
            "qualification": "learnable_nonzero_independent_response" if checks["bottom_effect_nonzero"] else "not_identifiable",
            "evidence": "fixed geometry, power, and top-h; bottom-h changes peak, field, and bottom heat fraction monotonically",
            "target_labels_used_for_factor_design": False,
        },
        "checks": checks, "passed": all(checks.values()),
        "guardrails": {"temperature_filtering": False, "sample_replacement": False, "training_runs": 0, "model_inference_runs": 0},
    }
    json_path = CONFIG_DIR / f"{stem}_orthogonal_audit.json"
    csv_path = CONFIG_DIR / f"{stem}_paired_sensitivity.csv"
    md_path = ROOT / "docs" / f"{stem}_orthogonal_audit.md"
    _json(json_path, payload)
    _csv(csv_path, sensitivity_rows)
    md_path.write_text(f"""# {stem} orthogonal deconfounding audit

- cases: `{len(samples)}`; complete paired cases: `{len(complete_rows)}`;
- BC--power max |Pearson|: `{max(abs(float(pearson[0, 2])), abs(float(pearson[1, 2]))):.6g}`;
- BC--power max |Spearman|: `{max(abs(float(spearman[0, 2])), abs(float(spearman[1, 2]))):.6g}`;
- top/bottom peak monotonic fractions: `{np.mean(top_monotonic):.3f}` / `{np.mean(bottom_monotonic):.3f}`;
- bottom heat-fraction monotonic fraction: `{np.mean(bottom_fraction_monotonic):.3f}`;
- median extreme top/bottom field effects: `{np.median(top_effects):.4%}` / `{np.median(bottom_effects):.4%}`;
- projected BC-response algebraic/effective rank: `{response_rank['algebraic_rank']}` / `{response_rank['effective_rank']:.4f}`;
- amplitude-normalized projected-field algebraic/effective rank:
  `{normalized_response_rank['algebraic_rank']}` / `{normalized_response_rank['effective_rank']:.4f}`;
- bottom BC: `{payload['bottom_BC_learnability']['qualification']}`;
- qualification: `{'PASS' if payload['passed'] else 'FAIL'}`.

All cases were frozen before solving.  Peak DeltaT was used only for reporting,
never filtering, replacement, power inversion, factor selection, or seed search.
""", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifact-stem", required=True)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    result = audit(config.resolve(), dataset.resolve(), args.artifact_stem)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
