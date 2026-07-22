#!/usr/bin/env python3
"""Read-only P1d deconfounding audit used to decide the P1e dataset policy."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin1024.yaml"
SAMPLES = ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin1024_samples.csv"
SOURCES = ROOT / "configs/heat3d_v6/v6_p1d_asymmetric_dual_robin1024_sources.csv"
OUTPUT_JSON = ROOT / "configs/heat3d_v6/v6_p1e_p1d_baseline_deconfounding_audit.json"
OUTPUT_MD = ROOT / "docs/v6_p1e_p1d_baseline_deconfounding_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def effective_rank(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    scale = np.std(matrix, axis=0, keepdims=True)
    matrix = matrix / np.where(scale > 0, scale, 1.0)
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular**2
    probability = energy / np.sum(energy)
    entropy_rank = float(np.exp(-np.sum(probability[probability > 0] * np.log(probability[probability > 0]))))
    return {
        "algebraic_rank": int(np.linalg.matrix_rank(matrix)),
        "effective_rank": entropy_rank,
        "singular_values": singular.tolist(),
    }


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = read_csv(SAMPLES)
    source_rows = read_csv(SOURCES)
    if len(rows) != 1024 or len(source_rows) != 8192:
        raise AssertionError("P1d tracked result count changed")
    names = (
        "top_h_W_m2K", "bottom_h_W_m2K", "package_total_power_W",
        "total_source_area_mm2", "layout_seed", "peak_deltaT_K",
    )
    values = {name: np.asarray([float(row[name]) for row in rows]) for name in names}

    correlations: dict[str, Any] = {}
    for boundary in ("top_h_W_m2K", "bottom_h_W_m2K"):
        correlations[f"{boundary}__package_total_power_W"] = {
            "pearson": float(np.corrcoef(values[boundary], values["package_total_power_W"])[0, 1]),
            "spearman": float(spearmanr(values[boundary], values["package_total_power_W"]).statistic),
        }
    correlations["top_h_W_m2K__bottom_h_W_m2K"] = {
        "pearson": float(np.corrcoef(values["top_h_W_m2K"], values["bottom_h_W_m2K"])[0, 1]),
        "spearman": float(spearmanr(values["top_h_W_m2K"], values["bottom_h_W_m2K"]).statistic),
    }

    raw_predictors = np.column_stack([
        values["top_h_W_m2K"], values["bottom_h_W_m2K"],
        values["total_source_area_mm2"],
    ])
    standardized = (raw_predictors - np.mean(raw_predictors, axis=0)) / np.std(raw_predictors, axis=0)
    predictors = np.column_stack([np.ones(len(rows)), standardized])
    target = values["package_total_power_W"]
    coefficients = np.linalg.lstsq(predictors, target, rcond=None)[0]
    # Elementwise reduction avoids a spurious Accelerate/BLAS matmul overflow
    # warning observed for this small, finite, well-conditioned design matrix.
    prediction = np.sum(predictors * coefficients[None, :], axis=1)
    power_r2 = float(1.0 - np.sum((target - prediction) ** 2) / np.sum((target - np.mean(target)) ** 2))

    per_family_power: defaultdict[str, set[float]] = defaultdict(set)
    per_bc_power: defaultdict[str, set[float]] = defaultdict(set)
    for row in rows:
        power = float(row["package_total_power_W"])
        per_family_power[row["family_id"]].add(power)
        key = f"top={float(row['top_h_W_m2K']):g}|bottom={float(row['bottom_h_W_m2K']):g}"
        per_bc_power[key].add(power)
    source_counts = Counter(row["sample_id"] for row in source_rows)
    source_areas = np.asarray([float(row["declared_source_area_m2"]) for row in source_rows])
    source_powers = np.asarray([float(row["source_power_W"]) for row in source_rows])

    design_matrix = np.column_stack([
        values["top_h_W_m2K"], values["bottom_h_W_m2K"],
        values["package_total_power_W"], values["total_source_area_mm2"], values["layout_seed"],
    ])
    audit = {
        "schema_version": "heat3d_v6_p1e_p1d_baseline_deconfounding_audit_v1",
        "mode": "read_only_tracked_artifact_audit",
        "inputs": {
            "config": str(CONFIG.relative_to(ROOT)), "config_sha256": sha256(CONFIG),
            "samples_csv": str(SAMPLES.relative_to(ROOT)), "samples_csv_sha256": sha256(SAMPLES),
            "sources_csv": str(SOURCES.relative_to(ROOT)), "sources_csv_sha256": sha256(SOURCES),
        },
        "sample_count": len(rows),
        "correlations": correlations,
        "power_from_top_bottom_area_linear_R2": power_r2,
        "numeric_factor_effective_rank": effective_rank(design_matrix),
        "per_family_power_levels_W": {key: sorted(value) for key, value in sorted(per_family_power.items())},
        "per_BC_pair_power_levels_W": {key: sorted(value) for key, value in sorted(per_bc_power.items())},
        "source_design": {
            "source_count_distribution": dict(sorted(Counter(source_counts.values()).items())),
            "all_samples_have_eight_sources": set(source_counts.values()) == {8},
            "nominal_equal_area_rule_in_config": config["source_contract"]["power_allocation"] == "equal_area_equal_power",
            "realized_declared_area_equal_within_every_sample": all(
                np.allclose(
                    [float(row["declared_source_area_m2"]) for row in source_rows if row["sample_id"] == sample_id],
                    float(next(row["declared_source_area_m2"] for row in source_rows if row["sample_id"] == sample_id)),
                    rtol=0.0, atol=1e-18,
                ) for sample_id in source_counts
            ),
            "equal_power_within_every_sample": all(
                len({row["source_power_W"] for row in source_rows if row["sample_id"] == sample_id}) == 1
                for sample_id in source_counts
            ),
            "source_area_m2": {"min": float(np.min(source_areas)), "max": float(np.max(source_areas))},
            "source_power_W": {"min": float(np.min(source_powers)), "max": float(np.max(source_powers))},
        },
        "qualification_gaps": {
            "fixed_power_geometry_BC_pairs_available": False,
            "independent_top_sensitivity_identifiable": False,
            "independent_bottom_sensitivity_identifiable": False,
            "bottom_BC_learnability_qualifiable": False,
            "group_locked_split_map_present": False,
            "layout_BC_source_count_power_density_OOD_present": False,
        },
        "decision": {
            "selected_policy": "rebuild_new_p1e1024_keep_p1d_as_provenance",
            "p1d_formal_training_allowed": False,
            "reason": (
                "P1d ties family-specific powers to BC, has no fixed-P/fixed-geometry BC pairs, "
                "and fixes every sample to eight equal-power sources under one nominal equal-area rule."
            ),
        },
        "guardrails": {"training_runs": 0, "model_inference_runs": 0, "data_generation_runs": 0},
        "config_declared_sample_count": int(config["sample_count"]),
    }
    OUTPUT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    top = correlations["top_h_W_m2K__package_total_power_W"]
    bottom = correlations["bottom_h_W_m2K__package_total_power_W"]
    OUTPUT_MD.write_text(f"""# V6-P1e P1d baseline deconfounding audit

This is a read-only audit of the frozen P1d tracked artifacts.  No solver,
generator, model training, or model inference was run.

## Finding

P1d is retained as provenance but is not qualified as the formal V6 training
dataset.  It must be replaced by a deconfounded P1e dataset.

- top-h versus power: Pearson `{top['pearson']:.6f}`, Spearman `{top['spearman']:.6f}`;
- bottom-h versus power: Pearson `{bottom['pearson']:.6f}`, Spearman `{bottom['spearman']:.6f}`;
- top/bottom/area linear prediction of power: R2 `{power_r2:.6f}`;
- all 1024 samples have exactly eight sources;
- all samples use the same nominal equal-area rule and exactly equal source
  powers; solver-grid realization can make declared source areas differ slightly;
- no fixed-power, fixed-geometry sweep independently varies top and bottom h;
- no group-locked train/IID/OOD split map exists.

Therefore the balanced temperature histogram in P1d does not remove BC-power
coupling.  P1e will use common power levels for every BC family, pre-solve
orthogonal pairing, variable source geometry, and group-locked split/OOD roles.
""", encoding="utf-8")
    print(json.dumps({
        "status": "ok", "sample_count": len(rows),
        "top_power_pearson": top["pearson"], "bottom_power_pearson": bottom["pearson"],
        "power_R2": power_r2, "decision": audit["decision"]["selected_policy"],
        "training_runs": 0, "model_inference_runs": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
