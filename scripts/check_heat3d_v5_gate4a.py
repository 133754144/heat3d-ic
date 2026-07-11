#!/usr/bin/env python3
"""Analytic fixture checks for V5 Gate 4A offline scale correction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import audit_heat3d_v5_gate4a as gate4  # noqa: E402


ROLE_COUNTS = {
    "train": 8,
    "valid_iid": 3,
    "test_iid": 2,
    "hard_train_holdout": 4,
    "hard_challenge_valid": 3,
    "hard_challenge_test": 2,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _coords() -> np.ndarray:
    return np.asarray(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 0.5, 1.0)],
        dtype=np.float64,
    )


def _metadata() -> dict[str, object]:
    return {
        "bc_feature_names": ["is_top", "is_bottom", "is_side", "is_interior"],
        "boundary_params": {
            "bottom": {"type": "dirichlet", "fixed_temperature_K": 300.0},
            "top": {"type": "robin", "h_W_m2K": 20.0, "ambient_temperature_K": 300.0},
            "side": {"type": "adiabatic"},
        },
        "boundary_regions": [
            {"name": "bottom", "point_indices": [0, 3, 6, 9]},
            {"name": "top", "point_indices": [2, 5, 8, 11]},
            {"name": "sides", "point_indices": [1, 4, 7, 10]},
        ],
    }


def _contract(role_counts: dict[str, int], gate1_hash: str, gate3_hash: str) -> dict[str, object]:
    return {
        "contract_id": "fixture-gate4a",
        "dataset_contract": {
            "dataset_id": "fixture_p5",
            "role_counts": role_counts,
            "total_sample_count": sum(role_counts.values()),
        },
        "target_and_reconstruction": {"physics_only": "delta_s_hat = 0", "shape_training": False},
        "input_feature_contract": {
            "global_physics_features": list(gate4.GLOBAL_FEATURES),
            "forbidden_input_categories": ["target", "oracle", "residual"],
        },
        "protocols": {
            "clean_only_zero_shot": {
                "fit_roles": ["train"],
                "selection_role": "valid_iid",
                "test_roles": ["test_iid", "hard_challenge_test"],
            },
            "hard_adapted": {
                "fit_roles": ["train", "hard_train_holdout"],
                "selection_role": "hard_challenge_valid",
                "test_roles": ["test_iid", "hard_challenge_test"],
            },
        },
        "frozen_predecessors": {
            "gate1_table": {"sha256": gate1_hash},
            "gate3_table": {"sha256": gate3_hash},
        },
    }


def _make_fixture(root: Path) -> dict[str, Path]:
    dataset = root / "dataset"
    dataset.mkdir()
    roles = [role for role, count in ROLE_COUNTS.items() for _ in range(count)]
    assignments: dict[str, str] = {}
    gate1_rows: list[dict[str, str]] = []
    gate3_rows: list[dict[str, str]] = []
    best_predictions: dict[str, np.ndarray] = {}
    final_predictions: dict[str, np.ndarray] = {}
    best_latents: dict[str, np.ndarray] = {}
    final_latents: dict[str, np.ndarray] = {}
    coords = _coords()
    z = coords[:, 2]
    x = coords[:, 0]
    y = coords[:, 1]
    volumes, _axes, _inverse, _shape = gate4.gate3._control_volumes(coords)
    base_pattern = 0.5 * z + 0.1 * x + 0.05 * y
    base_pattern[np.isclose(z, 0.0)] = 0.0
    base_pattern /= gate4.gate3._weighted_rms(base_pattern, volumes)
    for index, role in enumerate(roles):
        sample_id = f"sample_{index:04d}"
        assignments[sample_id] = role
        value = -0.45 + 0.06 * index
        s_phys = 0.45 + 0.015 * index
        s_true = s_phys * math.exp(value)
        sample = dataset / sample_id
        sample.mkdir()
        target = 300.0 + s_true * base_pattern
        q = (1.0 + 0.05 * index + x + 0.2 * y + z).reshape(-1, 1)
        q[np.isclose(z, 0.0)] = 0.0
        k = np.column_stack((3.0 + x + 0.1 * index, 4.0 + y, 2.0 + 0.3 * x + 0.1 * z))
        bc = np.zeros((coords.shape[0], 4), dtype=np.float64)
        bc[np.isclose(z, 1.0), 0] = 1.0
        bc[np.isclose(z, 0.0), 1] = 1.0
        bc[np.isclose(z, 0.5), 2] = 1.0
        np.save(sample / "coords.npy", coords)
        np.save(sample / "q_field.npy", q)
        np.save(sample / "k_field.npy", k)
        np.save(sample / "temperature.npy", target.reshape(-1, 1))
        np.save(sample / "bc_features.npy", bc)
        meta = _metadata()
        meta["p5_provenance"] = {"source_sample_id": f"origin_{index:04d}"}
        _write_json(sample / "sample_meta.json", meta)
        best_scale = s_true * (0.65 + 0.02 * (index % 3))
        final_scale = s_true * (0.7 + 0.015 * (index % 3))
        best_predictions[sample_id] = (300.0 + best_scale * base_pattern).reshape(-1, 1)
        final_predictions[sample_id] = (300.0 + final_scale * base_pattern).reshape(-1, 1)
        best_latents[sample_id] = np.asarray([index / 10.0, value, s_phys], dtype=np.float64)
        final_latents[sample_id] = np.asarray([index / 12.0, value * 0.8, s_phys * 1.1], dtype=np.float64)
        g1 = {
            "sample_id": sample_id,
            "role": role,
            "input_fingerprint": f"input_{index}",
            "full_fingerprint": f"full_{index}",
            "provenance_source_id": f"origin_{index:04d}",
            "P_operator_W": 1.0 + 0.1 * index,
            "raw_z_collapsed_1d_operator_K": s_phys,
            "harmonic_kx_W_mK": 3.0 + index * 0.1,
            "harmonic_ky_W_mK": 4.0 + index * 0.1,
            "harmonic_kz_W_mK": 2.0 + index * 0.05,
            "anisotropy_xy_over_z": 1.2 + index * 0.01,
            "Lx_m": 1.0,
            "Ly_m": 1.0,
            "Lz_m": 1.0,
            "top_area_m2": 1.0,
            "top_h_W_m2K": 20.0 + index,
            "T_bottom_K": 300.0,
            "T_inf_K": 300.0,
            "T_inf_minus_T_bottom_K": 0.0,
        }
        g3 = {
            "sample_id": sample_id,
            "role": role,
            "target_scale_cv_rms_K": s_true,
            "reference_temperature_K": 300.0,
            "q_weighted_local_kz_W_mK": 2.0 + 0.1 * index,
            "q_weighted_inverse_kz_mK_W": 0.5 - 0.005 * index,
            "q_low_k_overlap_fraction": 0.2 + 0.01 * (index % 4),
            "source_concentration": 1.1 + 0.02 * index,
            "source_z_centroid_normalized": 0.5,
            "source_layer_kz_heterogeneity_cv": 0.1 + 0.01 * (index % 3),
        }
        gate1_rows.append({key: format(value, ".17g") if isinstance(value, float) else str(value) for key, value in g1.items()})
        gate3_rows.append({key: format(value, ".17g") if isinstance(value, float) else str(value) for key, value in g3.items()})
    paths = {
        "dataset": dataset,
        "split": root / "split.json",
        "gate1": root / "gate1.csv",
        "gate3": root / "gate3.csv",
        "best_latents": root / "best_latents.npz",
        "final_latents": root / "final_latents.npz",
        "best_manifest": root / "best_latents_manifest.json",
        "final_manifest": root / "final_latents_manifest.json",
        "best_predictions": root / "best_predictions.npz",
        "final_predictions": root / "final_predictions.npz",
        "contract": root / "contract.json",
    }
    _write_json(paths["split"], {"dataset_id": "fixture_p5", "sample_splits": assignments})
    for path, values in ((paths["gate1"], gate1_rows), (paths["gate3"], gate3_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    np.savez_compressed(paths["best_latents"], **best_latents)
    np.savez_compressed(paths["final_latents"], **final_latents)
    np.savez_compressed(paths["best_predictions"], **best_predictions)
    np.savez_compressed(paths["final_predictions"], **final_predictions)
    for checkpoint, archive, manifest in (
        ("best", paths["best_latents"], paths["best_manifest"]),
        ("final", paths["final_latents"], paths["final_manifest"]),
    ):
        _write_json(
            manifest,
            {
                "checkpoint_sha256": f"fixture_{checkpoint}_checkpoint",
                "run_config_sha256": "fixture_run_config",
                "latent_archive_sha256": _sha256(archive),
                "sample_count": sum(ROLE_COUNTS.values()),
                "latent_dimension": 3,
                "max_prediction_abs_error_K": 0.0,
            },
        )
    _write_json(paths["contract"], _contract(ROLE_COUNTS, _sha256(paths["gate1"]), _sha256(paths["gate3"])))
    return paths


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heat3d_v5_gate4a_") as temp_name:
        root = Path(temp_name)
        paths = _make_fixture(root)
        table = root / "table.csv"
        models = root / "models.json"
        summary = root / "summary.json"
        closeout = root / "closeout.md"
        base = [
            sys.executable,
            "-B",
            "scripts/audit_heat3d_v5_gate4a.py",
            "--dataset", str(paths["dataset"]),
            "--split-map", str(paths["split"]),
            "--contract", str(paths["contract"]),
            "--gate1-table", str(paths["gate1"]),
            "--gate3-table", str(paths["gate3"]),
            "--best-latents", str(paths["best_latents"]),
            "--final-latents", str(paths["final_latents"]),
            "--best-latent-manifest", str(paths["best_manifest"]),
            "--final-latent-manifest", str(paths["final_manifest"]),
            "--best-predictions", str(paths["best_predictions"]),
            "--final-predictions", str(paths["final_predictions"]),
        ]
        dry = json.loads(_run(base + ["--dry-run"]).stdout)
        assert dry["planned_writes"] == []
        assert dry["dataset"]["sample_count"] == sum(ROLE_COUNTS.values())
        _run(base + [
            "--output-table", str(table),
            "--output-model-params", str(models),
            "--output-json", str(summary),
            "--output-md", str(closeout),
        ])
        check_summary = json.loads(_run([
            sys.executable, "-B", "scripts/audit_heat3d_v5_gate4a.py", "--verify-summary",
            "--table", str(table), "--summary-json", str(summary),
        ]).stdout)
        check_models = json.loads(_run([
            sys.executable, "-B", "scripts/audit_heat3d_v5_gate4a.py", "--verify-models",
            "--table", str(table), "--model-params", str(models),
        ]).stdout)
        assert check_summary["verification"] == "passed"
        assert check_models["verification"] == "passed"
        payload = json.loads(summary.read_text(encoding="utf-8"))
        assert payload["input_leakage_guard"]["target_or_oracle_inputs_used"] is False
        assert payload["input_leakage_guard"]["test_roles_used_for_fit_or_selection"] is False
        assert payload["reconstructed_from_table"]["duplicate_leakage"]["pass"]
        model_payload = json.loads(models.read_text(encoding="utf-8"))
        for record in model_payload["model_records"].values():
            assert not set(record["fit_roles"]) & {"test_iid", "hard_challenge_test"}
            assert not any("target" in value.lower() or "residual" in value.lower() for value in record["input_feature_columns"])
        with table.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == sum(ROLE_COUNTS.values())
        assert all(float(row["clean_only_zero_shot_best_physics_only_delta_s_hat"]) == 0.0 for row in rows)
        assert all(float(row["hard_adapted_final_physics_only_delta_s_hat"]) == 0.0 for row in rows)
        assert closeout.exists()
    print("V5 Gate 4A analytic fixture checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
