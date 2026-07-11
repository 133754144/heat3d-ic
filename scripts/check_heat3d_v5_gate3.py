#!/usr/bin/env python3
"""Analytic fixture checks for V5 Gate 3 boundary/general oracle diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import audit_heat3d_v5_gate3 as gate3  # noqa: E402


ROLE_COUNTS = {
    "train": 2,
    "valid_iid": 1,
    "test_iid": 1,
    "hard_train_holdout": 1,
    "hard_challenge_valid": 1,
    "hard_challenge_test": 1,
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sample_arrays(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(
        [[x, y, z] for x in (0.0, 2.0) for y in (0.0, 1.0) for z in (0.0, 0.7, 2.0)],
        dtype=np.float64,
    )
    z = coords[:, 2]
    x = coords[:, 0]
    y = coords[:, 1]
    q = (1.0 + index * 0.1 + 0.25 * x + 0.1 * y + 0.3 * z).reshape(-1, 1)
    q[np.isclose(z, 0.0)] = 0.0
    k = np.column_stack(
        (
            4.0 + 0.2 * index + 0.5 * x,
            5.0 + 0.1 * index + 0.25 * y,
            2.0 + 0.15 * index + 0.4 * x + 0.2 * z,
        )
    )
    delta = 0.5 * z + 0.08 * x + 0.03 * y + 0.04 * index * z
    delta[np.isclose(z, 0.0)] = 0.0
    temperature = (300.0 + delta).reshape(-1, 1)
    bc = np.zeros((coords.shape[0], 4), dtype=np.float64)
    bc[np.isclose(z, 2.0), 0] = 1.0
    bc[np.isclose(z, 0.0), 1] = 1.0
    bc[np.isclose(z, 0.7), 2] = 1.0
    return coords, q, k, temperature, bc


def _meta() -> dict[str, object]:
    return {
        "bc_feature_names": ["is_top", "is_bottom", "is_side", "is_interior"],
        "boundary_params": {
            "bottom": {"type": "dirichlet", "fixed_temperature_K": 300.0},
            "top": {"type": "robin", "h_W_m2K": 10.0, "ambient_temperature_K": 300.0},
            "side": {"type": "adiabatic"},
        },
        "boundary_regions": [
            {"name": "bottom", "point_indices": [0, 3, 6, 9]},
            {"name": "top", "point_indices": [2, 5, 8, 11]},
            {"name": "sides", "point_indices": [1, 4, 7, 10]},
        ],
        "p5_provenance": {"source_sample_id": None},
    }


def _contract(dataset_id: str, role_counts: dict[str, int]) -> dict[str, object]:
    return {
        "contract_id": "fixture-gate3",
        "dataset_contract": {
            "dataset_id": dataset_id,
            "total_sample_count": sum(role_counts.values()),
            "role_counts": role_counts,
        },
        "current_v5_semantics": {
            "reference_dirichlet_region": "bottom",
            "expected_region_types": {"bottom": "dirichlet", "top": "robin", "side": "adiabatic"},
        },
        "boundary_interface": {"allow_coordinate_fallback": False},
    }


def _write_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    dataset = root / "dataset"
    dataset.mkdir()
    assignments: dict[str, str] = {}
    best: dict[str, np.ndarray] = {}
    final: dict[str, np.ndarray] = {}
    gate1_rows: list[dict[str, object]] = []
    roles = [role for role, count in ROLE_COUNTS.items() for _ in range(count)]
    for index, role in enumerate(roles):
        sample_id = f"sample_{index:04d}"
        assignments[sample_id] = role
        sample = dataset / sample_id
        sample.mkdir()
        coords, q, k, temperature, bc = _sample_arrays(index)
        np.save(sample / "coords.npy", coords)
        np.save(sample / "q_field.npy", q)
        np.save(sample / "k_field.npy", k)
        np.save(sample / "temperature.npy", temperature)
        np.save(sample / "bc_features.npy", bc)
        metadata = _meta()
        metadata["p5_provenance"] = {"source_sample_id": f"origin_{index:04d}"}
        _write_json(sample / "sample_meta.json", metadata)
        volumes, _axes, _inverse, _shape = gate3._control_volumes(coords)
        target_delta = temperature.reshape(-1) - 300.0
        scale = gate3._weighted_rms(target_delta, volumes)
        best_prediction = temperature.reshape(-1) + 0.15 * target_delta + 0.04 * (coords[:, 0] - 1.0)
        final_prediction = temperature.reshape(-1) + 0.1 * target_delta - 0.03 * (coords[:, 1] - 0.5)
        best[sample_id] = best_prediction.reshape(-1, 1)
        final[sample_id] = final_prediction.reshape(-1, 1)
        gate1_prediction = scale * (0.8 + 0.01 * index)
        gate1_rows.append(
            {
                "sample_id": sample_id,
                "s_y_cv_rms_deltaT_K": format(scale, ".17g"),
                "pred_z_collapsed_1d_operator_K": format(gate1_prediction, ".17g"),
                "log_residual_z_collapsed_1d_operator": format(np.log(gate1_prediction / scale), ".17g"),
            }
        )
    split_map = root / "split.json"
    _write_json(
        split_map,
        {"dataset_id": "fixture_p5", "sample_splits": assignments, "actual_counts": ROLE_COUNTS},
    )
    contract = root / "contract.json"
    _write_json(contract, _contract("fixture_p5", ROLE_COUNTS))
    gate1_table = root / "gate1.csv"
    with gate1_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gate1_rows[0]))
        writer.writeheader()
        writer.writerows(gate1_rows)
    best_path = root / "best.npz"
    final_path = root / "final.npz"
    np.savez_compressed(best_path, **best)
    np.savez_compressed(final_path, **final)
    return dataset, split_map, contract, gate1_table, best_path, final_path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=True)


def _generic_boundary_checks() -> None:
    coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    meta = {
        "bc_feature_names": ["is_anchor_a", "is_anchor_b", "is_robin"],
        "boundary_params": {
            "anchor_a": {"type": "dirichlet", "fixed_temperature_K": 280.0},
            "anchor_b": {"type": "dirichlet", "fixed_temperature_K": 310.0},
            "robin": {"type": "robin", "h_W_m2K": 12.0, "ambient_temperature_K": 295.0},
        },
        "boundary_regions": [
            {"name": "anchor_a", "point_indices": [1]},
            {"name": "anchor_b", "point_indices": [2]},
            {"name": "robin", "point_indices": [0]},
        ],
    }
    bc = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    boundary = gate3._resolve_boundary_contract(
        meta=meta,
        bc_features=bc,
        coords=coords,
        reference_region_id="anchor_a",
        allow_coordinate_fallback=False,
    )
    assert np.array_equal(boundary.dirichlet_mask, np.asarray([False, True, True]))
    projected = gate3._boundary_project_raw(np.asarray([299.0, 100.0, 100.0]), boundary)
    assert np.allclose(projected, [299.0, 280.0, 310.0])
    assert projected[0] == 299.0

    neumann = json.loads(json.dumps(meta))
    neumann["boundary_params"]["robin"] = {"type": "neumann", "flux_W_m2": 7.5}
    neumann_boundary = gate3._resolve_boundary_contract(
        meta=neumann,
        bc_features=bc,
        coords=coords,
        reference_region_id="anchor_a",
        allow_coordinate_fallback=False,
    )
    assert {region.boundary_type for region in neumann_boundary.regions} == {"dirichlet", "neumann"}

    bad = json.loads(json.dumps(meta))
    bad["boundary_params"]["robin"]["type"] = "unrecognized"
    try:
        gate3._resolve_boundary_contract(
            meta=bad,
            bc_features=bc,
            coords=coords,
            reference_region_id="anchor_a",
            allow_coordinate_fallback=False,
        )
    except gate3.AuditError as exc:
        assert "unsupported or unknown" in str(exc)
    else:
        raise AssertionError("unknown BC type must fail loudly")

    mixed = json.loads(json.dumps(meta))
    mixed["boundary_params"]["robin"]["type"] = "mixed"
    try:
        gate3._resolve_boundary_contract(
            meta=mixed,
            bc_features=bc,
            coords=coords,
            reference_region_id="anchor_a",
            allow_coordinate_fallback=False,
        )
    except gate3.AuditError as exc:
        assert "must be expanded" in str(exc)
    else:
        raise AssertionError("literal mixed BC must fail loudly")

    fallback_meta = {
        "bc_feature_names": [],
        "boundary_params": {
            "right_anchor": {
                "type": "dirichlet",
                "fixed_temperature_K": 290.0,
                "coordinate_fallback": {"axis": "x", "extremum": "max", "tolerance": 1.0e-12},
            }
        },
        "boundary_regions": [],
    }
    try:
        gate3._resolve_boundary_contract(
            meta=fallback_meta,
            bc_features=np.zeros((3, 0)),
            coords=coords,
            reference_region_id="right_anchor",
            allow_coordinate_fallback=False,
        )
    except gate3.AuditError as exc:
        assert "coordinate inference is disabled" in str(exc)
    else:
        raise AssertionError("coordinate fallback must require explicit opt-in")
    fallback = gate3._resolve_boundary_contract(
        meta=fallback_meta,
        bc_features=np.zeros((3, 0)),
        coords=coords,
        reference_region_id="right_anchor",
        allow_coordinate_fallback=True,
    )
    assert np.array_equal(fallback.dirichlet_mask, np.asarray([False, False, True]))
    assert fallback.coordinate_fallback_used

    leakage = gate3._duplicate_summary(
        [
            {"sample_id": "sample_a", "role": "train", "input_fingerprint": "same", "full_fingerprint": "a", "provenance_source_id": "p_a"},
            {"sample_id": "sample_b", "role": "test_iid", "input_fingerprint": "same", "full_fingerprint": "b", "provenance_source_id": "p_b"},
        ]
    )
    assert not leakage["pass"]
    assert leakage["cross_role_model_input_duplicate_groups"]["group_count"] == 1


def main() -> int:
    _generic_boundary_checks()
    with tempfile.TemporaryDirectory(prefix="heat3d_v5_gate3_") as temporary:
        root = Path(temporary)
        dataset, split_map, contract, gate1_table, best, final = _write_fixture(root)
        output_table = root / "table.csv"
        output_json = root / "summary.json"
        output_md = root / "closeout.md"
        base = [
            sys.executable,
            "-B",
            "scripts/audit_heat3d_v5_gate3.py",
            "--dataset",
            str(dataset),
            "--split-map",
            str(split_map),
            "--contract",
            str(contract),
            "--gate1-table",
            str(gate1_table),
            "--best-predictions",
            str(best),
            "--final-predictions",
            str(final),
        ]
        dry = _run(base + ["--dry-run"])
        dry_payload = json.loads(dry.stdout)
        assert dry_payload["mode"] == "dry_run"
        assert dry_payload["planned_writes"] == []
        assert dry_payload["prediction_coverage"] == {"best": 7, "final": 7}
        _run(
            base
            + [
                "--output-table",
                str(output_table),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
        )
        verify = _run(
            [
                sys.executable,
                "-B",
                "scripts/audit_heat3d_v5_gate3.py",
                "--verify-summary",
                "--table",
                str(output_table),
                "--summary-json",
                str(output_json),
            ]
        )
        assert json.loads(verify.stdout)["verification"] == "passed"
        summary = json.loads(output_json.read_text(encoding="utf-8"))
        reconstructed = summary["reconstructed_from_table"]
        assert reconstructed["row_count"] == 7
        assert reconstructed["target_decomposition"]["decomposition_pass_count"] == 7
        assert reconstructed["target_decomposition"]["coordinate_fallback_used_count"] == 0
        assert reconstructed["duplicate_leakage"]["pass"]
        hard = reconstructed["hard_failure_decomposition"]
        assert set(hard) == {"best", "final"}
        assert output_md.exists()
        with output_table.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 7
        for row in rows:
            assert row["target_decomposition_pass"] == "1"
            assert abs(float(row["target_shape_cv_rms"]) - 1.0) <= 1.0e-9
            assert float(row["target_projected_dirichlet_max_abs_error_K"]) <= gate3.DIRICHLET_TOL_K
            assert float(row["target_projection_non_dirichlet_max_abs_change_K"]) <= gate3.RECONSTRUCTION_TOL_K
            for checkpoint in gate3.CHECKPOINTS:
                assert row[f"{checkpoint}_prediction_available"] == "1"
                assert float(row[f"{checkpoint}_boundary_projection_dirichlet_max_abs_error_K"]) <= gate3.DIRICHLET_TOL_K
                assert float(row[f"{checkpoint}_boundary_projection_non_dirichlet_max_abs_change_K"]) <= gate3.RECONSTRUCTION_TOL_K
                for variant in gate3.ORACLE_VARIANTS:
                    assert row[f"{checkpoint}_{variant}_cv_rmse_K"] != ""
    print("V5 Gate 3 analytic fixture checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
