#!/usr/bin/env python3
"""Analytic fixture checks for the V5 Gate 1 operator-scale closeout."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "scripts" / "audit_heat3d_v5_gate1.py"
SPEC = importlib.util.spec_from_file_location("heat3d_v5_gate1", AUDIT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Gate 1 audit module")
GATE1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE1)


def _grid_coords(
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    z: np.ndarray | None = None,
) -> np.ndarray:
    x = np.array([0.0, 1.0], dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64) if y is None else np.asarray(y, dtype=np.float64)
    z = np.array([0.0, 1.0, 2.0], dtype=np.float64) if z is None else np.asarray(z, dtype=np.float64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])


def _coords() -> np.ndarray:
    return _grid_coords()


def _boundary_features(coords: np.ndarray) -> tuple[np.ndarray, list[int]]:
    features = np.zeros((coords.shape[0], 4), dtype=np.float64)
    bottom = np.isclose(coords[:, 2], 0.0)
    top = np.isclose(coords[:, 2], 2.0)
    side = (
        np.isclose(coords[:, 0], 0.0)
        | np.isclose(coords[:, 0], 1.0)
        | np.isclose(coords[:, 1], 0.0)
        | np.isclose(coords[:, 1], 1.0)
    )
    features[:, 3] = 1.0
    features[side] = np.array([0.0, 0.0, 1.0, 0.0])
    features[top] = np.array([1.0, 0.0, 0.0, 0.0])
    features[bottom] = np.array([0.0, 1.0, 0.0, 0.0])
    return features, np.nonzero(bottom)[0].astype(int).tolist()


def _write_sample(
    dataset: Path,
    sample_id: str,
    *,
    source_q: float,
    bottom_q: float = 0.0,
    bc_offset: float = 0.0,
    top_h: float = 1.0,
    anisotropic: bool = False,
    provenance: str | None = None,
) -> None:
    sample = dataset / sample_id
    sample.mkdir()
    coords = _coords()
    q = np.zeros((coords.shape[0], 1), dtype=np.float64)
    q[np.isclose(coords[:, 2], 1.0), 0] = source_q
    q[np.isclose(coords[:, 2], 0.0), 0] = bottom_q
    if anisotropic:
        k = np.tile(np.array([[4.0, 6.0, 2.0]], dtype=np.float64), (coords.shape[0], 1))
    else:
        k = np.full((coords.shape[0], 1), 2.0, dtype=np.float64)
    bc, bottom_indices = _boundary_features(coords)
    # The label need only obey the Dirichlet boundary; it is not used by a solver.
    delta = (source_q * 0.4 + bc_offset) * (coords[:, 2:3] / 2.0)
    temperature = 300.0 + delta
    for name, value in {
        "coords": coords,
        "k_field": k,
        "q_field": q,
        "bc_features": bc,
        "temperature": temperature,
    }.items():
        np.save(sample / f"{name}.npy", value)
    meta = {
        "sample_id": sample_id,
        "bc_feature_names": ["is_top", "is_bottom", "is_side", "is_interior"],
        "boundary_regions": [{"name": "bottom", "point_indices": bottom_indices}],
        "boundary_params": {
            "bottom": {"fixed_temperature_K": 300.0},
            "top": {"ambient_temperature_K": 300.0 + bc_offset, "h_W_m2K": top_h},
        },
        "p5_provenance": {"source_sample_id": provenance or sample_id},
    }
    (sample / "sample_meta.json").write_text(json.dumps(meta))


def _minimal_contract(role_counts: dict[str, int]) -> dict:
    return {
        "contract_id": "fixture-gate1",
        "dataset_contract": {
            "dataset_id": "fixture_p5",
            "role_counts": role_counts,
            "total_sample_count": sum(role_counts.values()),
        },
        "calibration_and_selection": {
            "fit_role": "train",
            "selection_role": "valid_iid",
            "ood_inspection_role": "hard_challenge_valid",
            "test_roles": ["test_iid", "hard_challenge_test"],
        },
    }


def _assert_close(actual: float, expected: float, name: str, atol: float = 1.0e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=atol):
        raise AssertionError(f"{name}: {actual} != {expected}")


def _check_operator_semantics(root: Path) -> None:
    dataset = root / "operator_dataset"
    dataset.mkdir()
    _write_sample(dataset, "sample_0000", source_q=2.0, bottom_q=5.0)
    row = GATE1._sample_row(dataset / "sample_0000", "train")
    # 2x2x3 grid: bottom/middle/top layer volumes are 0.5/1.0/0.5 m3.
    _assert_close(row["P_array_W"], 2.0 + 2.5, "P_array")
    _assert_close(row["P_bottom_W"], 2.5, "P_bottom")
    _assert_close(row["P_operator_W"], 2.0, "P_operator")
    assert row["bottom_q_nonzero_count"] == 4
    assert row["bottom_mask_matches_bc"] == 1
    assert row["bottom_mask_matches_metadata"] == 1
    assert row["bottom_temperature_label_pass"] == 1
    assert row["driver_category"] == "source_driven"

    _write_sample(dataset, "sample_0001", source_q=0.0, bc_offset=20.0)
    bc_only = GATE1._sample_row(dataset / "sample_0001", "train")
    _assert_close(bc_only["P_operator_W"], 0.0, "BC-only P_operator")
    _assert_close(bc_only["T_inf_minus_T_bottom_K"], 20.0, "BC offset")
    assert bc_only["driver_category"] == "bc_driven"
    assert bc_only["raw_z_collapsed_1d_operator_K"] > 0.0
    assert bc_only["raw_power_only_W"] is None


def _check_q_linearity_and_1d(root: Path) -> None:
    dataset = root / "linearity_dataset"
    dataset.mkdir()
    _write_sample(dataset, "sample_0000", source_q=1.0, anisotropic=True)
    _write_sample(dataset, "sample_0001", source_q=2.0, anisotropic=True)
    one = GATE1._sample_row(dataset / "sample_0000", "train")
    two = GATE1._sample_row(dataset / "sample_0001", "train")
    for candidate in (
        "power_only",
        "q_rms_lz2_over_kz",
        "legacy_p_array_r_series",
        "source_centroid_two_path",
        "legacy_z_collapsed_1d",
        "z_collapsed_1d_operator",
    ):
        _assert_close(
            float(two[GATE1._raw_column(candidate)]) / float(one[GATE1._raw_column(candidate)]),
            2.0,
            f"q scaling {candidate}",
        )

    axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])]
    coords = _coords()
    volumes, _, inverse, _ = GATE1._control_volumes(coords)
    q_operator = np.zeros(coords.shape[0], dtype=np.float64)
    q_operator[np.isclose(coords[:, 2], 1.0)] = 1.0
    value, details = GATE1.z_collapsed_1d_operator(
        axes=axes,
        inverse=inverse,
        volumes=volumes,
        q_operator=q_operator,
        k_z=np.ones(coords.shape[0]),
        top_h=1.0,
        bc_offset=0.0,
        return_details=True,
    )
    # Independent V4-z network uses actual node distance, so G=1 per face:
    # T=[0, 2/3, 1/3], layer volumes=.5,1,.5.
    expected = math.sqrt((1.0 * (2.0 / 3.0) ** 2 + 0.5 * (1.0 / 3.0) ** 2) / 2.0)
    _assert_close(value, expected, "uniform operator 1D proxy")
    _assert_close(details["face_conductance_W_K"][0], 1.0, "uniform lower face")
    _assert_close(details["face_conductance_W_K"][1], 1.0, "uniform upper face")


def _check_nonuniform_operator_1d() -> None:
    axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 0.5, 2.0])]
    coords = _grid_coords(*axes)
    volumes, _, inverse, _ = GATE1._control_volumes(coords)
    q_operator = np.zeros(coords.shape[0], dtype=np.float64)
    q_operator[np.isclose(coords[:, 2], 0.5)] = 1.0
    value, details = GATE1.z_collapsed_1d_operator(
        axes=axes,
        inverse=inverse,
        volumes=volumes,
        q_operator=q_operator,
        k_z=np.ones(coords.shape[0]),
        top_h=1.0,
        bc_offset=0.0,
        return_details=True,
    )
    g_bottom, g_top = 2.0, 2.0 / 3.0
    matrix = np.array([[1.0, 0.0, 0.0], [-g_bottom, g_bottom + g_top, -g_top], [0.0, -g_top, g_top + 1.0]])
    delta = np.linalg.solve(matrix, np.array([0.0, 1.0, 0.0]))
    expected = math.sqrt((1.0 * delta[1] ** 2 + 0.75 * delta[2] ** 2) / 2.0)
    _assert_close(value, expected, "nonuniform-z operator 1D proxy")
    _assert_close(details["face_conductance_W_K"][0], g_bottom, "nonuniform-z lower face")
    _assert_close(details["face_conductance_W_K"][1], g_top, "nonuniform-z upper face")


def _check_xy_local_kz_operator_1d() -> None:
    axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])]
    coords = _grid_coords(*axes)
    volumes, _, inverse, shape = GATE1._control_volumes(coords)
    grid = -np.ones(tuple(shape), dtype=np.int64)
    grid[inverse[0], inverse[1], inverse[2]] = np.arange(coords.shape[0])
    k_z = np.ones(coords.shape[0], dtype=np.float64)
    lower = np.array([[1.0, 2.0], [4.0, 8.0]])
    middle = np.array([[2.0, 4.0], [8.0, 16.0]])
    top = np.array([[3.0, 6.0], [12.0, 24.0]])
    for ix in range(2):
        for iy in range(2):
            k_z[grid[ix, iy, 0]] = lower[ix, iy]
            k_z[grid[ix, iy, 1]] = middle[ix, iy]
            k_z[grid[ix, iy, 2]] = top[ix, iy]
    q_operator = np.zeros(coords.shape[0], dtype=np.float64)
    q_operator[grid[:, :, 1].reshape(-1)] = 1.0
    value, details = GATE1.z_collapsed_1d_operator(
        axes=axes,
        inverse=inverse,
        volumes=volumes,
        q_operator=q_operator,
        k_z=k_z,
        top_h=1.0,
        bc_offset=0.0,
        return_details=True,
    )
    harmonic = lambda left, right: 2.0 * left * right / (left + right)
    g_bottom = sum(0.25 * harmonic(lower[ix, iy], middle[ix, iy]) for ix in range(2) for iy in range(2))
    g_top = sum(0.25 * harmonic(middle[ix, iy], top[ix, iy]) for ix in range(2) for iy in range(2))
    matrix = np.array([[1.0, 0.0, 0.0], [-g_bottom, g_bottom + g_top, -g_top], [0.0, -g_top, g_top + 1.0]])
    delta = np.linalg.solve(matrix, np.array([0.0, 1.0, 0.0]))
    expected = math.sqrt((delta[1] ** 2 + 0.5 * delta[2] ** 2) / 2.0)
    _assert_close(value, expected, "x-y local-kz operator 1D proxy")
    _assert_close(details["face_conductance_W_K"][0], g_bottom, "x-y lower face sum")
    _assert_close(details["face_conductance_W_K"][1], g_top, "x-y upper face sum")


def _check_full_fixture_and_reconstruction(root: Path) -> None:
    dataset = root / "full_dataset"
    dataset.mkdir()
    assignments = {
        "sample_0000": "train",
        "sample_0001": "train",
        "sample_0002": "train",
        "sample_0003": "valid_iid",
        "sample_0004": "valid_iid",
        "sample_0005": "test_iid",
        "sample_0006": "hard_train_holdout",
        "sample_0007": "hard_challenge_valid",
        "sample_0008": "hard_challenge_test",
    }
    for index, sample_id in enumerate(assignments):
        _write_sample(
            dataset,
            sample_id,
            source_q=1.0 + 0.3 * index,
            top_h=1.0 + 0.1 * index,
            anisotropic=index % 2 == 0,
        )
    counts = dict(sorted(__import__("collections").Counter(assignments.values()).items()))
    split = root / "split.json"
    split.write_text(json.dumps({"dataset_id": "fixture_p5", "actual_counts": counts, "sample_splits": assignments}))
    contract = root / "contract.json"
    contract.write_text(json.dumps(_minimal_contract(counts)))
    table = root / "table.csv"
    summary = root / "summary.json"
    markdown = root / "summary.md"
    assert GATE1.main([
        "--dataset", str(dataset), "--split-map", str(split), "--contract", str(contract), "--dry-run",
    ]) == 0
    assert not table.exists()
    assert GATE1.main([
        "--dataset", str(dataset), "--split-map", str(split), "--contract", str(contract),
        "--output-table", str(table), "--output-json", str(summary), "--output-md", str(markdown),
        "--table-label", "configs/fixture/table.csv",
    ]) == 0
    assert GATE1.main(["--verify-summary", "--table", str(table), "--summary-json", str(summary)]) == 0
    payload = json.loads(summary.read_text())
    assert payload["dataset"]["sample_count"] == len(assignments)
    assert payload["per_sample_table"]["path"] == "configs/fixture/table.csv"
    assert payload["selection"]["test_roles_used_for_selection"] is False
    assert payload["reconstructed_from_table"]["duplicate_leakage"]["pass"] is True
    with table.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(assignments)
    assert "P_operator_W" in rows[0]
    assert "raw_power_only_W" in rows[0]
    assert "pred_z_collapsed_1d_operator_K" in rows[0]
    assert payload["selection"]["winner_vs_runner_up_paired_bootstrap"] is not None
    rendered = markdown.read_text()
    assert "Split" not in rendered
    assert "BC offset range K" in rendered
    assert "| train |" in rendered


def _check_role_leakage_guard() -> None:
    rows = [
        {"sample_id": "sample_a", "role": "train", "input_fingerprint": "same", "full_fingerprint": "same", "provenance_source_id": "origin_a"},
        {"sample_id": "sample_b", "role": "valid_iid", "input_fingerprint": "same", "full_fingerprint": "same", "provenance_source_id": "origin_a"},
    ]
    leakage = GATE1._duplicate_summary(rows)
    assert leakage["pass"] is False
    assert leakage["cross_role_model_input_duplicate_groups"]["group_count"] == 1
    assert leakage["cross_role_full_sample_duplicate_groups"]["group_count"] == 1
    assert leakage["cross_role_provenance_duplicate_groups"]["group_count"] == 1


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heat3d_v5_gate1_") as temporary:
        root = Path(temporary)
        _check_operator_semantics(root)
        _check_q_linearity_and_1d(root)
        _check_nonuniform_operator_1d()
        _check_xy_local_kz_operator_1d()
        _check_full_fixture_and_reconstruction(root)
        _check_role_leakage_guard()
    print("V5 Gate 1 analytic fixture checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
