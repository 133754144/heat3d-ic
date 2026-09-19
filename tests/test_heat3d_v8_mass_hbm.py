from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from rigno.heat3d_v8.adapter import _coordinates_and_volume
from rigno.heat3d_v8.boundary import BOUNDARY_NODE_FEATURES, aggregate_boundary_faces
from rigno.heat3d_v8.fingerprint import geometry_fingerprint
from rigno.heat3d_v8.geometry import masked_cells, validate_domain_mask
from rigno.heat3d_v8.interface import load_interface_representation
from rigno.heat3d_v8.query import QueryChunk, concatenate_query_chunks
from rigno.heat3d_v8.schema import ProvenanceClass, reject_oracle_features
from rigno.heat3d_v8.support import select_v8_support


def test_irregular_non_full_box_domain_mask() -> None:
    shape = (1, 3, 3)
    mask = validate_domain_mask(
        np.asarray([[[1, 1, 0], [1, 0, 0], [1, 1, 1]]], dtype=bool),
        shape,
    )
    coords = np.arange(27, dtype=np.float64).reshape(9, 3)
    volume = np.ones(9)
    selected_coords, selected_volume, indices = masked_cells(
        coords_m=coords,
        control_volume_m3=volume,
        valid_cell_mask=mask,
    )
    assert indices.tolist() == [0, 1, 3, 6, 7, 8]
    assert selected_coords.shape == (6, 3)
    assert float(np.sum(selected_volume)) == 6.0


def test_generic_boundary_descriptor_and_arbitrary_normals() -> None:
    normal = np.asarray([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]) / np.sqrt(2.0)
    result = aggregate_boundary_faces(
        cell_count=1,
        control_volume_m3=np.asarray([2.0]),
        cell_index=np.asarray([0, 0]),
        area_m2=np.asarray([3.0, 3.0]),
        outward_normal=normal,
        conductance_W_K=np.asarray([6.0, 6.0]),
        sink_temperature_K=np.asarray([310.0, 290.0]),
        prescribed_flux_W_m2=np.asarray([2.0, -1.0]),
        reference_temperature_K=300.0,
    )
    assert result.shape == (1, len(BOUNDARY_NODE_FEATURES))
    assert np.isclose(result[0, 0], 3.0)
    assert np.isclose(result[0, 1], 6.0)
    assert np.isclose(result[0, 4], 0.0)
    assert result[0, 5] > 0.0
    assert not any(token in " ".join(BOUNDARY_NODE_FEATURES) for token in ("top", "bottom", "side"))


def test_cell_volume_and_power_conservation() -> None:
    geometry = {
        "shape_zyx": [2, 2, 3],
        "dx_m": 0.5,
        "dy_m": 0.25,
        "z_cell_thicknesses_m": [0.1, 0.2],
        "x_extent_mm": 1500.0,
        "y_extent_mm": 500.0,
        "z_extent_mm": 300.0,
    }
    coords, volume = _coordinates_and_volume(geometry)
    assert coords.shape == (12, 3)
    assert np.isclose(np.sum(volume), 1.5 * 0.5 * 0.3)
    q = np.arange(1, 13, dtype=np.float64)
    expected = sum(float(qi * vi) for qi, vi in zip(q, volume))
    assert np.isclose(np.sum(q * volume), expected)


def test_future_track_b_rejects_oracle() -> None:
    with pytest.raises(ValueError, match="final_k"):
        reject_oracle_features(
            {"geometry": ProvenanceClass.PRE_SOLVE, "final_k": ProvenanceClass.ORACLE}
        )
    reject_oracle_features({"geometry": ProvenanceClass.PRE_SOLVE})


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fingerprint_fixture(root: Path) -> Path:
    (root / "structure").mkdir(parents=True)
    (root / "boundary_conditions").mkdir()
    (root / "temperature_output").mkdir()
    (root / "structure" / "solver_geometry.json").write_text(
        json.dumps({"shape_zyx": [1, 2, 2], "dx_m": 1.0, "dy_m": 1.0, "z_cell_thicknesses_m": [1.0]}),
        encoding="utf-8",
    )
    _write_csv(
        root / "structure" / "component_spatial_placement.csv",
        [{"component": "source", "injection_target": "region", "base_fraction": 1, "dram_fraction": 0, "distribution_method": "fixed", "temperature_metric": "none"}],
    )
    _write_csv(
        root / "structure" / "physical_interface_manifest.csv",
        [{"interface_representation": "rotated_homogenized_tensor", "assigned": False}],
    )
    (root / "boundary_conditions" / "boundary_conditions.json").write_text(
        json.dumps({"architecture": "ignored", "case_specific_boundary_and_design_fields": {"model_name": "ignored"}}),
        encoding="utf-8",
    )
    (root / "temperature_output" / "temperature_map.npy").write_bytes(b"label-one")
    return root


def test_geometry_fingerprint_target_independence(tmp_path: Path) -> None:
    case = _fingerprint_fixture(tmp_path / "case")
    first = geometry_fingerprint(case)
    (case / "temperature_output" / "temperature_map.npy").write_bytes(b"different-label")
    second = geometry_fingerprint(case)
    assert first.sha256 == second.sha256
    assert "architecture label" in second.excluded_sources


def test_geometry_fingerprint_ignores_geometry_case_identity(tmp_path: Path) -> None:
    case = _fingerprint_fixture(tmp_path / "case")
    path = case / "boundary_conditions" / "boundary_conditions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    design = payload["case_specific_boundary_and_design_fields"]
    design["geometry_case_id"] = "case-a"
    path.write_text(json.dumps(payload), encoding="utf-8")
    first = geometry_fingerprint(case)
    design["geometry_case_id"] = "case-b"
    path.write_text(json.dumps(payload), encoding="utf-8")
    second = geometry_fingerprint(case)
    assert first.sha256 == second.sha256


def test_support_selection_target_independence() -> None:
    n = 5000
    kwargs = dict(
        source_mask=np.arange(n) % 5 == 0,
        interface_mask=np.arange(n) % 7 == 0,
        boundary_sink_mask=np.arange(n) < 300,
        control_volume_m3=np.linspace(1.0, 2.0, n),
        sample_id="synthetic",
        count=1024,
        seed=7,
    )
    first = select_v8_support(**kwargs)
    # Temperature and q amplitudes are intentionally absent from the API.
    _unused_temperature = np.linspace(250.0, 500.0, n)
    _unused_q_amplitude = np.linspace(0.0, 1.0e12, n)
    second = select_v8_support(**kwargs)
    assert np.array_equal(first.indices, second.indices)
    assert sum(first.class_counts.values()) == 1024


def test_chunked_query_ordering() -> None:
    chunks = [
        QueryChunk(0, 3, np.zeros((3, 3)), "a"),
        QueryChunk(3, 5, np.zeros((2, 3)), "b"),
    ]
    result = concatenate_query_chunks(
        chunks,
        [np.asarray([[0], [1], [2]]), np.asarray([[3], [4]])],
        expected_count=5,
    )
    assert result[:, 0].tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="ordering"):
        concatenate_query_chunks(
            [QueryChunk(1, 3, np.zeros((2, 3)), "x")],
            [np.zeros((2, 1))],
            expected_count=2,
        )


def test_interface_double_count_guard(tmp_path: Path) -> None:
    case = tmp_path / "case"
    (case / "structure").mkdir(parents=True)
    (case / "thermal_parameters").mkdir()
    _write_csv(
        case / "structure" / "physical_interface_manifest.csv",
        [{"interface_representation": "rotated_homogenized_tensor"}],
    )
    np.savez(
        case / "thermal_parameters" / "final_interface_tbr_map.npz",
        interface_tbr_m2k_w=np.ones((1, 2, 2)),
    )
    _write_csv(case / "thermal_parameters" / "interfaces.csv", [{"double_count_check": "PASS_interface_only_bonding_representation"}])
    with pytest.raises(ValueError, match="double-count"):
        load_interface_representation(
            case_dir=case,
            shape_zyx=(2, 2, 2),
            dx_m=1.0,
            dy_m=1.0,
            control_volume_m3=np.ones(8),
        )
