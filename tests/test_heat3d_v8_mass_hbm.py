from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rigno.heat3d_v8.adapter import (
    CORE_LOCAL_FEATURES,
    V8_PHYSICAL_SCALE_FEATURES,
    _coordinates_and_volume,
    physical_scale_features,
)
from rigno.heat3d_v8.bridge import build_canonical_bridge
from rigno.heat3d_v8.boundary import (
    BOUNDARY_NODE_FEATURES,
    aggregate_boundary_faces,
    cuboid_external_faces,
)
from rigno.heat3d_v8.fingerprint import geometry_fingerprint
from rigno.heat3d_v8.geometry import masked_cells, validate_domain_mask
from rigno.heat3d_v8.interface import INTERFACE_NODE_FEATURES, load_interface_representation
from rigno.heat3d_v8.query import QueryChunk, concatenate_query_chunks
from rigno.heat3d_v8.schema import (
    MassHBMCase,
    ProvenanceClass,
    SupportProvenance,
    V8InterfaceRepresentation,
    V8OraclePhysicsView,
    reject_oracle_features,
)
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


def test_physical_scale_features_preserve_real_extent() -> None:
    small = SimpleNamespace(case=SimpleNamespace(geometry_metadata={
        "x_extent_mm": 65.0, "y_extent_mm": 65.0, "z_extent_mm": 1.51,
    }))
    tall = SimpleNamespace(case=SimpleNamespace(geometry_metadata={
        "x_extent_mm": 65.0, "y_extent_mm": 65.0, "z_extent_mm": 5.76,
    }))
    first = physical_scale_features(small)
    second = physical_scale_features(tall)
    assert tuple(first) == V8_PHYSICAL_SCALE_FEATURES
    assert np.isclose(first["log_Lx_m"], np.log(0.065))
    assert first["log_Lz_m"] != second["log_Lz_m"]
    assert first["log_domain_volume_m3"] != second["log_domain_volume_m3"]


def _canonical_view(tmp_path: Path) -> V8OraclePhysicsView:
    geometry = {
        "shape_zyx": [2, 2, 2], "dx_m": 0.5, "dy_m": 0.5,
        "z_cell_thicknesses_m": [0.25, 0.25],
        "x_extent_mm": 1000.0, "y_extent_mm": 1000.0, "z_extent_mm": 500.0,
    }
    coords, volume = _coordinates_and_volume(geometry)
    boundary = cuboid_external_faces(
        shape_zyx=(2, 2, 2), dx_m=0.5, dy_m=0.5,
        z_cell_thicknesses_m=np.asarray([0.25, 0.25]),
        top_h_W_m2K=10.0, bottom_h_W_m2K=10.0, side_h_W_m2K=1.0,
        ambient_temperature_K=300.0, control_volume_m3=volume,
        reference_temperature_K=300.0,
    )
    interface = V8InterfaceRepresentation(
        mode="rotated_homogenized_tensor",
        lower_cell_index=np.empty(0, dtype=np.int64),
        upper_cell_index=np.empty(0, dtype=np.int64),
        area_m2=np.empty(0), normal=np.empty((0, 3)), tbr_m2K_W=np.empty(0),
        conductance_W_K=np.empty(0), node_feature_names=INTERFACE_NODE_FEATURES,
        node_features=np.zeros((8, len(INTERFACE_NODE_FEATURES))),
        provenance={name: ProvenanceClass.ORACLE for name in INTERFACE_NODE_FEATURES},
        double_count_guard="PASS", explicit_feature_safe=False,
    )
    k = np.full((8, 3), 10.0)
    q = np.full(8, 1.0e8)
    local = np.column_stack((k, q, volume, boundary.node_features, interface.node_features))
    names = CORE_LOCAL_FEATURES + BOUNDARY_NODE_FEATURES + INTERFACE_NODE_FEATURES
    provenance = {
        **{name: ProvenanceClass.ORACLE for name in CORE_LOCAL_FEATURES[:4]},
        "cell_volume_m3": ProvenanceClass.PRE_SOLVE,
        **dict(boundary.provenance), **dict(interface.provenance),
    }
    case = MassHBMCase(
        dataset_root=tmp_path, case_directory="synthetic", sample_id="synthetic",
        architecture_metadata="metadata_only", shape_zyx=(2, 2, 2), coords_m=coords,
        control_volume_m3=volume, valid_cell_mask=np.ones(8, dtype=bool),
        geometry_metadata=geometry, boundary=boundary, interface=interface,
    )
    return V8OraclePhysicsView(
        case=case, k_diag_W_mK=k, q_W_m3=q,
        target_temperature_K=np.full(8, 310.0), local_feature_names=names,
        local_features=local, feature_provenance=provenance,
        metadata={"reference_temperature_K": 300.0},
    )


def test_canonical_bridge_is_zero_u_all_physics_c(tmp_path: Path) -> None:
    bridge = build_canonical_bridge(_canonical_view(tmp_path))
    assert np.all(np.asarray(bridge.inputs.u) == 0.0)
    assert bridge.inputs.c.shape == (1, 1, 8, 27)
    assert bridge.condition_feature_names[-5:] == V8_PHYSICAL_SCALE_FEATURES
    assert not np.any(np.asarray(bridge.inputs.u) == np.asarray(bridge.inputs.c[..., :1]))


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


def test_geometry_fingerprint_label_array_independence(tmp_path: Path) -> None:
    case = _fingerprint_fixture(tmp_path / "case")
    first = geometry_fingerprint(case)
    (case / "temperature_output" / "temperature_map.npy").write_bytes(b"different-label")
    second = geometry_fingerprint(case)
    assert first.sha256 == second.sha256
    assert "architecture label" in second.excluded_sources
    assert second.independence_claim.startswith("LABEL_ARRAY_INDEPENDENT")


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
        support_provenance=SupportProvenance.PRE_SOLVE_SUPPORT,
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
    assert first.provenance == SupportProvenance.PRE_SOLVE_SUPPORT


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
