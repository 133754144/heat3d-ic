from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_htc_converter_preserves_released_pde_coefficients() -> None:
    converter = load_script("convert_v7_g2_semiconductor_case.py")
    arrays, metadata = converter.htc_case(0.1, 0.25)
    coords = arrays["coords"]
    features = arrays["features"]
    assert coords.shape == (51**3, 3)
    assert features.shape == (51**3, 11)
    assert np.all(features[:, :3] == np.float32(0.2))
    assert np.all(features[:, 8] == np.float32(2.0))
    assert np.all(features[:, 9] == np.float32(0.8))
    assert set(np.unique(features[:, 3])) == {0.0, 1.0}
    assert metadata["representation"].startswith("lossless")


def test_htc_converter_tracks_single_vs_multi_contract() -> None:
    converter = load_script("convert_v7_g2_semiconductor_case.py")
    _, single = converter.htc_case(0.1, 0.2)
    _, multi = converter.htc_case(0.1, 0.25, "multi_htc_bc")
    assert single["benchmark"] == "single_htc_bc"
    assert multi["benchmark"] == "multi_htc_bc"


def test_large_npy_selection_is_sample_local(tmp_path: Path) -> None:
    converter = load_script("convert_v7_g2_semiconductor_case.py")
    path = tmp_path / "stack.npy"
    values = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
    np.save(path, values)
    selected = converter.select_array(path, (4, 5), 2)
    assert selected.dtype == np.float32
    assert np.array_equal(selected, values[2].astype(np.float32))


def test_surface_converter_keeps_flux_outside_volumetric_q(tmp_path: Path) -> None:
    converter = load_script("convert_v7_g2_semiconductor_case.py")
    power = np.arange(441, dtype=np.float32).reshape(21, 21)
    path = tmp_path / "surface.npy"
    np.save(path, power)
    arrays, metadata = converter.surface_case("deepoheat_v1_surface", path, None, 0)
    assert np.all(arrays["features_without_surface_flux"][:, 3] == 0)
    assert np.array_equal(arrays["top_neumann_flux_sensors"].reshape(21, 21), power)
    assert metadata["lossless_under_current_11_channel_contract"] is False


def test_population_statistic_hash_is_dtype_and_layout_canonical() -> None:
    statistics = load_script("materialize_v7_g2_p1i_statistics.py")
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    assert statistics.array_sha256(values) == statistics.array_sha256(
        np.asarray([1.0, 2.0, 3.0], dtype=">f8")
    )


def test_radius_audit_reports_edges_and_directional_radii() -> None:
    radius = load_script("audit_v7_g2_gino_radius.py")
    points = [np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])]
    latent = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 1.0, 1.0]])
    result = radius.role_audit(points, latent, input_radius=0.11, output_radius=0.01)
    assert result["input_gno"]["edge_count"] == 3
    assert result["output_gno"]["edge_count"] == 2
    assert result["input_gno"]["radius"] == 0.11
    assert result["output_gno"]["radius"] == 0.01
