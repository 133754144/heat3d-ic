from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v1_support_is_deterministic_label_independent_and_quota_exact():
    selector = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    power = np.zeros((101, 101), dtype=np.float64)
    power[10:30, 20:40] = 3.0
    power[60:80, 70:90] = 17.0
    first, strata, audit = selector.select_support(power, source_index=4, role="train")
    second, _, _ = selector.select_support(power * 9.0, source_index=4, role="train")
    assert np.array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 1024
    assert {name: strata.count(name) for name in selector.QUOTAS} == selector.QUOTAS
    assert min(audit["source_component_support_counts"]) > 0
    assert audit["temperature_prediction_error_or_test_used"] is False


def test_multi_htc_analytical_solution_satisfies_robin_and_interface_contract():
    reference = load_script("check_v7_g2_p5_multi_htc_reference.py")
    beta_top, beta_bottom = 0.13, 0.27
    coefficients = reference.analytical_coefficients(beta_top, beta_bottom)
    a1, b1, a2, b2, a3, b3 = coefficients
    assert np.isclose(b1 - beta_bottom * a1, reference.AMBIENT_U)
    assert np.isclose(a3 * reference.DOMAIN_Z + b3 + beta_top * a3, reference.AMBIENT_U)
    za, zb = reference.SOURCE_START, reference.SOURCE_END
    assert np.isclose(a1 * za + b1, -(za**2)/(2*reference.K) + a2*za + b2)
    assert np.isclose(-(zb**2)/(2*reference.K) + a2*zb + b2, a3*zb + b3)


def test_multi_htc_converter_names_beta_and_derives_physical_h():
    converter = load_script("convert_v7_g2_semiconductor_case.py")
    arrays, metadata = converter.htc_case(0.1, 0.25, "multi_htc_bc")
    contract = metadata["upstream_robin_parameter"]
    assert contract["name"] == "beta_or_k_Robin"
    assert contract["top_h"] == 2.0
    assert contract["bottom_h"] == 0.8
    assert np.allclose(arrays["features"][:, 8], 2.0)
    assert np.allclose(arrays["features"][:, 9], 0.8)
