from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    path = ROOT / "scripts" / "generate_v7_g2_p7_multi_htc_train_valid_labels.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multi_htc_historical_serialization_exactly_recovers_frozen_sha():
    generator = load_generator()
    rows = generator.canonical_rows()
    assert generator.canonical_rows_sha256(rows) == generator.FROZEN_CANONICAL_ROWS_SHA256
    assert rows[0]["case_id"] == 0
    assert rows[-1]["case_id"] == 1023
    assert [sum(row["role"] == role for row in rows) for role in ("train", "valid", "test")] == [768, 128, 128]


def test_multi_htc_label_obeys_robin_contract_without_test_input():
    generator = load_generator()
    beta_top, beta_bottom = 0.13, 0.27
    a1, b1, _a2, _b2, a3, b3 = generator.analytical_coefficients(beta_top, beta_bottom)
    assert np.isclose(b1 - beta_bottom * a1, generator.AMBIENT_U)
    assert np.isclose(a3 * generator.DOMAIN_Z + b3 + beta_top * a3, generator.AMBIENT_U)
    profile = generator.analytical_delta_t_z(beta_top, beta_bottom)
    assert profile.shape == (51,)
    assert np.isfinite(profile).all()
