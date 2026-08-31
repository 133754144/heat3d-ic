from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_transolver_uses_decoded_physical_space_objective() -> None:
    manifest = load("configs/heat3d_v7/g2_transolver_formal_launch_manifest.json")
    assert manifest["objective"] == (
        "decode_prediction_and_target_to_physical_deltaT_K_then_"
        "upstream_TestLoss_size_average_false_relative_L2"
    )
    assert manifest["normalization"]["decode_target_before_metrics"] is True


def test_v1_subset_indices_are_complete_disjoint_and_hash_bound() -> None:
    manifest = load("configs/heat3d_v7/g2_deepoheat_v1_volumetric_subset_manifest.json")
    decoded = {}
    for role, count, expected_sha in (
        ("train", 768, "2ca5fb6efe152ddbba3da0f8dedcc65737c82e6f8df16f9eeb48092c2c74dd49"),
        ("valid", 128, "6c04f5188fd83d0ce9ca4689cce526ed36e08a4a21421ed2af93ac7a1b6f20d4"),
    ):
        raw = base64.b64decode(manifest["roles"][role]["indices_base64"])
        values = np.frombuffer(raw, dtype="<u4").astype("<i8")
        assert len(values) == count
        assert np.all(values[:-1] < values[1:])
        assert hashlib.sha256(values.tobytes()).hexdigest() == expected_sha
        decoded[role] = set(map(int, values))
    assert decoded["train"].isdisjoint(decoded["valid"])
    assert manifest["official_test_release"]["untouched_by_selection"] is True
    assert manifest["temperature_labels_generated"] == 0


def test_multi_htc_case_table_hash_is_reproducible() -> None:
    manifest = load("configs/heat3d_v7/g2_deepoheat_multi_htc_case_manifest.json")
    rows = []
    for position in range(1024):
        lattice = (405 * position + 97) % 1024
        top_index, bottom_index = lattice % 32, lattice // 32
        role = "train" if position < 768 else ("valid" if position < 896 else "test")
        rows.append({
            "bottom_h": round(0.1 + bottom_index * 0.2 / 31, 4),
            "case_id": position,
            "role": role,
            "top_h": round(0.1 + top_index * 0.2 / 31, 4),
        })
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == manifest["case_table"]["canonical_rows_sha256"]


def test_p4_receipts_and_correct_paper_title() -> None:
    title = load("docs/v7_g2_p3_deepoheat_v1_data_receipt.json")["official_release"]["paper"]
    assert title == (
        "DeepOHeat-v1: Efficient Operator Learning for Fast and Trustworthy "
        "Thermal Simulation and Optimization in 3D-IC Design"
    )
    solver = load("docs/v7_g2_p4_deepoheat_v1_solver_fidelity_receipt.json")
    local = load("docs/v7_g2_p4_local_execution_receipt.json")
    assert solver["status"].startswith("PASS")
    assert max(row["field_rmse_deltaT_K"] for row in solver["cases"]) < 0.05
    assert local["DeepOHeat_v1_original_training_path"]["status"].startswith("PASS")
    assert local["Heat3D_on_DeepOHeat_v1_volumetric"]["status"].startswith("PASS")
    assert local["hard_boundaries"]["p1i_test_or_sealed_access"] is False


def test_remote_manifests_are_prepared_but_not_launched() -> None:
    protocol = load("configs/heat3d_v7/g2_formal_preparation_protocol_v5.json")
    semiconductor = load("configs/heat3d_v7/g2_semiconductor_remote_launch_manifest.json")
    assert protocol["hard_boundaries"]["formal_or_long_training_started"] is False
    assert protocol["hard_boundaries"]["P1i_test_iid_or_sealed_access"] is False
    assert semiconductor["formal_training_started"] is False
    assert "G1_21_run_complete" in semiconductor["global_gate"]
