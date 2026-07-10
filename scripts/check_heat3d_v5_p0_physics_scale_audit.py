#!/usr/bin/env python3
"""Fixture check for the V5-P0-1 read-only physics-scale audit."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit_heat3d_v5_p5_physics_scale.py"


def _coords() -> np.ndarray:
    axes = np.array([0.0, 1.0], dtype=np.float64)
    xx, yy, zz = np.meshgrid(axes, axes, axes, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])


def _write_sample(
    dataset: Path,
    sample_id: str,
    *,
    q_scale: float,
    top_h: float,
    provenance_id: str,
    isotropic_k: bool = False,
) -> None:
    sample_dir = dataset / sample_id
    sample_dir.mkdir()
    coords = _coords()
    q = np.full((coords.shape[0], 1), q_scale, dtype=np.float64)
    if isotropic_k:
        k = np.full((coords.shape[0], 1), 10.0, dtype=np.float64)
    else:
        k = np.tile(np.array([[10.0, 10.0, 5.0]], dtype=np.float64), (coords.shape[0], 1))
    bc = np.zeros((coords.shape[0], 4), dtype=np.float64)
    bc[:, 3] = 1.0
    bc[coords[:, 2] == 0.0] = np.array([0.0, 1.0, 0.0, 0.0])
    bc[coords[:, 2] == 1.0] = np.array([1.0, 0.0, 0.0, 0.0])
    temperature = 300.0 + (q_scale / 10.0) * (1.0 + coords[:, 2:3])
    for name, value in {
        "coords": coords,
        "k_field": k,
        "q_field": q,
        "bc_features": bc,
        "temperature": temperature,
    }.items():
        np.save(sample_dir / f"{name}.npy", value)
    power = float(q.reshape(-1).mean())
    meta = {
        "sample_id": sample_id,
        "boundary_params": {
            "bottom": {"T_fixed_K": 300.0},
            "top": {"h_W_m2K": top_h},
        },
        "q_power_audit": {
            "control_volume_weight_sum_m3": 1.0,
            "q_integral_from_array_W": power,
            "q_total_target_power_W": power,
        },
        "p5_provenance": {"source_sample_id": provenance_id},
    }
    (sample_dir / "sample_meta.json").write_text(json.dumps(meta))


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), *arguments],
        check=True,
        text=True,
        capture_output=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heat3d_v5_p0_audit_") as temporary:
        root = Path(temporary)
        dataset = root / "dataset"
        dataset.mkdir()
        _write_sample(dataset, "sample_0000", q_scale=2.0, top_h=10.0, provenance_id="source_a")
        _write_sample(
            dataset,
            "sample_0001",
            q_scale=4.0,
            top_h=20.0,
            provenance_id="source_b",
            isotropic_k=True,
        )
        _write_sample(dataset, "sample_0002", q_scale=2.0, top_h=10.0, provenance_id="source_a")

        split_map = root / "splits.json"
        split_map.write_text(
            json.dumps(
                {
                    "dataset_id": "fixture_p5",
                    "actual_counts": {"train": 2, "valid_iid": 1},
                    "sample_splits": {
                        "sample_0000": "train",
                        "sample_0001": "train",
                        "sample_0002": "valid_iid",
                    },
                }
            )
        )
        contract = root / "contract.json"
        contract.write_text(
            json.dumps(
                {
                    "contract_id": "fixture-contract",
                    "dataset_contract": {
                        "dataset_id": "fixture_p5",
                        "required_split_roles": ["train", "valid_iid"],
                        "expected_role_counts": {"train": 2, "valid_iid": 1},
                        "expected_total_sample_count": 3,
                    },
                    "audit_contract": {"mode": "read_only"},
                }
            )
        )

        dry = _run(
            [
                "--dataset",
                str(dataset),
                "--split-map",
                str(split_map),
                "--contract",
                str(contract),
                "--dry-run",
            ]
        )
        dry_payload = json.loads(dry.stdout)
        assert dry_payload["mode"] == "dry_run"
        assert dry_payload["planned_writes"] == []
        assert not (root / "audit.json").exists()

        bad_contract = root / "bad_contract.json"
        bad_contract.write_text(
            json.dumps(
                {
                    "dataset_contract": {
                        "dataset_id": "fixture_p5",
                        "required_split_roles": ["train", "valid_iid"],
                        "expected_total_sample_count": 4,
                    },
                    "audit_contract": {"mode": "read_only"},
                }
            )
        )
        bad = subprocess.run(
            [
                sys.executable,
                str(AUDIT_SCRIPT),
                "--dataset",
                str(dataset),
                "--split-map",
                str(split_map),
                "--contract",
                str(bad_contract),
                "--dry-run",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        assert bad.returncode == 2
        assert "expected_total_sample_count" in bad.stderr

        output_json = root / "audit.json"
        output_md = root / "audit.md"
        _run(
            [
                "--dataset",
                str(dataset),
                "--split-map",
                str(split_map),
                "--contract",
                str(contract),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
        )
        payload = json.loads(output_json.read_text())
        assert payload["dataset"]["sample_count"] == 3
        assert payload["read_only_guardrails"]["solver_calls"] == 0
        assert payload["split_summaries"]["train"]["sample_count"] == 2
        assert payload["split_summaries"]["train"]["effective_source_power"]["effective_source_power_W"]["mean"] == 3.0
        assert payload["split_summaries"]["train"]["control_volume_weights"]["k_field_width_counts"] == {"1": 1, "3": 1}
        duplicates = payload["duplicate_leakage"]
        assert duplicates["cross_role_model_input_duplicate_groups"]["group_count"] == 1
        assert duplicates["cross_role_full_sample_duplicate_groups"]["group_count"] == 1
        assert duplicates["cross_role_provenance_duplicate_groups"]["group_count"] == 1
        assert payload["audit_pass"] is False
        assert "Split Duplicate Leakage" in output_md.read_text()

    print("V5-P0 physics-scale audit fixture check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
