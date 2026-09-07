#!/usr/bin/env python3
"""Recover the frozen multi-HTC cases and generate train/valid labels only.

The official frozen case-table SHA covers all deterministic case parameters,
including the sealed role assignment.  This program verifies that historical
SHA in memory, but it writes labels and case rows only for train and valid.
No test temperature, prediction, or accuracy artifact is accepted or opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np


K = 0.2
AMBIENT_U = 0.2
SOURCE_START = 0.25
SOURCE_END = 0.30
DOMAIN_Z = 0.55
GRID_SHAPE = (51, 51, 51)
FROZEN_CANONICAL_ROWS_SHA256 = "5ac5c716aac8d283eec23d9ea9157f7a2a0614af15670f3150a2d113301e4253"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_rows() -> list[dict[str, Any]]:
    axis = [round(0.1 + index * 0.2 / 31, 4) for index in range(32)]
    rows: list[dict[str, Any]] = []
    for position in range(1024):
        lattice_index = (405 * position + 97) % 1024
        top_beta = axis[lattice_index % 32]
        bottom_beta = axis[lattice_index // 32]
        role = "train" if position < 768 else ("valid" if position < 896 else "test")
        rows.append(
            {
                # Git history of the superseded v1 manifest fixes this exact
                # serialization: case_id is the integer table position.
                "case_id": position,
                "role": role,
                "top_beta": top_beta,
                "bottom_beta": bottom_beta,
                "top_h": K / top_beta,
                "bottom_h": K / bottom_beta,
            }
        )
    return rows


def canonical_rows_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def analytical_coefficients(beta_top: float, beta_bottom: float) -> np.ndarray:
    za, zb, length = SOURCE_START, SOURCE_END, DOMAIN_Z
    matrix = np.zeros((6, 6), dtype=np.float64)
    rhs = np.zeros(6, dtype=np.float64)
    matrix[0] = [za, 1, -za, -1, 0, 0]; rhs[0] = -(za**2) / (2 * K)
    matrix[1] = [1, 0, -1, 0, 0, 0]; rhs[1] = -za / K
    matrix[2] = [0, 0, zb, 1, -zb, -1]; rhs[2] = (zb**2) / (2 * K)
    matrix[3] = [0, 0, 1, 0, -1, 0]; rhs[3] = zb / K
    matrix[4] = [-beta_bottom, 1, 0, 0, 0, 0]; rhs[4] = AMBIENT_U
    matrix[5] = [0, 0, 0, 0, length + beta_top, 1]; rhs[5] = AMBIENT_U
    return np.linalg.solve(matrix, rhs)


def analytical_delta_t_z(beta_top: float, beta_bottom: float) -> np.ndarray:
    a1, b1, a2, b2, a3, b3 = analytical_coefficients(beta_top, beta_bottom)
    z = np.linspace(0.0, DOMAIN_Z, GRID_SHAPE[2], dtype=np.float64)
    u = np.where(
        z < SOURCE_START,
        a1 * z + b1,
        np.where(z <= SOURCE_END, -(z**2) / (2 * K) + a2 * z + b2, a3 * z + b3),
    )
    return np.asarray(25.0 * (u - AMBIENT_U), dtype=np.float32)


def common_inputs() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, GRID_SHAPE[0])
    y = np.linspace(0.0, 1.0, GRID_SHAPE[1])
    z = np.linspace(0.0, DOMAIN_Z, GRID_SHAPE[2])
    coords = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    xx, yy, zz = coords.T
    top = np.isclose(zz, DOMAIN_Z)
    bottom = np.isclose(zz, 0.0)
    lateral = np.isclose(xx, 0.0) | np.isclose(xx, 1.0) | np.isclose(yy, 0.0) | np.isclose(yy, 1.0)
    side = lateral & ~(top | bottom)
    interior = ~(top | bottom | side)
    features = np.zeros((len(coords), 11), dtype=np.float32)
    features[:, :3] = K
    features[(zz >= SOURCE_START - 1e-12) & (zz <= SOURCE_END + 1e-12), 3] = 25.0
    features[:, 4:8] = np.column_stack((top, bottom, side, interior))
    # Columns 8 and 9 are filled losslessly from each case's h=K/beta.
    # Column 10 remains zero because both Robin ambients equal T_ref.
    return coords, features


def atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = canonical_rows()
    recovered_sha = canonical_rows_sha256(rows)
    if recovered_sha != FROZEN_CANONICAL_ROWS_SHA256:
        raise RuntimeError(f"FAIL-CLOSED canonical row SHA mismatch: {recovered_sha}")
    selected = {role: [row for row in rows if row["role"] == role] for role in ("train", "valid")}
    if (len(selected["train"]), len(selected["valid"])) != (768, 128):
        raise RuntimeError("FAIL-CLOSED role-count mismatch")

    coords, feature_template = common_inputs()
    atomic_save(output / "common_coords.npy", coords)
    atomic_save(output / "common_feature_template.npy", feature_template)
    expanded_rows: list[dict[str, Any]] = []
    artifact_rows: dict[str, Any] = {}
    for role in ("train", "valid"):
        role_rows = selected[role]
        parameters = np.asarray(
            [[row["top_beta"], row["bottom_beta"], row["top_h"], row["bottom_h"]] for row in role_rows],
            dtype=np.float64,
        )
        targets = np.stack(
            [analytical_delta_t_z(row["top_beta"], row["bottom_beta"]) for row in role_rows]
        )
        parameters_path = output / f"{role}_case_parameters.npy"
        targets_path = output / f"{role}_deltaT_z_K.npy"
        rows_path = output / f"{role}_case_rows.json"
        atomic_save(parameters_path, parameters)
        atomic_save(targets_path, targets)
        rows_path.write_text(json.dumps(role_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact_rows[role] = {
            "count": len(role_rows),
            "case_parameters": {"file": parameters_path.name, "file_sha256": file_sha256(parameters_path), "array_sha256": canonical_array_sha256(parameters)},
            "deltaT_z_K": {"file": targets_path.name, "file_sha256": file_sha256(targets_path), "array_sha256": canonical_array_sha256(targets), "lossless_full_field_expansion": "broadcast z profile over 51x51"},
            "case_rows": {"file": rows_path.name, "file_sha256": file_sha256(rows_path)},
        }
        for row, profile in zip(role_rows, targets, strict=True):
            full_field = np.broadcast_to(profile, GRID_SHAPE)
            expanded_rows.append(
                {
                    "case_id": row["case_id"],
                    "role": role,
                    "case_input_sha256": hashlib.sha256(canonical_json_bytes(row)).hexdigest(),
                    "deltaT_z_K_sha256": canonical_array_sha256(profile),
                    "deltaT_full_51x51x51_K_sha256": canonical_array_sha256(full_field),
                }
            )

    expanded_path = output / "expanded_train_valid_receipt.json"
    expanded_path.write_text(json.dumps(expanded_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    common_artifacts = {}
    for name in ("common_coords.npy", "common_feature_template.npy"):
        path = output / name
        common_artifacts[name] = {"file_sha256": file_sha256(path), "array_sha256": canonical_array_sha256(np.load(path, allow_pickle=False))}
    receipt = {
        "schema_version": "heat3d_v7_g2_p7_multi_htc_train_valid_labels_v1",
        "status": "PASS_PROVENANCE_REPAIRED_TRAIN768_VALID128_LABELS_GENERATED",
        "historical_recovery": {
            "source": "Git history superseded v1 case manifest plus deterministic v2 beta correction",
            "case_id_serialization": "integer position 0..1023",
            "frozen_canonical_rows_sha256": FROZEN_CANONICAL_ROWS_SHA256,
            "recovered_canonical_rows_sha256": recovered_sha,
            "exact_match": True,
        },
        "physics": {"beta_range": [0.1, 0.3], "conductivity": K, "h_conversion": "h=0.2/beta", "grid": list(GRID_SHAPE), "target": "deltaT_K=25*(u-0.2)"},
        "labels": {"train": 768, "valid": 128, "test": 0, "representation": "lossless 51-point z profile broadcast over the uniform 51x51 xy plane", "artifacts": artifact_rows},
        "common_artifacts": common_artifacts,
        "expanded_train_valid_receipt_sha256": file_sha256(expanded_path),
        "generator_sha256": file_sha256(Path(__file__)),
        "solver": "piecewise analytical closed form qualified by P5 against the independent 51x51x51 deterministic solver",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "device": "CPU"},
        "isolation": {"test_case_rows_written": 0, "test_labels_generated": 0, "test_temperature_prediction_or_accuracy_read": False},
    }
    receipt_path = output / "label_generation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
