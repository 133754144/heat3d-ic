#!/usr/bin/env python3
"""Fail-closed negative tests for the P1i domain-identity gate.

The tests mutate only temporary metadata/coordinate fixtures.  They never
modify the frozen truth archive, predictions, checkpoints, or split files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


NODE_COUNT = 65 * 65 * 57


def sha256_array(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def validate_domain(
    cases: list[dict[str, Any]],
    truth_archive: Path,
    prediction_spec: dict[str, Any],
    expected_coordinate_sha256: str,
    coordinate_override: np.ndarray | None = None,
) -> None:
    if len(cases) != 128:
        raise ValueError("valid case count mismatch")
    if any(case.get("split_role") != "valid_iid" for case in cases):
        raise ValueError("non-valid split role")
    with h5py.File(truth_archive, "r") as archive:
        truth_ids = archive["samples/sample_id"]
        truth_roles = archive["samples/split_role"]
        coordinates = (
            np.asarray(coordinate_override)
            if coordinate_override is not None
            else np.asarray(archive["shared/coords_m"][:])
        )
        if sha256_array(coordinates) != expected_coordinate_sha256:
            raise ValueError("coordinates_sha256_mismatch")
        ids = {str(case["sample_id"]) for case in cases}
        if len(ids) != 128:
            raise ValueError("duplicate sample IDs")
        for case in cases:
            sid = str(case["sample_id"])
            row = int(case["truth_row"])
            if decode(truth_ids[row]) != sid:
                raise ValueError(f"truth_row_mapping_mismatch:{sid}")
            if decode(truth_roles[row]) != "valid_iid":
                raise ValueError(f"truth_role_mismatch:{sid}")
        for label, spec in prediction_spec.items():
            path = Path(spec["path"])
            with np.load(path, allow_pickle=False) as archive_npz:
                if set(archive_npz.files) != ids:
                    raise ValueError(f"prediction_sample_id_mismatch:{label}")
                for sid in ids:
                    value = np.asarray(archive_npz[sid])
                    if value.size != NODE_COUNT:
                        raise ValueError(f"prediction_node_count_mismatch:{label}:{sid}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-cases", type=Path, required=True)
    parser.add_argument("--truth-archive", type=Path, required=True)
    parser.add_argument("--prediction-spec", type=Path, required=True)
    parser.add_argument("--expected-coordinate-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.valid_cases)
    spec = json.loads(args.prediction_spec.read_text(encoding="utf-8"))
    with h5py.File(args.truth_archive, "r") as archive:
        canonical_coords = np.asarray(archive["shared/coords_m"][:])

    tests: list[tuple[str, list[dict[str, Any]], np.ndarray | None]] = []
    mutated = copy.deepcopy(cases)
    mutated[0]["sample_id"] = "invalid_sample_id_not_in_prediction_archive"
    tests.append(("sample_id_mutation", mutated, None))
    reordered = canonical_coords[::-1].copy()
    tests.append(("coordinate_order_mutation", cases, reordered))
    remapped = copy.deepcopy(cases)
    remapped[0]["truth_row"] = int(cases[1]["truth_row"])
    tests.append(("truth_row_mapping_mutation", remapped, None))

    results = []
    for name, fixture_cases, coords in tests:
        try:
            validate_domain(
                fixture_cases,
                args.truth_archive,
                spec,
                args.expected_coordinate_sha256,
                coordinate_override=coords,
            )
        except Exception as exc:  # expected fail-closed path
            results.append({"name": name, "status": "EXPECTED_FAIL", "error": str(exc)})
        else:
            results.append({"name": name, "status": "UNEXPECTED_PASS", "error": None})
    payload = {
        "schema_version": "heat3d_v7_g2_p1i_negative_domain_gate_v1",
        "status": "PASS_FAIL_CLOSED_NEGATIVE_TESTS" if all(r["status"] == "EXPECTED_FAIL" for r in results) else "FAIL_NEGATIVE_GATE",
        "baseline_contract": "valid_iid=128, 240825 nodes, coordinate hash and truth-row identity",
        "tests": results,
        "test_iid_read": False,
        "sealed_read": False,
        "deepoheat_official100_read": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if payload["status"] == "PASS_FAIL_CLOSED_NEGATIVE_TESTS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
