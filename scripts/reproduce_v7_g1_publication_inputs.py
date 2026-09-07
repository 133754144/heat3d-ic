#!/usr/bin/env python3
"""Reproduce the small publication-input fixtures from frozen inputs only.

This recipe is deliberately narrower than a training or evaluation loader.  It
reads only the shared geometry and explicitly named valid_iid ``deltaT_K`` rows
from the frozen full-field H5, plus the six geometry/support arrays and the
allowlisted layout metadata for one frozen 1024-point support sample.  It never
opens checkpoint, prediction, temperature, or sealed/test data.

The output NPZ key order and JSON formatting are stable so an independent run
can compare byte-level SHA256 values with the recorded fixture bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


EXPECTED_SAMPLE_IDS = (
    "v6p1if1_0533",
    "v6p1if1_0203",
    "v6p1if1_0563",
)
EXPECTED_FULL_FIELD_SHA256 = "49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb"
EXPECTED_TRUTH_SHA256 = "74415add51bd92b8129d8d30d4740ed695701c3a987956470b1b56e6707f60ad"
EXPECTED_SUPPORT_SHA256 = "7c595f7e147cdcae678dbda4d961fc2ce23c131c7eb895083ec4d3cc6aa22944"
EXPECTED_META_SHA256 = "dce35ad47963c05337effde229338c4fa0d66c20ddc8892b98778620a7242453"

META_KEYS = (
    "bottom_h_W_m2K",
    "dataset_id",
    "group_id",
    "k_block_values_W_mK",
    "k_blocks",
    "package_total_power_W",
    "q_block_power_fractions",
    "q_blocks",
    "sample_id",
    "top_h_W_m2K",
)
PHYSICS_KEYS = ("ambient_K", "footprint_m", "layers_bottom_to_top")
BANNED_KEY_TOKENS = ("accuracy", "metric", "rmse", "mae", "temperature", "prediction", "target", "loss")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_strings(values: Any) -> list[str]:
    result: list[str] = []
    for value in np.asarray(values).reshape(-1).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value))
    return result


def _assert_no_banned_keys(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in BANNED_KEY_TOKENS):
                raise ValueError(f"banned output key at {path}: {key}")
            _assert_no_banned_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_banned_keys(child, f"{path}[{index}]")


def sanitize_support_meta(source: Path, support_sample_id: str) -> dict[str, Any]:
    raw = json.loads(source.read_text(encoding="utf-8"))
    if str(raw.get("sample_id")) != support_sample_id:
        raise ValueError("support metadata sample_id does not match the selected support sample")
    if str(raw.get("split_role")) != "valid_iid":
        raise ValueError("support metadata is not from valid_iid")
    missing = [key for key in META_KEYS if key not in raw]
    if missing:
        raise ValueError(f"support metadata missing allowlisted keys: {missing}")
    physics = raw.get("physics") or {}
    if any(key not in physics for key in PHYSICS_KEYS):
        raise ValueError("support metadata physics stack is incomplete")
    result = {key: raw[key] for key in META_KEYS}
    result["physics"] = {key: physics[key] for key in PHYSICS_KEYS}
    _assert_no_banned_keys(result)
    return result


def load_support_arrays(sample_dir: Path) -> dict[str, np.ndarray]:
    filenames = {
        "coords_1024": "coords.npy",
        "control_volume_1024": "control_volume.npy",
        "layer_id_1024": "layer_id.npy",
        "bc_features_1024": "bc_features.npy",
        "k_field_1024": "k_field.npy",
        "q_field_1024": "q_field.npy",
    }
    arrays: dict[str, np.ndarray] = {}
    for key, filename in filenames.items():
        path = sample_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        arrays[key] = np.load(path, allow_pickle=False)
    expected_shapes = {
        "coords_1024": (1024, 3),
        "control_volume_1024": (1024,),
        "layer_id_1024": (1024,),
        "bc_features_1024": (1024, 7),
        "k_field_1024": (1024, 3),
        "q_field_1024": (1024, 1),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape or not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"invalid support array {key}: {arrays[key].shape}")
    return arrays


def extract_truth(full_field_h5: Path, sample_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for H5 fixture reproduction") from exc

    with h5py.File(full_field_h5.resolve(), "r") as handle:
        allowed_shared = {"coords_m", "control_volume_m3", "layer_id", "boundary_flags"}
        allowed_samples = {"sample_id", "split_role", "deltaT_K"}
        if set(handle["shared"].keys()) < allowed_shared:
            raise ValueError("full-field shared geometry is incomplete")
        if set(handle["samples"].keys()) < allowed_samples:
            raise ValueError("full-field valid_iid source is incomplete")
        coords = np.asarray(handle["shared"]["coords_m"][:], dtype=np.float64)
        control_volume = np.asarray(handle["shared"]["control_volume_m3"][:], dtype=np.float64)
        layer_id = np.asarray(handle["shared"]["layer_id"][:], dtype=np.int32)
        boundary_flags = np.asarray(handle["shared"]["boundary_flags"][:], dtype=np.float64)
        all_ids = _decode_strings(handle["samples"]["sample_id"][:])
        all_roles = _decode_strings(handle["samples"]["split_role"][:])
        indices: list[int] = []
        for sample_id in sample_ids:
            matches = [index for index, value in enumerate(all_ids) if value == sample_id]
            if len(matches) != 1:
                raise ValueError(f"sample ID lookup is not unique: {sample_id}")
            index = matches[0]
            if all_roles[index] != "valid_iid":
                raise ValueError(f"sample is not valid_iid: {sample_id}")
            indices.append(index)
        # h5py requires fancy indices in increasing order; restore the
        # preregistered sample-id order after the geometry-only read.
        ordered = sorted(enumerate(indices), key=lambda item: item[1])
        sorted_indices = [item[1] for item in ordered]
        delta_sorted = np.asarray(handle["samples"]["deltaT_K"][sorted_indices, :], dtype=np.float32)
        delta = np.empty_like(delta_sorted)
        for sorted_row, (requested_row, _) in enumerate(ordered):
            delta[requested_row] = delta_sorted[sorted_row]

    if coords.shape != (240825, 3) or control_volume.shape != (240825,):
        raise ValueError("unexpected shared full-field shape")
    if layer_id.shape != (240825,) or boundary_flags.shape != (240825, 4):
        raise ValueError("unexpected shared full-field auxiliary shape")
    if delta.shape != (len(sample_ids), 240825) or not np.all(np.isfinite(delta)):
        raise ValueError("unexpected valid_iid deltaT shape")
    return {
        "sample_ids": np.asarray(sample_ids, dtype="<U12"),
        "coords": coords,
        "control_volume": control_volume,
        "layer_id": layer_id,
        "boundary_flags": boundary_flags,
        "truth_deltaT_K": delta,
    }


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-field-h5", type=Path, required=True)
    parser.add_argument("--support-sample-dir", type=Path, required=True)
    parser.add_argument("--support-meta", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-sample-id", default="v6p1if1_0533")
    parser.add_argument("--sample-ids", nargs=3, default=list(EXPECTED_SAMPLE_IDS))
    parser.add_argument("--expected-full-field-sha256", default=EXPECTED_FULL_FIELD_SHA256)
    parser.add_argument("--expected-truth-sha256", default=EXPECTED_TRUTH_SHA256)
    parser.add_argument("--expected-support-sha256", default=EXPECTED_SUPPORT_SHA256)
    parser.add_argument("--expected-meta-sha256", default=EXPECTED_META_SHA256)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    full_field_h5 = args.full_field_h5.resolve()
    support_dir = args.support_sample_dir.resolve()
    support_meta_path = args.support_meta.resolve()
    config_path = args.config_path.resolve()
    output_dir = args.output_dir.resolve()
    if len(set(args.sample_ids)) != 3:
        raise ValueError("exactly three unique valid_iid sample IDs are required")
    for path in (full_field_h5, support_dir, support_meta_path, config_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    full_field_sha256 = sha256_file(full_field_h5)
    if full_field_sha256 != args.expected_full_field_sha256:
        raise ValueError(f"full-field H5 SHA mismatch: {full_field_sha256}")
    truth = extract_truth(full_field_h5, tuple(args.sample_ids))
    support = load_support_arrays(support_dir)
    support_meta = sanitize_support_meta(support_meta_path, args.support_sample_id)

    truth_path = output_dir / "truth_fixture.npz"
    support_path = output_dir / "support_fixture.npz"
    meta_path = output_dir / "support_metadata.json"
    _save_npz(truth_path, truth)
    _save_npz(support_path, support)
    meta_path.write_text(json.dumps(support_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outputs = {
        "truth_fixture.npz": {"sha256": sha256_file(truth_path), "bytes": truth_path.stat().st_size},
        "support_fixture.npz": {"sha256": sha256_file(support_path), "bytes": support_path.stat().st_size},
        "support_metadata.json": {"sha256": sha256_file(meta_path), "bytes": meta_path.stat().st_size},
    }
    expected = {
        "truth_fixture.npz": args.expected_truth_sha256,
        "support_fixture.npz": args.expected_support_sha256,
        "support_metadata.json": args.expected_meta_sha256,
    }
    mismatches = {name: row["sha256"] for name, row in outputs.items() if row["sha256"] != expected[name]}
    if mismatches:
        raise ValueError(f"reproduced fixture SHA mismatch; artifacts were not silently accepted: {mismatches}")

    source = {
        "full_field_h5": {"path": str(full_field_h5), "sha256": full_field_sha256},
        "selected_valid_iid_ids": list(args.sample_ids),
        "support_sample": {
            "sample_id": args.support_sample_id,
            "directory": str(support_dir),
            "file_sha256": {filename: sha256_file(support_dir / filename) for filename in (
                "coords.npy", "control_volume.npy", "layer_id.npy", "bc_features.npy", "k_field.npy", "q_field.npy", "sample_meta.json"
            )},
        },
        "frozen_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "read_policy": [
            "shared geometry datasets only",
            "samples/sample_id and samples/split_role",
            "samples/deltaT_K rows for the three selected valid_iid IDs",
            "six named 1024 support arrays and allowlisted support metadata",
        ],
        "forbidden": ["checkpoint", "model forward", "training", "prediction", "temperature dataset", "test_iid", "sealed"],
    }
    manifest = {
        "schema_version": "heat3d_v7_g1_publication_fixture_reproduction_v1",
        "status": "REPRODUCED_AND_BYTE_VERIFIED",
        "source": source,
        "outputs": outputs,
        "expected_sha256": expected,
        "reproduction": {
            "numpy": np.__version__,
            "truth_npz_key_order": list(truth),
            "support_npz_key_order": list(support),
            "metadata_allowlist": list(META_KEYS) + ["physics." + key for key in PHYSICS_KEYS],
            "no_new_inference": True,
            "test_iid_access": False,
            "sealed_access": False,
        },
    }
    _assert_no_banned_keys(manifest)
    manifest_path = output_dir / "reproduction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "outputs": outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
