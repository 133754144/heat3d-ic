#!/usr/bin/env python3
"""Build the P23 valid-only case manifest from frozen metadata.

Only split metadata, shared geometry, and per-case physics definitions are
read.  No temperature/target dataset is opened.  The script deliberately
does not accept a mixed-role CSV as input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def masked_region(coords: np.ndarray, layer_id: np.ndarray, layer_names: tuple[str, ...], footprint: tuple[float, float], block: dict[str, Any]) -> np.ndarray:
    x0, x1, y0, y1 = [float(v) for v in block["bbox_fraction_xy"]]
    layer = layer_names.index(str(block["layer"]))
    return (
        (layer_id == layer)
        & (coords[:, 0] / footprint[0] >= x0)
        & (coords[:, 0] / footprint[0] <= x1)
        & (coords[:, 1] / footprint[1] >= y0)
        & (coords[:, 1] / footprint[1] <= y1)
    )


def material_log10_weighted_sd(
    coords: np.ndarray,
    layer_id: np.ndarray,
    control_volume: np.ndarray,
    meta: dict[str, Any],
) -> float:
    layers = meta["physics"]["layers_bottom_to_top"]
    layer_names = tuple(str(row["id"]) for row in layers)
    footprint = tuple(float(v) for v in meta["physics"]["footprint_m"])
    background = np.asarray(
        [row["background_k_xyz_W_mK"] for row in layers], dtype=np.float64
    )
    k_field = background[layer_id].copy()
    for block, value in zip(meta["k_blocks"], meta["k_block_values_W_mK"], strict=True):
        mask = masked_region(coords, layer_id, layer_names, footprint, block)
        k_field[mask, :] = float(value)
    if not np.isfinite(k_field).all() or np.any(k_field <= 0.0):
        raise ValueError("non-positive or non-finite conductivity in valid case")
    log_gmean = np.log10(np.cbrt(np.prod(k_field, axis=1)))
    weight = np.asarray(control_volume, dtype=np.float64)
    mean = float(np.sum(weight * log_gmean) / np.sum(weight))
    variance = float(np.sum(weight * np.square(log_gmean - mean)) / np.sum(weight))
    return float(np.sqrt(max(variance, 0.0)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--full-field-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {args.output}")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    assignment = {str(k): str(v) for k, v in split["assignment"].items()}
    valid_ids = sorted(sid for sid, role in assignment.items() if role == "valid_iid")
    if len(valid_ids) != 128:
        raise ValueError(f"expected 128 valid_iid IDs, got {len(valid_ids)}")
    if any("test" in sid.lower() or "sealed" in sid.lower() for sid in valid_ids):
        raise ValueError("valid-only manifest contains a forbidden-role-looking ID")

    with h5py.File(args.full_field_archive, "r") as archive:
        coords = np.asarray(archive["shared/coords_m"][:], dtype=np.float64)
        control_volume = np.asarray(archive["shared/control_volume_m3"][:], dtype=np.float64)
        layer_id = np.asarray(archive["shared/layer_id"][:], dtype=np.int32)
        h5_ids = [decode(v) for v in archive["samples/sample_id"][:]]
        h5_roles = [decode(v) for v in archive["samples/split_role"][:]]
    if len(set(h5_ids)) != len(h5_ids):
        raise ValueError("duplicate IDs in full-field metadata")
    row_by_id = {sid: i for i, sid in enumerate(h5_ids)}
    cases: list[dict[str, Any]] = []
    for sid in valid_ids:
        if sid not in row_by_id:
            raise ValueError(f"valid ID missing from full-field metadata: {sid}")
        truth_row = int(row_by_id[sid])
        if h5_roles[truth_row] != "valid_iid":
            raise ValueError(f"role mismatch for valid ID {sid}")
        meta_path = args.sample_root / sid / "sample_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("sample_id")) != sid or str(meta.get("split_role")) != "valid_iid":
            raise ValueError(f"sample metadata mismatch for {sid}")
        total_power = float(meta["package_total_power_W"])
        source_powers = total_power * np.asarray(meta["q_block_power_fractions"], dtype=np.float64)
        if source_powers.size == 0 or np.any(source_powers <= 0.0):
            raise ValueError(f"invalid source powers for {sid}")
        source_mean = float(np.mean(source_powers))
        source_cv = float(np.std(source_powers, ddof=0) / source_mean)
        cases.append(
            {
                "sample_id": sid,
                "truth_row": truth_row,
                "split_role": "valid_iid",
                "source_count": int(len(meta["q_blocks"])),
                "total_power_W": total_power,
                "source_power_heterogeneity": source_cv,
                "top_h_W_m2K": float(meta["top_h_W_m2K"]),
                "bottom_h_W_m2K": float(meta["bottom_h_W_m2K"]),
                "k_region_count": int(len(meta["k_blocks"])),
                "material_conductivity_heterogeneity": material_log10_weighted_sd(
                    coords, layer_id, control_volume, meta
                ),
                "sample_meta_sha256": sha256_file(meta_path)
            }
        )
    payload = {
        "schema_version": "heat3d_v7_g2_p23_valid_only_case_manifest_v1",
        "status": "FROZEN_VALID_ONLY_INPUT_METADATA",
        "dataset_id": "heat3d_v6_p1i_continuous_physics1024_v1",
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "full_field_metadata_source": str(args.full_field_archive),
        "full_field_metadata_sha256": sha256_file(args.full_field_archive),
        "sample_root": str(args.sample_root),
        "role": "valid_iid",
        "case_count": len(cases),
        "valid_ids_sha256": sha256_ids(valid_ids),
        "target_or_prediction_read": False,
        "test_iid_read": False,
        "sealed_read": False,
        "deepoheat_official100_read": False,
        "condition_definitions": {
            "source_power_heterogeneity": "population CV = std(P_source, ddof=0) / mean(P_source)",
            "material_conductivity_heterogeneity": "control-volume-weighted population SD of log10((kx*ky*kz)^(1/3)); conductivity reconstructed from case-definition background and k blocks"
        },
        "cases": cases
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "case_count": len(cases), "valid_ids_sha256": payload["valid_ids_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
