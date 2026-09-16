#!/usr/bin/env python3
"""Split-preserving native 65x65 Therm-FM adapter for V7 G2 P22.

This module deliberately consumes only the frozen P1i train/valid roles.  It
constructs Therm-FM's dense channels directly from per-case physics metadata;
no sparse-point interpolation, learned adapter, or test/sealed target read is
performed.  The upstream ScOT model performs its own patch/window padding when
configured with ``image_size=65`` and crops recovery output back to 65x65.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


P_PER_LAYER = 13
GRID = (65, 65, 57)
CHANNEL_ORDER = (
    "x_norm",
    "y_norm",
    "z_norm",
    "kx",
    "ky",
    "kz",
    "q",
    "boundary_front",
    "boundary_back",
    "boundary_left",
    "boundary_right",
    "h_top",
    "h_bottom",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: Iterable[str]) -> str:
    payload = ("\n".join(ids) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decode_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


class P1iThermFMSource:
    """Frozen P1i source and deterministic case-definition rasterizer."""

    def __init__(
        self,
        split_manifest: Path,
        sample_root: Path,
        full_field_archive: Path,
        roles: tuple[str, ...] = ("train", "valid_iid"),
    ) -> None:
        self.split_manifest_path = split_manifest
        self.sample_root = sample_root
        self.full_field_archive = full_field_archive
        split = json.loads(split_manifest.read_text())
        assignment = split["assignment"]
        self.all_assignment = {str(k): str(v) for k, v in assignment.items()}
        self.ids_by_role = {
            role: tuple(sorted(sid for sid, assigned in self.all_assignment.items() if assigned == role))
            for role in roles
        }
        self.ids = tuple(sid for role in roles for sid in self.ids_by_role[role])
        if len(set(self.ids)) != len(self.ids):
            raise RuntimeError("duplicate IDs in authorized split roles")
        self._h5: h5py.File | None = None

        with h5py.File(full_field_archive, "r") as handle:
            coords = np.asarray(handle["shared/coords_m"], dtype=np.float64)
            self.control_volume = np.asarray(handle["shared/control_volume_m3"], dtype=np.float64)
            self.layer_id = np.asarray(handle["shared/layer_id"], dtype=np.int32)
            self.boundary_flags = np.asarray(handle["shared/boundary_flags"], dtype=np.float64)
            h5_ids = tuple(_decode_string(v) for v in handle["samples/sample_id"][:])
            h5_roles = tuple(_decode_string(v) for v in handle["samples/split_role"][:])
        if coords.shape != (GRID[0] * GRID[1] * GRID[2], 3):
            raise RuntimeError(f"unexpected shared coordinate shape {coords.shape}")
        x = np.unique(coords[:, 0])
        y = np.unique(coords[:, 1])
        z = np.unique(coords[:, 2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        expected = np.column_stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)))
        if not np.array_equal(expected, coords):
            raise RuntimeError("P1i coordinate ordering is not C-order x,y,z")
        if (len(x), len(y), len(z)) != GRID:
            raise RuntimeError(f"unexpected P1i grid {(len(x), len(y), len(z))}")
        if len(set(h5_ids)) != len(h5_ids):
            raise RuntimeError("duplicate sample IDs in full-field archive")
        row_by_id = {sid: i for i, sid in enumerate(h5_ids)}
        self.row_by_id: dict[str, int] = {}
        for sid in self.ids:
            if sid not in row_by_id:
                raise RuntimeError(f"missing authorized sample {sid} in full-field archive")
            row = row_by_id[sid]
            expected_role = self.all_assignment[sid]
            if h5_roles[row] != expected_role:
                raise RuntimeError(f"split role mismatch for {sid}: {h5_roles[row]} != {expected_role}")
            self.row_by_id[sid] = row

        self.coords = coords
        self.x, self.y, self.z = x, y, z
        self.layer_names: tuple[str, ...] | None = None
        self.footprint: tuple[float, float] | None = None

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        self.close()

    def _open_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.full_field_archive, "r")
        return self._h5

    def _metadata(self, sid: str) -> dict:
        path = self.sample_root / sid / "sample_meta.json"
        meta = json.loads(path.read_text())
        if str(meta.get("sample_id")) != sid:
            raise RuntimeError(f"sample metadata ID mismatch for {sid}")
        if str(meta.get("split_role")) != self.all_assignment[sid]:
            raise RuntimeError(f"sample metadata split mismatch for {sid}")
        return meta

    def _ensure_geometry_semantics(self, meta: dict) -> None:
        if self.layer_names is None:
            layers = meta["physics"]["layers_bottom_to_top"]
            self.layer_names = tuple(str(row["id"]) for row in layers)
            footprint = meta["physics"]["footprint_m"]
            self.footprint = (float(footprint[0]), float(footprint[1]))

    def raw_input(self, sid: str) -> np.ndarray:
        meta = self._metadata(sid)
        self._ensure_geometry_semantics(meta)
        assert self.layer_names is not None and self.footprint is not None
        coords = self.coords
        layer_id = self.layer_id
        background = np.asarray(
            [row["background_k_xyz_W_mK"] for row in meta["physics"]["layers_bottom_to_top"]],
            dtype=np.float64,
        )
        k_field = background[layer_id].copy()

        def mask_for(block: dict) -> np.ndarray:
            x0, x1, y0, y1 = [float(v) for v in block["bbox_fraction_xy"]]
            layer = self.layer_names.index(str(block["layer"]))
            return (
                (layer_id == layer)
                & (coords[:, 0] / self.footprint[0] >= x0)
                & (coords[:, 0] / self.footprint[0] <= x1)
                & (coords[:, 1] / self.footprint[1] >= y0)
                & (coords[:, 1] / self.footprint[1] <= y1)
            )

        for block, value in zip(meta["k_blocks"], meta["k_block_values_W_mK"], strict=True):
            k_field[mask_for(block), :] = float(value)

        q_field = np.zeros(len(coords), dtype=np.float64)
        total_power = float(meta["package_total_power_W"])
        for block, fraction in zip(meta["q_blocks"], meta["q_block_power_fractions"], strict=True):
            mask = mask_for(block)
            if not np.any(mask) or np.any(q_field[mask] != 0.0):
                raise RuntimeError(f"invalid or overlapping q block for {sid}")
            q_field[mask] = total_power * float(fraction) / float(np.sum(self.control_volume[mask]))
        if not np.isclose(
            np.sum(q_field * self.control_volume),
            total_power,
            rtol=2e-12,
            atol=1e-12,
        ):
            raise RuntimeError(f"q conservation failed for {sid}")

        norm_coords = coords / np.asarray(
            [self.footprint[0], self.footprint[1], float(self.z[-1])], dtype=np.float64
        )
        node_channels = np.column_stack(
            (
                norm_coords,
                k_field,
                q_field[:, None],
                self.boundary_flags,
                np.full((len(coords), 1), float(meta["top_h_W_m2K"])),
                np.full((len(coords), 1), float(meta["bottom_h_W_m2K"])),
            )
        )
        if node_channels.shape != (len(coords), P_PER_LAYER):
            raise RuntimeError(f"unexpected channel shape {node_channels.shape}")
        # Shared geometry is C-order x,y,z.  Convert to (layer, channel, x, y)
        # and flatten layer-major exactly as the upstream thermal loader does.
        p_l_xy = node_channels.reshape(GRID[0], GRID[1], GRID[2], P_PER_LAYER).transpose(3, 2, 0, 1)
        return p_l_xy.transpose(1, 0, 2, 3).reshape(GRID[2] * P_PER_LAYER, GRID[0], GRID[1]).astype(
            np.float32,
            copy=False,
        )

    def raw_target(self, sid: str) -> np.ndarray:
        row = self.row_by_id[sid]
        # This access is intentionally limited to train/valid rows selected by
        # the frozen assignment; test_iid/sealed rows are never indexed.
        # Reuse the read-only archive handle so a qualification pass over the
        # frozen train/valid split does not repeatedly reopen a 0.9 GB file.
        handle = self._open_h5()
        flat = np.asarray(handle["samples/deltaT_K"][row], dtype=np.float32)
        return flat.reshape(GRID[0], GRID[1], GRID[2]).transpose(2, 0, 1)


class P1iThermFMDataset(Dataset):
    def __init__(self, source: P1iThermFMSource, ids: tuple[str, ...], stats_path: Path):
        self.source = source
        self.ids = ids
        stats = json.loads(stats_path.read_text())
        self.input_mean = np.asarray(stats["input"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
        self.input_std = np.asarray(stats["input"]["std"], dtype=np.float32).reshape(-1, 1, 1)
        self.output_mean = np.asarray(stats["output"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
        self.output_std = np.asarray(stats["output"]["std"], dtype=np.float32).reshape(-1, 1, 1)
        if self.input_mean.shape[0] != GRID[2] * P_PER_LAYER or self.output_mean.shape[0] != GRID[2]:
            raise RuntimeError("normalization channel count does not match native 65x65 contract")
        self.input_std = np.where(self.input_std > 0.0, self.input_std, 1.0)
        self.output_std = np.where(self.output_std > 0.0, self.output_std, 1.0)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        sid = self.ids[index]
        inputs = (self.source.raw_input(sid) - self.input_mean) / self.input_std
        target = (self.source.raw_target(sid) - self.output_mean) / self.output_std
        return {
            "pixel_values": torch.from_numpy(np.ascontiguousarray(inputs)),
            "labels": torch.from_numpy(np.ascontiguousarray(target)),
            "sample_id": sid,
        }


def compute_train_stats(source: P1iThermFMSource, train_ids: tuple[str, ...], output_path: Path, receipt_path: Path) -> dict:
    if any(source.all_assignment[sid] != "train" for sid in train_ids):
        raise RuntimeError("normalization fit attempted with a non-train ID")
    input_sum = np.zeros(GRID[2] * P_PER_LAYER, dtype=np.float64)
    input_sq = np.zeros_like(input_sum)
    output_sum = np.zeros(GRID[2], dtype=np.float64)
    output_sq = np.zeros_like(output_sum)
    count = len(train_ids) * GRID[0] * GRID[1]
    for sid in train_ids:
        x = source.raw_input(sid).astype(np.float64, copy=False)
        y = source.raw_target(sid).astype(np.float64, copy=False)
        input_sum += x.sum(axis=(1, 2))
        input_sq += np.square(x).sum(axis=(1, 2))
        output_sum += y.sum(axis=(1, 2))
        output_sq += np.square(y).sum(axis=(1, 2))
    input_mean = input_sum / count
    output_mean = output_sum / count
    input_std = np.sqrt(np.maximum(input_sq / count - np.square(input_mean), 0.0))
    output_std = np.sqrt(np.maximum(output_sq / count - np.square(output_mean), 0.0))
    input_std = np.where(input_std > 0.0, input_std, 1.0)
    output_std = np.where(output_std > 0.0, output_std, 1.0)
    payload = {
        "schema_version": "heat3d_v7_g2_p22_thermfm_native65_train_only_normalization_v1",
        "fit_roles": ["train"],
        "fit_count": len(train_ids),
        "train_ids_sha256": sha256_ids(train_ids),
        "physical_grid": list(GRID),
        "physical_pixels_per_case": GRID[0] * GRID[1],
        "padding_pixels_used_for_fit": 0,
        "input_layout": "layer-major [z0:(13 channels), z1:(13 channels), ...]",
        "channel_order_per_layer": list(CHANNEL_ORDER),
        "input": {"mean": input_mean.tolist(), "std": input_std.tolist()},
        "output": {"mean": output_mean.tolist(), "std": output_std.tolist()},
        "target": "deltaT_K from samples/deltaT_K; train rows only",
        "test_iid_read": False,
        "sealed_read": False,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    receipt = {
        "schema_version": "heat3d_v7_g2_p22_thermfm_native65_normalization_receipt_v1",
        "status": "PASS_TRAIN_ONLY_STATS_NATIVE_65X65",
        "normalization_path": str(output_path),
        "normalization_sha256": sha256_file(output_path),
        "source_split_manifest": str(source.split_manifest_path),
        "source_split_manifest_sha256": sha256_file(source.split_manifest_path),
        "source_full_field_archive": str(source.full_field_archive),
        "source_full_field_archive_sha256": sha256_file(source.full_field_archive),
        "train_count": len(train_ids),
        "valid_count": len(source.ids_by_role.get("valid_iid", ())),
        "train_ids_sha256": sha256_ids(train_ids),
        "valid_ids_sha256": sha256_ids(source.ids_by_role.get("valid_iid", ())),
        "physical_grid": list(GRID),
        "padding_pixels_used_for_fit": 0,
        "target_source": "deltaT_K",
        "test_iid_read": False,
        "sealed_read": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stats",), required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--full-field-archive", type=Path, required=True)
    parser.add_argument("--stats-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    args = parser.parse_args()
    source = P1iThermFMSource(args.split_manifest, args.sample_root, args.full_field_archive)
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    receipt = compute_train_stats(source, source.ids_by_role["train"], args.stats_out, args.receipt_out)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
