#!/usr/bin/env python3
"""Prepare label-independent remaining-valid96 supports and frozen reconstruction maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path: sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "frozen_after_E_no_go_before_confirmation":
        raise RuntimeError("confirmation protocol not frozen")
    binding = highn._binding(args)
    if sha(args.manifest) != binding["dataset"]["manifest_sha256"] or sha(args.full_fields) != binding["dataset"]["full_field_archive_sha256"]:
        raise RuntimeError("dataset binding drifted")
    dataset = highn._dataset(args)
    sample_ids = sorted(
        dataset.split_ids["valid_iid"], key=lambda sample_id: hashlib.sha256(sample_id.encode()).hexdigest()
    )[32:]
    if len(sample_ids) != 96:
        raise RuntimeError("remaining valid population is not 96")
    index = dataset.sample_index_by_id()
    examples = [dataset[index[sample_id]] for sample_id in sample_ids]
    full, archive_lookup = highn._full_shared(args)
    supports = {str(n): [] for n in protocol["resolutions"]}
    maps = {str(n): [] for n in protocol["resolutions"]}
    oracle_rows = {str(n): [] for n in protocol["resolutions"]}
    sample_rows = []
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.full_fields, "r") as archive:
        for number, example in enumerate(examples, start=1):
            anchor_indices, distance = highn._anchor_indices(example, full["coords"], 1.0e-14)
            mesh, full_k, full_q, power = highn._physics_fields(example, full)
            if power["relative_power_error"] > 1.0e-10:
                raise RuntimeError(f"{example.sample_id}: power drift")
            physics_path = args.artifact_root / "physics" / f"{example.sample_id}.npz"
            physics_path.parent.mkdir(parents=True, exist_ok=True)
            with physics_path.open("wb") as handle:
                np.savez_compressed(handle, k_xyz=full_k, q_W_m3=full_q)
            order, order_audit = highn.deterministic_nested_query_order(
                sample_id=example.sample_id, anchor_indices=anchor_indices,
                full_coords=full["coords"], full_control_volume=full["cv"],
                full_layer_id=full["layer"], full_q=full_q,
                layer_boundaries_m=np.asarray(mesh["boundaries"], dtype=np.float64),
                selection_seed=int(binding["nested_support"]["selection_seed"]),
            )
            truth = np.asarray(archive["samples/deltaT_K"][archive_lookup[example.sample_id]], dtype=np.float64)
            for resolution in protocol["resolutions"]:
                indices = np.asarray(order[:resolution], dtype=np.int64)
                effective_cv, cv_audit = highn.conservative_selected_control_volume(
                    full_coords=full["coords"], full_control_volume=full["cv"],
                    full_layer_id=full["layer"], selected_indices=indices,
                )
                if cv_audit["relative_volume_error"] > 1.0e-12:
                    raise RuntimeError(f"{example.sample_id}: CV drift")
                support_path = args.artifact_root / "support" / str(resolution) / f"{example.sample_id}.npz"
                support_path.parent.mkdir(parents=True, exist_ok=True)
                with support_path.open("wb") as handle:
                    np.savez_compressed(
                        handle, selected_indices=indices.astype(np.int32),
                        operator_control_volume=effective_cv, k_xyz=full_k[indices],
                        q_W_m3=full_q[indices], layer_id=full["layer"][indices],
                    )
                support_hash = highn.array_sha256(indices.astype(np.int32))
                support_row = {
                    "sample_id": example.sample_id, "resolution": resolution,
                    "support_file": str(support_path), "support_file_sha256": sha(support_path),
                    "ordered_support_hash": support_hash,
                    "volume_relative_error": cv_audit["relative_volume_error"],
                    "nonzero_q_count": int(np.sum(full_q[indices] > 0.0)),
                }
                if support_row["nonzero_q_count"] <= 0:
                    raise RuntimeError(f"{example.sample_id}: source coverage")
                supports[str(resolution)].append(support_row)
                boundaries = highn._boundaries(example, float(np.min(full["coords"][:, 2])))
                mapping, build = highn.build_reconstruction_map(
                    coords=full["coords"], layer_id=full["layer"], boundaries=boundaries,
                    support_indices=indices.astype(np.int32), empty_domain_fallback="same_layer",
                )
                map_path = args.artifact_root / "reconstruction_cache" / str(resolution) / f"{example.sample_id}.npz"
                io = highn.save_reconstruction_map(map_path, mapping)
                maps[str(resolution)].append({
                    "sample_id": example.sample_id, "cache_file": str(map_path),
                    "cache_file_sha256": sha(map_path), "mapping_hash": highn._mapping_sha256(mapping),
                    "build": build, "io": io,
                })
                oracle = mapping.reconstruct(truth[indices])
                oracle_rows[str(resolution)].append(highn._metric_row(
                    oracle, truth, full["cv"], full["coords"], full["layer"], full_q,
                ))
            sample_rows.append({
                "sample_id": example.sample_id, "anchor_max_distance_m": distance,
                "physics_cache_file": str(physics_path), "physics_cache_sha256": sha(physics_path),
                "nested_order": order_audit,
            })
            print(f"[confirm-preflight] {number}/96 {example.sample_id}", flush=True)
    oracle = {
        str(n): qualification.metric_accumulate(oracle_rows[str(n)], full=True)
        for n in protocol["resolutions"]
    }
    payload = {
        "schema_version": "heat3d_v6_p1i_graph_policy_confirmation_inputs_v1",
        "status": "passed", "sample_ids": sample_ids, "sample_count": 96,
        "population_rule": protocol["population"]["rule"],
        "supports": supports, "reconstruction_maps": maps,
        "oracle_reconstruction_floor": oracle, "samples": sample_rows,
        "dataset": {"manifest_sha256": sha(args.manifest), "full_fields_sha256": sha(args.full_fields)},
        "role_contract": {"training": False, "test": False, "sealed": False, "valid_iid_only": True},
    }
    write_json(args.artifact_root / "actual_data_preflight.json", payload)
    print(json.dumps({"status": "passed", "samples": 96, "resolutions": protocol["resolutions"]}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
