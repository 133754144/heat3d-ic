#!/usr/bin/env python3
"""One-time, preregistered V6_03 transfer audit on random-block test.

This evaluator is intentionally explicit: it accepts only the frozen random-block
dataset ID, reconstructs the canonical 11 input channels from raw arrays and
metadata, and reuses the checkpoint's frozen P1h normalization/context payload.
It never refits or silently adapts normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import evaluate_heat3d_v6_common_valid_probe as common  # noqa: E402
from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget  # noqa: E402
from rigno.heat3d_v6_dataset import V6_DUAL_ROBIN_CONDITION_FEATURES, V6DualRobinExample  # noqa: E402
from rigno.heat3d_v6_full_field import FullFieldMetricAccumulator, build_reconstruction_map  # noqa: E402


DATASET_ID = "heat3d_v6_randomblock_formal1024_v2"
CHECKPOINT_SPEC = {
    "checkpoint_epoch": 111,
    "checkpoint_sha256": "3ad58c2b34a46481acb74722c80bdcadbf55a0d613bc25c4fe2d7646b91aa1f2",
    "training_commit": "950a1ce",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def flags(coords: np.ndarray) -> np.ndarray:
    top = np.isclose(coords[:, 2], np.max(coords[:, 2]), atol=1e-15)
    bottom = np.isclose(coords[:, 2], np.min(coords[:, 2]), atol=1e-15)
    side = (
        np.isclose(coords[:, 0], np.min(coords[:, 0]), atol=1e-15)
        | np.isclose(coords[:, 0], np.max(coords[:, 0]), atol=1e-15)
        | np.isclose(coords[:, 1], np.min(coords[:, 1]), atol=1e-15)
        | np.isclose(coords[:, 1], np.max(coords[:, 1]), atol=1e-15)
    ) & ~top & ~bottom
    return np.column_stack((top, bottom, side, ~(top | bottom | side))).astype(float)


def boundaries(meta: dict[str, Any]) -> np.ndarray:
    values = [0.0]
    for layer in meta["layers_bottom_to_top"]:
        values.append(values[-1] + float(layer["thickness_m"]))
    return np.asarray(values, dtype=np.float64)


def load_test(dataset_root: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset_id") != DATASET_ID or len(manifest.get("samples", [])) != 1024:
        raise RuntimeError("random-block frozen manifest identity/count mismatch")
    rows = [row for row in manifest["samples"] if row["split_role"] == "test"]
    if len(rows) != 128:
        raise RuntimeError("random-block test count must be 128")
    examples = []
    support_truth = {}
    support_public = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        sample_dir = dataset_root / str(row["sample_dir"])
        meta = json.loads((sample_dir / "sample_meta.json").read_text())
        if meta["dataset_id"] != DATASET_ID or meta["split_role"] != "test":
            raise RuntimeError(f"{sample_id}: provenance/split mismatch")
        coords = np.load(sample_dir / "coords.npy").astype(np.float64)
        k = np.load(sample_dir / "k_field.npy").astype(np.float64)
        q = np.load(sample_dir / "q_field.npy").astype(np.float64).reshape(-1)
        temperature = np.load(sample_dir / "temperature.npy").astype(np.float64).reshape(-1)
        cv = np.load(sample_dir / "control_volume.npy").astype(np.float64).reshape(-1)
        layer = np.load(sample_dir / "layer_id.npy").astype(np.int32).reshape(-1)
        stored_flags = np.load(sample_dir / "bc_features.npy").astype(np.float64)
        if coords.shape != (1024, 3) or stored_flags.shape != (1024, 4):
            raise RuntimeError(f"{sample_id}: support schema drift")
        if not np.array_equal(stored_flags, flags(coords)):
            raise RuntimeError(f"{sample_id}: BC flag semantics drift")
        top_bc = meta["boundary_conditions"]["top"]
        bottom_bc = meta["boundary_conditions"]["bottom"]
        if top_bc["type"] != "robin" or bottom_bc["type"] != "robin":
            raise RuntimeError(f"{sample_id}: expected dual Robin")
        top_h, bottom_h = float(top_bc["h_W_m2K"]), float(bottom_bc["h_W_m2K"])
        top_t, bottom_t = float(top_bc["T_inf_K"]), float(bottom_bc["T_inf_K"])
        condition = np.column_stack((
            k, q, stored_flags,
            np.full(1024, top_h), np.full(1024, bottom_h), np.full(1024, top_t-bottom_t),
        ))
        if condition.shape != (1024, len(V6_DUAL_ROBIN_CONDITION_FEATURES)):
            raise RuntimeError(f"{sample_id}: canonical 11D schema mismatch")
        enriched = dict(meta)
        enriched["v6_adapter"] = {
            "dataset_id": DATASET_ID,
            "manifest_split_role": "test",
            "group_id": str(row["group_id"]),
            "reference_temperature_K": bottom_t,
            "top_T_inf_K": top_t,
            "bottom_T_inf_K": bottom_t,
            "bottom_boundary_semantics": "robin_not_dirichlet",
            "operator_point_measure": "control_volume_randomblock1024",
        }
        example = V6DualRobinExample(
            sample_id=sample_id,
            condition=V1SteadyConditionInput(coords=coords, condition_features=condition,
                condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES, k_encoding_mode="diag3"),
            target=V1SteadyTarget(target_u=temperature[:, None]), meta=enriched,
            operator_point_weights=cv,
        )
        examples.append(example)
        support_truth[sample_id] = {"deltaT_K": temperature-bottom_t, "q_W_m3": q}
        support_public[sample_id] = {"control_volume": cv, "layer_id": layer, "coords": coords}
    return manifest, rows, examples, support_truth, support_public


def aggregate_support(predictions, examples, truth, public):
    sse = energy = volume = 0.0
    sample_rel = []
    bias_num = 0.0
    for example in examples:
        sid = example.sample_id
        ref = float(example.meta["v6_adapter"]["reference_temperature_K"])
        pred = np.asarray(predictions[sid]) - ref
        true = truth[sid]["deltaT_K"]
        cv = public[sid]["control_volume"]
        err = pred-true
        row_sse = float(np.sum(cv*err*err)); row_energy=float(np.sum(cv*true*true))
        sse += row_sse; energy += row_energy; volume += float(np.sum(cv)); bias_num += float(np.sum(cv*err))
        sample_rel.append(math.sqrt(row_sse/row_energy)*100)
    return {
        "point_global_cv_relative_rmse_pct": math.sqrt(sse/energy)*100,
        "sample_first_cv_relative_rmse_pct": float(np.mean(sample_rel)),
        "raw_cv_weighted_rmse_K": math.sqrt(sse/volume),
        "cv_weighted_bias_K": bias_num/volume,
    }


def evaluate(args):
    manifest, rows, examples, support_truth, support_public = load_test(args.dataset, args.manifest)
    predictions, checkpoint = common._predict(
        run_dir=args.run_dir, spec=CHECKPOINT_SPEC, examples=examples, batch_size=args.batch_size
    )
    support_metrics = aggregate_support(predictions, examples, support_truth, support_public)
    archive_path = args.dataset / manifest["full_field_archive"]["relative_path"]
    if sha256(archive_path) != manifest["full_field_archive"]["sha256"]:
        raise RuntimeError("random-block full-field archive SHA mismatch")
    with h5py.File(archive_path, "r") as h:
        full_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in h["sample_id"][:]]
        lookup = {sid:i for i,sid in enumerate(full_ids)}
        full_coords = np.asarray(h["coords"][:], dtype=np.float64)
        full_cv = np.asarray(h["control_volume"][:], dtype=np.float64)
        full_layer = np.asarray(h["layer_id"][:], dtype=np.int32)
        first_meta = examples[0].meta
        acc = FullFieldMetricAccumulator(control_volume=full_cv, layer_id=full_layer,
            boundaries=boundaries(first_meta), coords=full_coords)
        reconstruction_hashes = []
        for index, example in enumerate(examples, start=1):
            sid=example.sample_id; row=lookup[sid]
            support_coords=np.asarray(example.condition.coords)
            distance, support_indices = cKDTree(full_coords).query(support_coords, k=1)
            if float(np.max(distance)) > 1e-14 or len(np.unique(support_indices)) != 1024:
                raise RuntimeError(f"{sid}: support is not exact full-mesh subset")
            mapping, audit = build_reconstruction_map(coords=full_coords, layer_id=full_layer,
                boundaries=boundaries(example.meta), support_indices=support_indices)
            reconstruction_hashes.append(hashlib.sha256(mapping.neighbor_local_indices.tobytes()+mapping.neighbor_weights.tobytes()).hexdigest())
            ref=float(example.meta["v6_adapter"]["reference_temperature_K"])
            pred_support=np.asarray(predictions[sid])-ref
            pred_full=mapping.reconstruct(pred_support)
            true_full=np.asarray(h["temperature_K"][row],dtype=np.float64)-ref
            q_full=np.asarray(h["q_W_m3"][row],dtype=np.float64)
            acc.add(kind="model",sample_id=sid,prediction_delta=pred_full,truth_delta=true_full,q=q_full)
            if index % 16 == 0: print(f"[randomblock-transfer] full-field {index}/128",flush=True)
        full_metrics=acc.summarize("model")
    payload = {
        "schema_version":"heat3d_v6_p1i_randomblock_transfer_v1", "status":"passed",
        "evaluation_role":"randomblock_test_one_time_preregistered", "sample_count":128,
        "selection_or_tuning":False, "silent_adapter":False,
        "dataset":{"id":DATASET_ID,"manifest_sha256":sha256(args.manifest),
            "manifest_payload_sha256":manifest["manifest_payload_sha256"],
            "full_field_archive_sha256":sha256(archive_path)},
        "checkpoint":checkpoint,
        "compatibility":{"condition_feature_names":list(V6_DUAL_ROBIN_CONDITION_FEATURES),
            "condition_width":11,"dual_robin":True,"diag3_k":True,
            "frozen_checkpoint_normalization":True,"normalization_refit":False,
            "frozen_checkpoint_global_context_standardizer":True,
            "explicit_adapter":"randomblock_raw_arrays_metadata_v1"},
        "support_metrics":support_metrics,"full_field_metrics":full_metrics,
        "reconstruction_map_hash_count":len(set(reconstruction_hashes)),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    args.markdown.write_text(
        "# V6 P1i random-block one-time transfer audit\n\n"
        "Frozen V6_03 e111 was evaluated once on the preregistered random-block test. "
        "No normalization/context refit, checkpoint selection, or tuning was performed.\n\n"
        f"- support point-global CV relative RMSE: {support_metrics['point_global_cv_relative_rmse_pct']:.6f}%\n"
        f"- support sample-first CV relative RMSE: {support_metrics['sample_first_cv_relative_rmse_pct']:.6f}%\n"
        f"- support raw CV RMSE: {support_metrics['raw_cv_weighted_rmse_K']:.6f} K\n"
        f"- full-field point-global CV relative RMSE: {full_metrics['cv_weighted_point_global_relative_rmse_pct']:.6f}%\n"
        f"- full-field raw CV RMSE: {full_metrics['cv_weighted_rmse_K']:.6f} K\n"
        f"- full-field peak RMSE: {full_metrics['peak_error_rmse_K']:.6f} K\n"
    )
    return payload


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--output",type=Path,default=ROOT/"configs/heat3d_v6_p1i/v6_p1i_randomblock_transfer.json")
    parser.add_argument("--markdown",type=Path,default=ROOT/"docs/v6_p1i_randomblock_transfer.md")
    args=parser.parse_args(); evaluate(args); print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
