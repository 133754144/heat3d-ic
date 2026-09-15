#!/usr/bin/env python3
"""Evaluate frozen Heat3D-on-v1 checkpoints on the 128 valid full fields.

This is an inference-only companion to the frozen 200-epoch cohort.  It does
not train, alter, or overwrite any existing checkpoint and never discovers or
opens a test/sealed file.  The purpose is to put Heat3D and the matched
DeepOHeat-v1 cohort on the same 101x101x56 temperature-space evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_runtime.evaluation import EvaluationCore, EvaluationSample
from rigno.heat3d_training import (
    block_until_ready,
    build_p1i_batches,
    model_apply_full,
)
from rigno.heat3d_training.p1i import (
    attach_input_contexts,
    attach_native_physics,
    attach_qk_features,
    fit_native_loss_references,
)
from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget
from rigno.heat3d_v6_dataset import V6_DUAL_ROBIN_CONDITION_FEATURES, V6DualRobinExample
from rigno.heat3d_v6_full_field import build_reconstruction_map
from rigno.heat3d_v6_p1i_anchor_query import conservative_selected_control_volume
from rigno.models.rigno import RIGNO


SUBSET_SHA = "e719665176a22213487ee92c1aac993dd01b02a51555c7cd68bf81a13b861558"
LABEL_RECEIPT_SHA = "a4bb99638a977b2004a93a88b469166ff7da697e89181e64e04152c7f96fe4fd"
NORMALIZATION_PAYLOAD_SHA = "3a0273bb92b8c060df8a214b1e0e7dd0e4b5df6bece86b7dea15197ca56ed0db"
MESH_SHAPE = (101, 101, 56)
FORBIDDEN = {"fs_test_volume.npy", "u_test_volume.npy"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def install_guard() -> None:
    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        candidate = arguments[0]
        if isinstance(candidate, (str, bytes, os.PathLike)) and Path(os.fsdecode(candidate)).name in FORBIDDEN:
            raise PermissionError("FAIL-CLOSED: evaluator attempted to open a forbidden test file")

    sys.addaudithook(audit)


def load_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("payload_sha256")
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if claimed != actual or actual != NORMALIZATION_PAYLOAD_SHA or payload["valid_or_test_used_to_fit"] is not False:
        raise ValueError("frozen normalization mismatch")
    stats = dict(payload["statistics"])
    for key in ("coord_min", "coord_span", "condition_mean", "condition_std", "target_delta_mean", "target_delta_std"):
        stats[key] = np.asarray(stats[key], dtype=np.float32)
    return stats


def checkpoint_params(path: Path) -> tuple[Any, dict[str, Any]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if "params" not in payload:
        raise ValueError(f"checkpoint has no params: {path}")
    return payload["params"], {key: payload.get(key) for key in ("step", "epoch", "seed", "selection_metric", "selection_value")}


def build_examples(
    *, dataset: Any, role: str, full_coords: np.ndarray, full_cv: np.ndarray, layer_id: np.ndarray,
    converter: Any,
) -> list[V6DualRobinExample]:
    rows: list[V6DualRobinExample] = []
    for index in range(len(dataset)):
        compact = dataset[index]
        weights, audit = conservative_selected_control_volume(
            full_coords=full_coords, full_control_volume=full_cv,
            full_layer_id=layer_id, selected_indices=compact["support_indices"],
        )
        if audit["relative_volume_error"] > 1.0e-12:
            raise ValueError("support control-volume conservation failed")
        features = compact["features"].astype(np.float64)
        metadata = {
            "split": role,
            "physics": {"ambient_K": 298.15, "footprint_m": [1.0, 1.0], "layers_bottom_to_top": [
                {"name": "lower", "thickness_m": 0.1, "k_W_mK": 2.0},
                {"name": "upper", "thickness_m": 0.45, "k_W_mK": 0.1},
            ]},
            "package_total_power_W": float(np.dot(np.maximum(features[:, 3], 0), weights)),
            "v6_adapter": {"dataset_id": "deepoheat_v1_volumetric_method_native_1024", "manifest_split_role": role,
                "group_id": compact["sample_id"], "reference_temperature_K": 298.15,
                "top_T_inf_K": 298.15, "bottom_T_inf_K": 298.15,
                "bottom_boundary_semantics": "robin_not_dirichlet", "official_source_index": compact["source_index"]},
        }
        rows.append(V6DualRobinExample(
            sample_id=compact["sample_id"],
            condition=V1SteadyConditionInput(
                coords=compact["coords"].astype(np.float64), condition_features=features,
                condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES, k_encoding_mode="diag3",
            ),
            target=V1SteadyTarget(target_u=(298.15 + compact["target_1024"]).reshape(-1, 1)),
            meta=metadata, operator_point_weights=weights,
        ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    install_guard()
    if sha256(args.subset_manifest) != SUBSET_SHA or sha256(args.labels_root / "label_generation_receipt.json") != LABEL_RECEIPT_SHA:
        raise ValueError("frozen subset/label receipt mismatch")
    if args.checkpoint.exists() is False:
        raise FileNotFoundError(args.checkpoint)
    if sha256(args.fs_train) != "a39a4f51e853f9114d86feb88f74553914b2bfc68ab1c553a3a31df25893fff7":
        raise ValueError("official fs_train SHA mismatch")
    stats = load_stats(args.normalization)
    config = json.loads(args.heat3d_config.read_text(encoding="utf-8"))
    loader = load_script("load_v7_g2_p6_deepoheat_v1_compact.py")
    helper = load_script("run_v7_g2_p5_heat3d_v1_dual_output_smoke.py")
    support = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    converter = load_script("convert_v7_g2_semiconductor_case.py")
    train_data = loader.CompactDeepOHeatV1Dataset(fs_train=args.fs_train, labels_root=args.labels_root, role="train")
    valid_data = loader.CompactDeepOHeatV1Dataset(fs_train=args.fs_train, labels_root=args.labels_root, role="valid", verify_source_file=False)
    mesh = support.mesh_arrays()
    train_examples = build_examples(dataset=train_data, role="train", full_coords=mesh["coords"], full_cv=mesh["control_volume"], layer_id=mesh["layer_id"], converter=converter)
    valid_examples = build_examples(dataset=valid_data, role="valid", full_coords=mesh["coords"], full_cv=mesh["control_volume"], layer_id=mesh["layer_id"], converter=converter)
    train_batches = build_p1i_batches(train_examples, stats, Heat3DGraphBuilder(**config["graph"]), label="p14_eval_train", batch_size=24, graph_seed=args.seed)
    valid_builder = Heat3DGraphBuilder(**config["graph"])
    valid_batches = build_p1i_batches(valid_examples, stats, valid_builder, label="p14_eval_valid", batch_size=32, graph_seed=args.seed)
    context = attach_input_contexts(train_batches + valid_batches, train_examples, train_examples + valid_examples, config["model"])
    by_id = {row.sample_id: row for row in train_examples + valid_examples}
    for batches in (train_batches, valid_batches):
        attach_native_physics(batches, by_id, context_by_id=context["raw_context_by_id"])
        attach_qk_features(batches, by_id, feature_version=str(config["model"]["qk_region_feature_version"]))
    model_config = helper.resolve_model_config(config["model"], tuple(stats["feature_names"]))
    model = RIGNO(**model_config)
    params, checkpoint_meta = checkpoint_params(args.checkpoint)
    # Build deterministic full-field truth and support maps from valid-only labels.
    label_receipt = json.loads((args.labels_root / "label_generation_receipt.json").read_text(encoding="utf-8"))
    valid_truth: dict[str, np.ndarray] = {}
    valid_q: dict[str, np.ndarray] = {}
    support_by_id: dict[str, np.ndarray] = {}
    for row in label_receipt["rows"]:
        if row.get("role") != "valid":
            continue
        directory = args.labels_root / "valid" / row["sample_id"]
        full = np.load(directory / row["artifacts"]["full_reference"]["file"], mmap_mode="r", allow_pickle=False)
        if array_sha(full) != row["artifacts"]["full_reference"]["sha256"]:
            raise ValueError(f"truth SHA drift: {row['sample_id']}")
        valid_truth[row["sample_id"]] = np.asarray(full, dtype=np.float64).reshape(-1)
        support_by_id[row["sample_id"]] = np.asarray(np.load(directory / row["artifacts"]["support_indices"]["file"], allow_pickle=False), dtype=np.int64)
        # ``features[:,3]`` is the deterministic volumetric q field at the
        # support points; rebuild the full q field from the same source input
        # for the evaluator's source/background bookkeeping.
        source_index = int(row["source_index"])
        power = np.asarray(valid_data.fs_train[source_index], dtype=np.float32)
        valid_q[row["sample_id"]] = np.asarray(converter.volume_v1_arrays(power)["features"][:, 3], dtype=np.float64)
    if len(valid_truth) != 128:
        raise ValueError("valid truth count is not 128")
    predictions: list[Any] = []
    started = time.perf_counter()
    for batch in valid_batches:
        prediction = model_apply_full(model, params, batch, None)
        block_until_ready(prediction)
        predictions.append(prediction)
    inference_seconds = time.perf_counter() - started
    samples: list[EvaluationSample] = []
    for prediction_batch, batch in zip(predictions, valid_batches, strict=True):
        native = np.asarray(prediction_batch[0]["deltaT_hat"], dtype=np.float64)[:, 0, :, 0]
        for row_index, sample_id in enumerate(batch.sample_ids):
            mapping, _audit = build_reconstruction_map(
                coords=mesh["coords"], layer_id=mesh["layer_id"], boundaries=np.asarray([0.0, 0.1, 0.55]),
                support_indices=support_by_id[str(sample_id)], empty_domain_fallback="same_layer", query_workers=1,
            )
            pred_full = mapping.reconstruct(native[row_index])
            truth = valid_truth[str(sample_id)]
            samples.append(EvaluationSample(
                sample_id=str(sample_id), prediction_deltaT_K=pred_full, truth_deltaT_K=truth,
                control_volumes_m3=mesh["control_volume"], coords=mesh["coords"],
                layer_id=mesh["layer_id"], q_W_m3=valid_q[str(sample_id)], split="valid_iid",
            ))
    evaluation = EvaluationCore().evaluate(samples)
    receipt = {
        "schema_version": "heat3d_v7_g2_p14_heat3d_fullfield_valid_evaluation_v1",
        "status": "PASS_VALID_ONLY_INFERENCE",
        "seed": args.seed,
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint), "metadata": checkpoint_meta},
        "runner": {"script": "scripts/evaluate_v7_g2_p14_heat3d_v1_fullfield.py", "repo_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "data": {"train": 768, "valid": 128, "subset_sha256": SUBSET_SHA, "labels_receipt_sha256": LABEL_RECEIPT_SHA, "normalization_payload_sha256": NORMALIZATION_PAYLOAD_SHA, "test_or_sealed_access": False},
        "domain": {"shape": list(MESH_SHAPE), "count": int(np.prod(MESH_SHAPE)), "temperature_space": "deltaT_K", "reconstruction": "frozen layer/interface-aware inverse-distance support map", "conditioning_count": 1024},
        "inference": {"wall_seconds": inference_seconds, "sample_count": evaluation["sample_count"], "metrics": evaluation["metrics"]},
        "hard_boundaries": {"p1i_test_iid_accessed": False, "sealed_accessed": False, "deepoheat_official100_accessed": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
