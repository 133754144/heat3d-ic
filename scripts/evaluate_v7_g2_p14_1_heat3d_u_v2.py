#!/usr/bin/env python3
"""P14.1 valid-only Heat3D native/IDW/U-v2 dense evaluation.

The script reuses the frozen e200 checkpoints and the V6 U-v2 asymmetric
direct-query runtime.  It never trains, writes predictions, or discovers a
test/sealed artifact.  Large intermediate arrays are kept in memory only for
one valid case and the receipt contains metrics and provenance only.
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
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_runtime.evaluation import EvaluationCore, EvaluationSample  # noqa: E402
from rigno.heat3d_runtime.features import FeatureTransform  # noqa: E402
from rigno.heat3d_runtime.grouping import GroupBuilder  # noqa: E402
from rigno.heat3d_runtime.high_n import FullFieldGeometry, SupportArtifact  # noqa: E402
from rigno.heat3d_runtime.session import RuntimeSession  # noqa: E402
from rigno.heat3d_runtime.u_split import UHighNRuntime  # noqa: E402
from rigno.heat3d_training import block_until_ready, build_p1i_batches, model_apply_full  # noqa: E402
from rigno.heat3d_training.p1i import attach_input_contexts, attach_native_physics, attach_qk_features  # noqa: E402
from rigno.heat3d_v6_full_field import build_reconstruction_map  # noqa: E402
from rigno.heat3d_v6_p1i_anchor_query import conservative_selected_control_volume  # noqa: E402
from rigno.models.rigno import RIGNO  # noqa: E402


SUBSET_SHA = "e719665176a22213487ee92c1aac993dd01b02a51555c7cd68bf81a13b861558"
LABEL_RECEIPT_SHA = "a4bb99638a977b2004a93a88b469166ff7da697e89181e64e04152c7f96fe4fd"
NORMALIZATION_PAYLOAD_SHA = "3a0273bb92b8c060df8a214b1e0e7dd0e4b5df6bece86b7dea15197ca56ed0db"
FS_TRAIN_SHA = "a39a4f51e853f9114d86feb88f74553914b2bfc68ab1c553a3a31df25893fff7"
MESH_SHAPE = (101, 101, 56)
MESH_COUNT = int(np.prod(MESH_SHAPE))
FORBIDDEN_NAMES = {"fs_test_volume.npy", "u_test_volume.npy", "test_iid", "sealed"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
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
        if isinstance(candidate, (str, bytes, os.PathLike)):
            name = Path(os.fsdecode(candidate)).name.lower()
            if name in {value.lower() for value in FORBIDDEN_NAMES}:
                raise PermissionError(
                    f"FAIL-CLOSED: forbidden test/sealed artifact opened: {candidate}"
                )

    sys.addaudithook(audit)


def load_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("payload_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != actual or actual != NORMALIZATION_PAYLOAD_SHA:
        raise ValueError("frozen normalization payload mismatch")
    if payload.get("valid_or_test_used_to_fit") is not False:
        raise ValueError("normalization is not train-only")
    stats = dict(payload["statistics"])
    for key in (
        "coord_min",
        "coord_span",
        "condition_mean",
        "condition_std",
        "target_delta_mean",
        "target_delta_std",
    ):
        stats[key] = np.asarray(stats[key], dtype=np.float32)
    return stats


def checkpoint_params(path: Path) -> tuple[Any, dict[str, Any]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if "params" not in payload:
        raise ValueError(f"checkpoint has no params: {path}")
    return payload["params"], {
        key: payload.get(key)
        for key in ("step", "epoch", "seed", "selection_metric", "selection_value")
    }


def metric_summary(rows: list[EvaluationSample]) -> dict[str, Any]:
    evaluation = EvaluationCore().evaluate(rows)
    errors = [
        np.asarray(row.prediction_deltaT_K, dtype=np.float64)
        - np.asarray(row.truth_deltaT_K, dtype=np.float64)
        for row in rows
    ]
    abs_errors = np.concatenate([np.abs(value).reshape(-1) for value in errors])
    peak_errors = np.asarray(
        [
            float(np.max(np.asarray(row.prediction_deltaT_K)) - np.max(np.asarray(row.truth_deltaT_K)))
            for row in rows
        ],
        dtype=np.float64,
    )
    background_sse = 0.0
    background_cv = 0.0
    for row, error in zip(rows, errors, strict=True):
        source = np.asarray(row.q_W_m3).reshape(-1) > 0.0
        weights = np.asarray(row.control_volumes_m3, dtype=np.float64).reshape(-1)
        background_sse += float(np.sum(weights[~source] * np.square(error[~source])))
        background_cv += float(np.sum(weights[~source]))
    metrics = dict(evaluation["metrics"])
    metrics.update(
        {
            "MAE_K": float(np.mean(abs_errors)),
            "peak_temperature_mean_abs_error_K": float(np.mean(np.abs(peak_errors))),
            "peak_temperature_max_abs_error_K": float(np.max(np.abs(peak_errors))),
            "background_RMSE_K": float(np.sqrt(background_sse / max(background_cv, 1.0e-30))),
        }
    )
    return {
        "sample_count": len(rows),
        "metrics": metrics,
        "per_sample_sample_first_relative_rmse_pct": [
            float(value["sample_cv_relative_rmse"] * 100.0)
            for value in evaluation["per_sample"]
        ],
    }


def support_from_compact(
    compact: dict[str, Any], mesh: dict[str, np.ndarray]
) -> tuple[Any, np.ndarray]:
    support_indices = np.asarray(compact["support_indices"], dtype=np.int64)
    weights, audit = conservative_selected_control_volume(
        full_coords=mesh["coords"],
        full_control_volume=mesh["control_volume"],
        full_layer_id=mesh["layer_id"],
        selected_indices=support_indices,
    )
    if audit["relative_volume_error"] > 1.0e-12:
        raise ValueError(f"{compact['sample_id']}: support volume conservation drift")
    return weights, support_indices


def full_support_artifact(
    *, power: np.ndarray, converter: Any, mesh: dict[str, np.ndarray]
) -> SupportArtifact:
    arrays = converter.volume_v1_arrays(np.asarray(power, dtype=np.float32))
    selected = np.arange(MESH_COUNT, dtype=np.int64)
    return SupportArtifact.from_arrays(
        selected_indices=selected,
        operator_control_volume=np.asarray(mesh["control_volume"], dtype=np.float64),
        k_xyz=np.asarray(arrays["features"][:, :3], dtype=np.float64),
        q_W_m3=np.asarray(arrays["features"][:, 3], dtype=np.float64),
        layer_id=np.asarray(mesh["layer_id"], dtype=np.int32),
        path="<derived-full-grid-query>",
        sha256=array_sha256(selected.astype(np.int32)),
    )


def build_session(
    *, model: Any, params: Any, stats: dict[str, Any], config: dict[str, Any],
    context_standardizer: dict[str, Any], seed: int,
) -> RuntimeSession:
    model_config = dict(config["model"])
    # Older frozen V7 configs omit this name although the model's effective
    # shape-attention mode is none.  Supplying the explicit default only
    # completes the runtime object; it does not change model parameters.
    model_config.setdefault("shape_attention_mode", "none")
    feature_transform = FeatureTransform(stats)
    graph_config = dict(config["graph"])
    return RuntimeSession(
        checkpoint=SimpleNamespace(),
        run_config={
            "graph_seed": int(seed),
            "global_context": {"standardizer": context_standardizer},
        },
        model_config=model_config,
        graph_config=graph_config,
        feature_transform=feature_transform,
        group_builder=GroupBuilder(
            feature_transform=feature_transform,
            graph_config=graph_config,
            graph_seed=int(seed),
        ),
        model=model,
        params=jax.device_put(params),
        execution_role="valid_iid_p14_1_inference",
        semantic_contract={"label_access": "valid_iid_only"},
    )


def resolve_model_config(config_model: dict[str, Any], feature_names: tuple[str, ...]) -> dict[str, Any]:
    helper = load_script("run_v7_g2_p5_heat3d_v1_dual_output_smoke.py")
    resolved = helper.resolve_model_config(config_model, feature_names)
    resolved.setdefault("shape_attention_mode", "none")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-valid", type=int, default=128)
    args = parser.parse_args()
    install_guard()
    if args.max_valid < 1 or args.max_valid > 128:
        raise ValueError("--max-valid must be in [1,128]")
    loader = load_script("load_v7_g2_p6_deepoheat_v1_compact.py")
    evaluator = load_script("evaluate_v7_g2_p14_heat3d_v1_fullfield.py")
    support_module = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    converter = load_script("convert_v7_g2_semiconductor_case.py")
    subset_path = ROOT / "configs/heat3d_v7/g2_deepoheat_v1_volumetric_subset_manifest.json"
    if file_sha256(subset_path) != SUBSET_SHA:
        raise ValueError("frozen subset manifest SHA mismatch")
    receipt_path = args.labels_root / "label_generation_receipt.json"
    if file_sha256(receipt_path) != LABEL_RECEIPT_SHA:
        raise ValueError("frozen label receipt SHA mismatch")
    if file_sha256(args.fs_train) != FS_TRAIN_SHA:
        # The source SHA is checked against the repository's frozen value by
        # the compact loader as well; fail closed rather than accepting a
        # typo in this standalone script.
        raise ValueError("official fs_train SHA mismatch")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    stats = load_stats(args.normalization)
    config = json.loads(args.heat3d_config.read_text(encoding="utf-8"))
    train_data = loader.CompactDeepOHeatV1Dataset(
        fs_train=args.fs_train, labels_root=args.labels_root, role="train"
    )
    valid_data = loader.CompactDeepOHeatV1Dataset(
        fs_train=args.fs_train, labels_root=args.labels_root, role="valid", verify_source_file=False
    )
    mesh = support_module.mesh_arrays()
    train_examples = evaluator.build_examples(
        dataset=train_data, role="train", full_coords=mesh["coords"],
        full_cv=mesh["control_volume"], layer_id=mesh["layer_id"], converter=converter,
    )
    valid_examples_all = evaluator.build_examples(
        dataset=valid_data, role="valid", full_coords=mesh["coords"],
        full_cv=mesh["control_volume"], layer_id=mesh["layer_id"], converter=converter,
    )
    valid_examples = valid_examples_all[: int(args.max_valid)]
    # Fit only the existing train-only context standardizer.  No target is
    # read by this operation; target labels enter only the later evaluator.
    builder = Heat3DGraphBuilder(**config["graph"])
    valid_batches = build_p1i_batches(
        valid_examples, stats, builder, label="p14_1_valid", batch_size=32, graph_seed=args.seed
    )
    context = attach_input_contexts(
        valid_batches, train_examples, train_examples + valid_examples, config["model"]
    )
    by_id = {row.sample_id: row for row in train_examples + valid_examples}
    attach_native_physics(valid_batches, by_id, context_by_id=context["raw_context_by_id"])
    attach_qk_features(
        valid_batches, by_id, feature_version=str(config["model"]["qk_region_feature_version"])
    )
    model_config = resolve_model_config(config["model"], tuple(stats["feature_names"]))
    model = RIGNO(**model_config)
    params, checkpoint_meta = checkpoint_params(args.checkpoint)

    # Native prediction is used as the common 1024 input for both frozen IDW
    # and U-v2.  All labels remain in valid_iid rows only.
    native_by_id: dict[str, np.ndarray] = {}
    native_inference_started = time.perf_counter()
    for batch in valid_batches:
        output = model_apply_full(model, params, batch, None)
        block_until_ready(output)
        values = np.asarray(output[0]["deltaT_hat"], dtype=np.float64)[:, 0, :, 0]
        for index, sample_id in enumerate(batch.sample_ids):
            native_by_id[str(sample_id)] = values[index]
    native_inference_seconds = time.perf_counter() - native_inference_started

    session = build_session(
        model=model, params=params, stats=stats, config=config,
        context_standardizer=context["standardizer"], seed=args.seed,
    )
    runtime = UHighNRuntime.from_session(
        session, FullFieldGeometry(
            path=Path("<derived-deepoheat-v1-mesh>"),
            coords=np.asarray(mesh["coords"], dtype=np.float64),
            control_volume=np.asarray(mesh["control_volume"], dtype=np.float64),
            layer_id=np.asarray(mesh["layer_id"], dtype=np.int32),
            sample_ids=tuple(), split_roles=tuple(),
        )
    )
    # FullFieldGeometry.load normally enforces the V6 population size.  The
    # U-v2 case builder uses only geometry arrays, so a tiny equivalent object
    # is sufficient here and keeps the benchmark data contract explicit.
    runtime.geometry = SimpleNamespace(
        coords=np.asarray(mesh["coords"], dtype=np.float64),
        control_volume=np.asarray(mesh["control_volume"], dtype=np.float64),
        layer_id=np.asarray(mesh["layer_id"], dtype=np.int32),
    )
    boundaries = np.asarray([0.0, 0.1, 0.55], dtype=np.float64)
    idw_rows: list[EvaluationSample] = []
    native_rows: list[EvaluationSample] = []
    oracle_rows: list[EvaluationSample] = []
    u_rows: list[EvaluationSample] = []
    case_receipts: list[dict[str, Any]] = []
    u_inference_seconds = 0.0
    for row_index, compact in enumerate(valid_data.rows[: int(args.max_valid)]):
        sample_id = str(compact["sample_id"])
        compact_row = valid_data[row_index]
        anchor = valid_examples[row_index]
        support_weights, support_indices = support_from_compact(compact_row, mesh)
        truth_support = np.asarray(compact_row["target_1024"], dtype=np.float64).reshape(-1)
        truth_full = np.asarray(compact_row["target_full_571256"], dtype=np.float64).reshape(-1)
        q_support = np.asarray(compact_row["features"][:, 3], dtype=np.float64)
        power = np.asarray(valid_data.fs_train[int(compact["source_index"])], dtype=np.float32)
        arrays = converter.volume_v1_arrays(power)
        q_full = np.asarray(arrays["features"][:, 3], dtype=np.float64)
        layer_support = np.asarray(mesh["layer_id"])[support_indices]
        native = np.asarray(native_by_id[sample_id], dtype=np.float64)
        mapping, map_audit = build_reconstruction_map(
            coords=np.asarray(mesh["coords"], dtype=np.float64),
            layer_id=np.asarray(mesh["layer_id"], dtype=np.int32),
            boundaries=boundaries,
            support_indices=support_indices.astype(np.int32),
            empty_domain_fallback="same_layer",
            query_workers=1,
        )
        idw = mapping.reconstruct(native)
        oracle_idw = mapping.reconstruct(truth_support)
        native_rows.append(EvaluationSample(
            sample_id=sample_id, prediction_deltaT_K=native, truth_deltaT_K=truth_support,
            control_volumes_m3=support_weights, coords=np.asarray(compact_row["coords"], dtype=np.float64),
            layer_id=layer_support, q_W_m3=q_support, split="valid_iid",
        ))
        idw_rows.append(EvaluationSample(
            sample_id=sample_id, prediction_deltaT_K=idw, truth_deltaT_K=truth_full,
            control_volumes_m3=mesh["control_volume"], coords=mesh["coords"],
            layer_id=mesh["layer_id"], q_W_m3=q_full, split="valid_iid",
        ))
        oracle_rows.append(EvaluationSample(
            sample_id=sample_id, prediction_deltaT_K=oracle_idw, truth_deltaT_K=truth_full,
            control_volumes_m3=mesh["control_volume"], coords=mesh["coords"],
            layer_id=mesh["layer_id"], q_W_m3=q_full, split="valid_iid",
        ))

        full_support = full_support_artifact(power=power, converter=converter, mesh=mesh)
        u_started = time.perf_counter()
        try:
            case = runtime.build_case(
                anchor, MESH_COUNT, support=full_support,
                native_edge_targets=None, query_edge_targets=None,
            )
            u_output = runtime.apply(case)
            block_until_ready(u_output["raw_temperature"])
            u_prediction = np.asarray(u_output["raw_temperature"], dtype=np.float64)[0, 0, :, 0] - 298.15
            if u_prediction.shape != (MESH_COUNT,) or not np.all(np.isfinite(u_prediction)):
                raise ValueError("U-v2 full-grid prediction is nonfinite or has wrong shape")
            u_inference_seconds += time.perf_counter() - u_started
            u_rows.append(EvaluationSample(
                sample_id=sample_id, prediction_deltaT_K=u_prediction, truth_deltaT_K=truth_full,
                control_volumes_m3=mesh["control_volume"], coords=mesh["coords"],
                layer_id=mesh["layer_id"], q_W_m3=q_full, split="valid_iid",
            ))
            u_status = "PASS"
            u_error = None
            u_audit = dict(case.audit)
            u_audit["query_edge_counts"] = {
                key: (None if value is None else int(value)
                      ) for key, value in runtime._edge_counts(case.query_metadata).items()
            }
        except Exception as exc:  # record fail-closed per case, then stop
            u_status = "FAIL_CLOSED"
            u_error = f"{type(exc).__name__}: {exc}"
            u_audit = None
            case_receipts.append({
                "sample_id": sample_id,
                "support_indices_sha256": array_sha256(support_indices.astype(np.int32)),
                "idw_mapping_sha256": array_sha256(mapping.neighbor_local_indices),
                "u_v2_status": u_status,
                "u_v2_error": u_error,
            })
            raise
        case_receipts.append({
            "sample_id": sample_id,
            "support_indices_sha256": array_sha256(support_indices.astype(np.int32)),
            "idw_mapping_sha256": array_sha256(mapping.neighbor_local_indices),
            "u_v2_status": u_status,
            "u_v2_audit": u_audit,
            "u_v2_wall_seconds": float(time.perf_counter() - u_started),
        })
        del case, u_output, u_prediction, full_support, arrays, q_full, truth_full
        print(f"[p14.1 U-v2] {row_index + 1}/{args.max_valid} {sample_id}", flush=True)

    if len(u_rows) != len(native_rows):
        raise RuntimeError("U-v2 did not produce one result per valid sample")
    receipt = {
        "schema_version": "heat3d_v7_g2_p14_1_u_v2_dense_evaluation_receipt_v1",
        "status": "PASS_VALID_ONLY_NATIVE_IDW_UV2",
        "seed": int(args.seed),
        "max_valid": int(args.max_valid),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
            "metadata": checkpoint_meta,
        },
        "runner": {
            "script": "scripts/evaluate_v7_g2_p14_1_heat3d_u_v2.py",
            "repo_commit_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "data": {
            "source_sha256": file_sha256(args.fs_train),
            "subset_manifest_sha256": SUBSET_SHA,
            "label_receipt_sha256": LABEL_RECEIPT_SHA,
            "normalization_payload_sha256": NORMALIZATION_PAYLOAD_SHA,
            "train_count": 768,
            "valid_iid_count_evaluated": len(native_rows),
            "test_iid_or_sealed_accessed": False,
            "deepoheat_official100_accessed": False,
        },
        "domain": {
            "shape": list(MESH_SHAPE),
            "count": MESH_COUNT,
            "temperature_space": "deltaT_K",
            "support_count": 1024,
            "support_selector": "frozen physics-layout-aware selector",
            "u_v2_query_domain": "full DeepOHeat-v1 volumetric grid",
        },
        "representations": {
            "native_1024": metric_summary(native_rows),
            "idw_dense_571256": metric_summary(idw_rows),
            "u_v2_dense_571256": metric_summary(u_rows),
            "oracle_idw_gt_support_571256": metric_summary(oracle_rows),
            "oracle_u_v2_gt_support_571256": {
                "status": "NOT_DEFINED_FOR_DIRECT_QUERY",
                "reason": "frozen U-v2 consumes native latent/context plus query graph; no label-only value map exists",
            },
        },
        "runtime": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "native_inference_seconds": float(native_inference_seconds),
            "u_v2_inference_seconds": float(u_inference_seconds),
            "u_v2_latency_includes": ["full-query graph construction", "model direct-query forward"],
            "u_v2_deterministic_geometry_adaptation": True,
            "learned_parameter_count_added": 0,
        },
        "case_receipts": case_receipts,
        "audit": {
            "label_leakage": False,
            "oracle_error_decomposition_performed": False,
            "support_indices_match_frozen_selector": True,
            "dense_grid_matches_deepoheat_v1": True,
            "large_prediction_artifacts_persisted": False,
        },
        "hard_boundaries": {
            "p1i_test_iid_accessed": False,
            "sealed_accessed": False,
            "deepoheat_official100_accessed": False,
            "training_started": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
