#!/usr/bin/env python3
"""Valid-only V7 e200 U-v2 inference on the P1i 240825-node grid.

This runner is intentionally inference-only.  It loads the frozen Heat3D
e200 parameters trained on the DeepOHeat-v1 volumetric domain, builds the
native P1i 1024 support and deterministic full query fields from P1i case
metadata, and evaluates the frozen U-v2 direct-query path at 65x65x57.  It
never opens test/sealed labels and never writes prediction arrays.
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

import h5py
import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_runtime.evaluation import EvaluationSample  # noqa: E402
from rigno.heat3d_runtime.high_n import SupportArtifact  # noqa: E402
from rigno.heat3d_runtime.u_split import UHighNRuntime  # noqa: E402
from rigno.heat3d_training.full_field import (  # noqa: E402
    FullFieldGeometry as TrainingFullFieldGeometry,
    load_full_field_geometry,
    materialize_full_input_fields,
)
from rigno.heat3d_v6_dataset import (  # noqa: E402
    CONTINUOUS_PHYSICS_V6_DATASET_ID,
    Heat3DV6DualRobinDataset,
)


P1I_MANIFEST_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
VALID_MANIFEST_SHA = "f42cca611b6c70551b153f4f887b2d47724d9082846e4b381cab4ad9c652e57c"
FULL_FIELDS_SHA = "49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb"
NORMALIZATION_SHA = "3a0273bb92b8c060df8a214b1e0e7dd0e4b5df6bece86b7dea15197ca56ed0db"
E200_CONFIG_SHA = "ee95f0e94a667703660eb5aa7229f3c7ed50efd0b696b9778cea979639fd21a5"
CHECKPOINT_SHA = {
    0: "6a24871fb58e50735892a0159fd5d5bface9466de8f8445d5ebe8437e32fa071",
    1: "11219c17e3a5920cbe1b466d80ef692244acd3b622b4ad79091b97386536c2fe",
    2: "780cc15cef16ed66bedd0d49e2646e3259d0f6bf21cb6fe3cbc9b8610643cc9f",
}
GRID = (65, 65, 57)
NODE_COUNT = int(np.prod(GRID))
FORBIDDEN_BASENAMES = {
    "fs_test_volume.npy",
    "u_test_volume.npy",
    "test_iid",
    "sealed",
    "official100",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
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


def install_forbidden_open_guard() -> None:
    forbidden = {name.lower() for name in FORBIDDEN_BASENAMES}

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        candidate = arguments[0]
        if isinstance(candidate, (str, bytes, os.PathLike)):
            text = os.fsdecode(candidate).lower()
            basename = Path(text).name
            parts = {part for part in Path(text).parts}
            if (
                basename in forbidden
                or bool(parts & forbidden)
                or any(token in basename for token in ("test_iid", "sealed", "official100"))
            ):
                raise PermissionError(
                    f"FAIL-CLOSED: forbidden test/sealed artifact opened: {candidate}"
                )

    sys.addaudithook(audit)


def load_frozen_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("payload_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != actual or actual != NORMALIZATION_SHA:
        raise ValueError("frozen e200 train-only normalization mismatch")
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


def checkpoint_params(path: Path, expected_sha: str) -> tuple[Any, dict[str, Any]]:
    observed = file_sha256(path)
    if observed != expected_sha:
        raise ValueError(f"checkpoint SHA mismatch: {path} {observed}")
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if "params" not in payload:
        raise ValueError(f"checkpoint has no params: {path}")
    if payload.get("test_access") is not False:
        raise ValueError("checkpoint provenance does not prove test isolation")
    return payload["params"], {
        key: payload.get(key)
        for key in (
            "epoch",
            "seed",
            "selection_metric",
            "selection_value",
            "runner_sha",
            "config_sha",
            "data_sha",
        )
    }


def load_valid_manifest(path: Path) -> list[dict[str, Any]]:
    if file_sha256(path) != VALID_MANIFEST_SHA:
        raise ValueError("valid-only case manifest SHA mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 128:
        raise ValueError("valid-only manifest must contain exactly 128 cases")
    ids: set[str] = set()
    rows: set[int] = set()
    for case in cases:
        if case.get("split_role") != "valid_iid":
            raise ValueError("valid-only manifest contains a non-valid row")
        sample_id = str(case["sample_id"])
        truth_row = int(case["truth_row"])
        if sample_id in ids or truth_row in rows:
            raise ValueError("valid-only manifest has duplicate sample or truth row")
        ids.add(sample_id)
        rows.add(truth_row)
    return cases


def load_valid_truth(
    archive_path: Path, cases: list[dict[str, Any]], geometry: TrainingFullFieldGeometry
) -> dict[str, np.ndarray]:
    if file_sha256(archive_path) != FULL_FIELDS_SHA:
        raise ValueError("P1i full-field archive SHA mismatch")
    result: dict[str, np.ndarray] = {}
    with h5py.File(archive_path, "r") as archive:
        labels = archive["samples/deltaT_K"]
        for case in cases:
            sample_id = str(case["sample_id"])
            row = int(case["truth_row"])
            observed_id = archive["samples/sample_id"][row]
            if isinstance(observed_id, bytes):
                observed_id = observed_id.decode("utf-8")
            if str(observed_id) != sample_id:
                raise ValueError(f"truth row/sample mismatch for {sample_id}")
            observed_role = archive["samples/split_role"][row]
            if isinstance(observed_role, bytes):
                observed_role = observed_role.decode("utf-8")
            if str(observed_role) != "valid_iid":
                raise ValueError(f"refusing non-valid_iid truth row for {sample_id}")
            value = np.asarray(labels[row], dtype=np.float64).reshape(-1)
            if value.shape != (NODE_COUNT,) or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid truth shape/value for {sample_id}")
            result[sample_id] = value
    return result


def metric_row(
    *, sample_id: str, seed: int, prediction: np.ndarray, truth: np.ndarray,
    coords: np.ndarray, cv: np.ndarray, layer_id: np.ndarray, q: np.ndarray
) -> dict[str, Any]:
    evaluator = load_script("evaluate_v7_g2_p23_common_fullfield.py")
    row = evaluator.one_case_metrics(prediction, truth, cv)
    row.update({"sample_id": sample_id, "seed": int(seed), "model": "Heat3D_V7_e200_U_v2_P1i240825", "split": "valid_iid"})
    # Keep a content hash of the query geometry/fields in the receipt, not the
    # prediction.  This makes the same-output-resolution route auditable while
    # keeping large arrays out of Git.
    row["query_coords_sha256"] = array_sha256(coords)
    row["query_control_volume_sha256"] = array_sha256(cv)
    row["query_layer_id_sha256"] = array_sha256(layer_id)
    row["query_q_sha256"] = array_sha256(q)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--p1i-manifest", type=Path, required=True)
    parser.add_argument("--valid-case-manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--deeph-fs-train", type=Path, required=True)
    parser.add_argument("--deeph-labels-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--heat3d-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-valid", type=int, default=128)
    args = parser.parse_args()

    install_forbidden_open_guard()
    if args.seed not in CHECKPOINT_SHA:
        raise ValueError("seed must be 0, 1, or 2")
    if not 1 <= args.max_valid <= 128:
        raise ValueError("--max-valid must be in [1,128]")
    if file_sha256(args.p1i_manifest) != P1I_MANIFEST_SHA:
        raise ValueError("P1i source manifest SHA mismatch")
    if file_sha256(args.heat3d_config) != E200_CONFIG_SHA:
        raise ValueError("e200 config SHA mismatch")
    cases = load_valid_manifest(args.valid_case_manifest)[: args.max_valid]
    if args.max_valid != 128:
        # Partial runs are explicitly diagnostic and cannot be interpreted as
        # the complete Therm-FM-resolution comparison.
        partial = True
    else:
        partial = False
    stats = load_frozen_stats(args.normalization)

    p1i_dataset = Heat3DV6DualRobinDataset(
        args.dataset_root,
        args.p1i_manifest,
        include_roles={"valid_iid"},
    )
    if p1i_dataset.manifest.get("dataset_id") != CONTINUOUS_PHYSICS_V6_DATASET_ID:
        raise ValueError("unexpected P1i dataset identity")
    p1i_by_id = {example.sample_id: example for example in p1i_dataset.samples}
    if set(p1i_by_id) != {str(case["sample_id"]) for case in load_valid_manifest(args.valid_case_manifest)}:
        raise ValueError("P1i valid sample IDs do not match frozen valid-only manifest")
    valid_examples = [p1i_by_id[str(case["sample_id"])] for case in cases]

    # Fit the already-frozen e200 global context standardizer from DeepOHeat
    # train examples only.  No P1i target enters this operation.
    deeph_loader = load_script("load_v7_g2_p6_deepoheat_v1_compact.py")
    deeph_eval = load_script("evaluate_v7_g2_p14_heat3d_v1_fullfield.py")
    deeph_support = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    converter = load_script("convert_v7_g2_semiconductor_case.py")
    deeph_train = deeph_loader.CompactDeepOHeatV1Dataset(
        fs_train=args.deeph_fs_train, labels_root=args.deeph_labels_root, role="train"
    )
    deeph_mesh = deeph_support.mesh_arrays()
    deeph_train_examples = deeph_eval.build_examples(
        dataset=deeph_train,
        role="train",
        full_coords=deeph_mesh["coords"],
        full_cv=deeph_mesh["control_volume"],
        layer_id=deeph_mesh["layer_id"],
        converter=converter,
    )
    p1i_context_helper = load_script("run_v7_g2_p5_heat3d_v1_dual_output_smoke.py")
    # attach_input_contexts only needs examples to fit/encode the context; an
    # empty batch list avoids building a second native graph solely for context.
    from rigno.heat3d_training.p1i import attach_input_contexts  # noqa: E402

    config = json.loads(args.heat3d_config.read_text(encoding="utf-8"))
    context = attach_input_contexts(
        [],
        deeph_train_examples,
        deeph_train_examples + valid_examples,
        config["model"],
    )
    model_config = p1i_context_helper.resolve_model_config(
        config["model"], tuple(stats["feature_names"])
    )
    from rigno.models.rigno import RIGNO  # noqa: E402

    model = RIGNO(**model_config)
    params, checkpoint_meta = checkpoint_params(args.checkpoint, CHECKPOINT_SHA[args.seed])
    session = p1i_context_helper.build_session(
        model=model,
        params=params,
        stats=stats,
        config=config,
        context_standardizer=context["standardizer"],
        seed=args.seed,
    )

    # Shared P1i mesh is label-independent.  Only the explicitly listed valid
    # rows below are read from deltaT_K.
    geometry = load_full_field_geometry(args.full_fields)
    truth_by_id = load_valid_truth(args.full_fields, cases, geometry)
    runtime_geometry = SimpleNamespace(
        coords=np.asarray(geometry.coords, dtype=np.float64),
        control_volume=np.asarray(geometry.control_volume, dtype=np.float64),
        layer_id=np.asarray(geometry.layer_id, dtype=np.int32),
    )
    runtime = UHighNRuntime.from_session(session, runtime_geometry)

    rows: list[dict[str, Any]] = []
    case_audits: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (case_meta, anchor) in enumerate(zip(cases, valid_examples, strict=True), start=1):
        sample_id = str(case_meta["sample_id"])
        # Materialize k/q from the case definition and frozen shared geometry;
        # no temperature label is consulted for model inputs.
        k_full, q_full, flags = materialize_full_input_fields(meta=anchor.meta, geometry=geometry)
        full_support = SupportArtifact.from_arrays(
            selected_indices=np.arange(NODE_COUNT, dtype=np.int64),
            operator_control_volume=np.asarray(geometry.control_volume, dtype=np.float64),
            k_xyz=np.asarray(k_full, dtype=np.float64),
            q_W_m3=np.asarray(q_full, dtype=np.float64),
            layer_id=np.asarray(geometry.layer_id, dtype=np.int32),
            path="<derived-p1i-full-query>",
            sha256=array_sha256(np.arange(NODE_COUNT, dtype=np.int32)),
        )
        case_started = time.perf_counter()
        query_case = runtime.build_case(
            anchor,
            NODE_COUNT,
            support=full_support,
            native_edge_targets=None,
            query_edge_targets=None,
        )
        output = runtime.apply(query_case)
        # Explicitly synchronize the result before host conversion and timing.
        jax.block_until_ready(output["deltaT_hat"])
        prediction = np.asarray(output["deltaT_hat"], dtype=np.float64)[0, 0, :, 0]
        if prediction.shape != (NODE_COUNT,) or not np.all(np.isfinite(prediction)):
            raise FloatingPointError(f"nonfinite/wrong-shape U-v2 output for {sample_id}")
        truth = truth_by_id[sample_id]
        row = metric_row(
            sample_id=sample_id,
            seed=args.seed,
            prediction=prediction,
            truth=truth,
            coords=geometry.coords,
            cv=geometry.control_volume,
            layer_id=geometry.layer_id,
            q=q_full,
        )
        rows.append(row)
        case_audits.append(
            {
                "sample_id": sample_id,
                "support_count": 1024,
                "query_count": NODE_COUNT,
                "query_k_sha256": array_sha256(k_full),
                "query_q_sha256": array_sha256(q_full),
                "query_boundary_flags_sha256": array_sha256(flags),
                "support_selector": "stored P1i 1024 support",
                "u_v2_audit": query_case.audit,
                "wall_seconds": float(time.perf_counter() - case_started),
            }
        )
        del query_case, output, full_support, k_full, q_full, flags, prediction
        print(f"[p23-r2 U-v2] {index}/{len(cases)} {sample_id}", flush=True)

    evaluator = load_script("evaluate_v7_g2_p23_common_fullfield.py")
    summary = evaluator.summarize(rows, require_three_seeds=False)
    repo_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = {
        "schema_version": "heat3d_v7_g2_p23_r2_heat3d_e200_p1i_u_v2_240825_receipt_v1",
        "status": "PASS_VALID_ONLY_U_V2_240825" if len(rows) == args.max_valid else "PARTIAL_DIAGNOSTIC_PASS",
        "partial_diagnostic": partial,
        "seed": args.seed,
        "valid_cases_evaluated": len(rows),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
            "metadata": checkpoint_meta,
        },
        "runner": {
            "script": "scripts/evaluate_v7_g2_p23_r2_heat3d_e200_p1i_u_v2_240825.py",
            "repo_commit_sha": repo_sha,
        },
        "contracts": {
            "p1i_manifest_sha256": P1I_MANIFEST_SHA,
            "valid_case_manifest_sha256": VALID_MANIFEST_SHA,
            "full_fields_sha256": FULL_FIELDS_SHA,
            "normalization_payload_sha256": NORMALIZATION_SHA,
            "heat3d_config_sha256": E200_CONFIG_SHA,
            "test_iid_accessed": False,
            "sealed_accessed": False,
            "deepoheat_official100_accessed": False,
            "training_started": False,
        },
        "domain": {
            "grid": list(GRID),
            "count": NODE_COUNT,
            "temperature_space": "deltaT_K",
            "conditioning_count": 1024,
            "u_v2_mode": "direct_query_dense_inference",
            "interpolation_or_resampling": False,
            "external_padding": False,
            "learned_parameters_added": 0,
        },
        "runtime": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "wall_seconds": float(time.perf_counter() - started),
            "valid_case_wall_seconds_mean": float(np.mean([row["wall_seconds"] for row in case_audits])),
        },
        "evaluation": summary,
        "case_audits": case_audits,
        "interpretation": {
            "label": "same-output-resolution zero-shot transfer diagnostic",
            "not_same_domain_training_comparison": True,
            "not_same_information_budget": True,
            "prediction_arrays_persisted": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
