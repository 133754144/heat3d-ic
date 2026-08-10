#!/usr/bin/env python3
"""Compare two graph-policy timing/equivalence runs without new inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
from rigno.heat3d_graph_cache import load_metadata  # noqa: E402


def prediction_diff(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "max_abs_K": float(np.max(np.abs(delta))),
        "rmse_K": float(math.sqrt(np.mean(np.square(delta)))),
        "mean_abs_K": float(np.mean(np.abs(delta))),
    }


def real_edges(value: object, n_sender: int, n_receiver: int) -> np.ndarray:
    if value is None:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.asarray(value)[0].astype(np.int64)
    return edges[(edges[:, 0] < n_sender) & (edges[:, 1] < n_receiver)]


def edge_diff(left: object, right: object, n_sender: int, n_receiver: int) -> dict[str, int | bool]:
    a = real_edges(left, n_sender, n_receiver)
    b = real_edges(right, n_sender, n_receiver)
    sa, sb = set(map(tuple, a.tolist())), set(map(tuple, b.tolist()))
    return {
        "left_count": len(a), "right_count": len(b),
        "left_only": len(sa - sb), "right_only": len(sb - sa),
        "set_exact": sa == sb,
    }


def metrics(
    predictions: np.ndarray, sample_ids: list[str], *, full_fields: Path,
    preflight: dict,
) -> dict:
    with h5py.File(full_fields, "r") as archive:
        archive_ids = [value.decode() if isinstance(value, bytes) else str(value) for value in archive["samples/sample_id"][:]]
        lookup = {sample_id: index for index, sample_id in enumerate(archive_ids)}
        coords = np.asarray(archive["shared/coords_m"][:], dtype=np.float64)
        cv = np.asarray(archive["shared/control_volume_m3"][:], dtype=np.float64)
        layer = np.asarray(archive["shared/layer_id"][:], dtype=np.int32)
        rows = []
        for sample_id, prediction in zip(sample_ids, predictions, strict=True):
            truth = np.asarray(archive["samples/deltaT_K"][lookup[sample_id]], dtype=np.float64)
            physics_path = Path(next(row for row in preflight["samples"] if row["sample_id"] == sample_id)["physics_cache_file"])
            with np.load(physics_path, allow_pickle=False) as physics:
                q = np.asarray(physics["q_W_m3"], dtype=np.float64)
            rows.append(highn._metric_row(prediction, truth, cv, coords, layer, q))
    return qualification.metric_accumulate(rows, full=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = json.loads((args.reference_dir / "result.json").read_text())
    right = json.loads((args.candidate_dir / "result.json").read_text())
    with np.load(args.reference_dir / "predictions.npz", allow_pickle=False) as a, np.load(args.candidate_dir / "predictions.npz", allow_pickle=False) as b:
        ids_a = [str(value) for value in np.asarray(a["sample_ids"]).tolist()]
        ids_b = [str(value) for value in np.asarray(b["sample_ids"]).tolist()]
        if ids_a != ids_b:
            if ids_a[: len(ids_b)] != ids_b:
                raise RuntimeError("sample order differs")
            ids_a = ids_a[: len(ids_b)]
            support_a = np.asarray(a["support_deltaT_K"])[: len(ids_b)]
            full_a = np.asarray(a["full_deltaT_K"])[: len(ids_b)]
        else:
            support_a = np.asarray(a["support_deltaT_K"])
            full_a = np.asarray(a["full_deltaT_K"])
        support_b, full_b = np.asarray(b["support_deltaT_K"]), np.asarray(b["full_deltaT_K"])
    graph_rows = []
    for row_a, row_b in zip(
        left["graph_metadata_artifacts"][: len(right["graph_metadata_artifacts"])],
        right["graph_metadata_artifacts"], strict=True,
    ):
        meta_a = load_metadata(Path(row_a["path"]))[0]
        meta_b = load_metadata(Path(row_b["path"]))[0]
        n_p = int(np.asarray(meta_a.x_pnodes_inp).shape[1] - 1)
        n_r = int(np.asarray(meta_a.x_rnodes).shape[1] - 1)
        r2p_a = meta_a.r2p_edge_indices
        r2p_b = meta_b.r2p_edge_indices
        if r2p_a is None:
            r2p_a = np.flip(np.asarray(meta_a.p2r_edge_indices), axis=-1)
        if r2p_b is None:
            r2p_b = np.flip(np.asarray(meta_b.p2r_edge_indices), axis=-1)
        graph_rows.append({
            "sample_id": row_a["sample_id"],
            "p2r": edge_diff(meta_a.p2r_edge_indices, meta_b.p2r_edge_indices, n_p, n_r),
            "r2r": edge_diff(meta_a.r2r_edge_indices, meta_b.r2r_edge_indices, n_r, n_r),
            "r2p": edge_diff(r2p_a, r2p_b, n_r, n_p),
        })
    preflight = json.loads(args.preflight.read_text())
    metrics_a = metrics(full_a, ids_a, full_fields=args.full_fields, preflight=preflight)
    metrics_b = metrics(full_b, ids_b, full_fields=args.full_fields, preflight=preflight)
    keys = [
        "point_global_true_rms_relative_rmse_pct", "raw_cv_weighted_rmse_K",
        "source_rmse_K", "peak_rmse_K", "interface_drop_rmse_K",
    ]
    payload = {
        "schema_version": "heat3d_v6_p1i_graph_optimization_equivalence_v1",
        "status": "passed" if all(
            row[field]["set_exact"] for row in graph_rows for field in ("p2r", "r2r", "r2p")
        ) else "failed_edge_difference",
        "reference_mode": left.get("optimization_mode"),
        "candidate_mode": right.get("optimization_mode"),
        "sample_count": len(ids_a),
        "support_prediction_difference": prediction_diff(support_a, support_b),
        "full_prediction_difference": prediction_diff(full_a, full_b),
        "metric_reference": {key: metrics_a[key] for key in keys},
        "metric_candidate": {key: metrics_b[key] for key in keys},
        "metric_delta_candidate_minus_reference": {key: metrics_b[key] - metrics_a[key] for key in keys},
        "edge_summary": {
            field: {
                "left_only_total": sum(row[field]["left_only"] for row in graph_rows),
                "right_only_total": sum(row[field]["right_only"] for row in graph_rows),
                "all_samples_exact": all(row[field]["set_exact"] for row in graph_rows),
            }
            for field in ("p2r", "r2r", "r2p")
        },
        "role_contract": {"training": False, "test": False, "sealed": False, "valid_iid": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "sample_count": payload["sample_count"]}))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
