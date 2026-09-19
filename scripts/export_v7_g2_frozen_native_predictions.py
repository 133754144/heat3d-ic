#!/usr/bin/env python3
"""Export valid-only native-1024 predictions from frozen external checkpoints.

This is an inference-only utility for the final evidence lock.  It refuses
train/test/sealed roles, never constructs an optimizer, and never writes into
the formal output directories.  The resulting NPZ is intended for an ignored
staging directory and is consumed by the resolution-agnostic final evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", choices=("GINO", "Transolver"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("FAIL_CLOSED: frozen external prediction export requires CUDA")
    sys.path.insert(0, str(args.repo.resolve()))
    sys.path.insert(0, str(args.upstream_root.resolve()))
    from scripts.run_v7_g2_p1i_external_formal import (  # noqa: E402
        P1iRoleDataset,
        build_gino,
        build_transolver,
        latent_queries,
        load_stats,
        normalize,
        predict,
    )

    device = torch.device("cuda")
    valid = P1iRoleDataset(args.dataset_root, args.dataset_manifest, "valid_iid")
    stats = load_stats(args.statistics)
    if args.model == "GINO":
        model = build_gino(0.15, 0.033, use_open3d=True, use_torch_scatter=True).to(device)
        grid = latent_queries(32).to(device)
        if not model.gno_in.neighbor_search.use_open3d or not model.gno_out.neighbor_search.use_open3d:
            raise RuntimeError("GINO did not use Open3D FixedRadiusSearch")
        if not model.gno_in.integral_transform.use_torch_scatter or not model.gno_out.integral_transform.use_torch_scatter:
            raise RuntimeError("GINO did not use torch-scatter reduction")
    else:
        model = build_transolver(args.upstream_root.resolve()).to(device)
        grid = None
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model" not in payload:
        raise ValueError("checkpoint has no model state")
    model.load_state_dict(payload["model"])
    model.eval()
    sample_ids: list[str] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for index in range(len(valid)):
            row = valid[index]
            coords, features, _target, _target_n, local = normalize(row, stats, device)
            pred_n = predict(args.model, model, coords, features, grid)
            pred = (pred_n * local["target_std"] + local["target_mean"]).detach().cpu().numpy().reshape(-1)
            if pred.shape != (1024,) or not np.isfinite(pred).all():
                raise ValueError(f"invalid prediction shape/finite status for {row['sample_id']}")
            sample_ids.append(str(row["sample_id"]))
            predictions.append(pred.astype(np.float32, copy=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray(sample_ids),
        prediction_deltaT_K=np.stack(predictions),
        split=np.asarray("valid_iid"),
        model=np.asarray(args.model),
        seed=np.asarray(args.seed, dtype=np.int32),
        checkpoint_sha256=np.asarray(sha256(args.checkpoint)),
        domain_id=np.asarray("registered_support_1024"),
    )
    print(json.dumps({
        "status": "COMPLETE_VALID_ONLY_INFERENCE",
        "model": args.model,
        "seed": args.seed,
        "valid_cases": len(valid),
        "checkpoint_sha256": sha256(args.checkpoint),
        "prediction_sha256": sha256(args.output),
        "test_iid_read": False,
        "sealed_read": False,
        "deepoheat_official100_read": False,
        "optimizer_constructed": False,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
