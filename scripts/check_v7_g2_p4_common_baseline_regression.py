#!/usr/bin/env python3
"""Minimal G2-P4 regression for frozen GINO and Transolver contracts.

GINO is constructed with the formal asymmetric radii and receives one
train-only-normalized P1i tensor; no optimizer step is repeated. Transolver
performs exactly one CPU optimizer step with prediction and target decoded to
physical delta-T before upstream TestLoss. Only train role data are accepted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_v7_g2_p1_local_qualification import (  # noqa: E402
    build_gino,
    build_transolver,
    latent_queries,
    load_manifest,
    load_sample,
    relative_l2,
)


def official_testloss(path: Path):
    spec = importlib.util.spec_from_file_location("transolver_official_testloss", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TestLoss(size_average=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--gino-root", type=Path, required=True)
    parser.add_argument("--transolver-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not str(args.receipt.resolve()).startswith(("/tmp/", "/private/tmp/")):
        raise ValueError("regression receipt must remain under /tmp")

    _manifest, rows = load_manifest(args.manifest)
    coords_np, features_np, target_np, meta = load_sample(
        args.subset, rows, "v6p1if1_0000", "train"
    )
    if meta["role"] != "train":
        raise ValueError("only train data may enter this regression")
    stats_payload = json.loads(args.statistics.read_text(encoding="utf-8"))
    if stats_payload["fit_role"] != "train_only":
        raise ValueError("normalization statistics are not train-only")
    values = stats_payload["statistics"]
    cmin = torch.tensor(values["coordinate_min"], dtype=torch.float32).view(1, 1, 3)
    cmax = torch.tensor(values["coordinate_max"], dtype=torch.float32).view(1, 1, 3)
    fmean = torch.tensor(values["feature_mean"], dtype=torch.float32).view(1, 1, 11)
    fstd = torch.tensor(values["feature_std"], dtype=torch.float32).view(1, 1, 11)
    fstd = torch.where(fstd > 0, fstd, torch.ones_like(fstd))
    ymean = torch.tensor(values["target_mean"], dtype=torch.float32).view(1, 1, 1)
    ystd = torch.tensor(values["target_std"], dtype=torch.float32).view(1, 1, 1)
    coords = torch.from_numpy(coords_np).unsqueeze(0)
    features = torch.from_numpy(features_np).unsqueeze(0)
    target = torch.from_numpy(target_np).unsqueeze(0)
    coords_n = (coords - cmin) / torch.clamp(cmax - cmin, min=1.0e-12)
    features_n = (features - fmean) / fstd
    target_n = (target - ymean) / ystd
    if not all(torch.isfinite(x).all() for x in (coords_n, features_n, target_n)):
        raise FloatingPointError("formal normalization produced non-finite tensors")

    sys.path.insert(0, str(args.gino_root.resolve()))
    gino = build_gino(in_radius=0.15, out_radius=0.033)
    gino_contract = {
        "status": "PASS_FORMAL_CONSTRUCTION_AND_INPUT_NORMALIZATION",
        "input_radius": float(gino.in_gno_radius),
        "output_radius": float(gino.out_gno_radius),
        "latent_grid_shape": list(latent_queries(32).shape),
        "input_shape": list(features_n.shape),
        "normalization_payload_sha256": stats_payload["payload_sha256"],
        "optimizer_step_repeated": False,
    }
    if gino_contract["input_radius"] != 0.15 or gino_contract["output_radius"] != 0.033:
        raise AssertionError("GINO asymmetric radius was not applied")

    torch.manual_seed(0)
    transolver = build_transolver(args.transolver_root.resolve())
    optimizer = torch.optim.AdamW(
        transolver.parameters(), lr=1.0e-3, weight_decay=1.0e-5
    )
    before = [parameter.detach().clone() for parameter in transolver.parameters()]
    optimizer.zero_grad(set_to_none=True)
    pred_n = transolver(coords_n, features_n)
    pred_physical = pred_n * ystd + ymean
    target_physical = target_n * ystd + ymean
    loss = relative_l2(pred_physical, target_physical)
    upstream_loss = official_testloss(
        args.transolver_root
        / "PDE-Solving-StandardBenchmark"
        / "utils"
        / "testloss.py"
    )(pred_physical, target_physical)
    if not torch.equal(loss, upstream_loss):
        raise AssertionError("decoded loss differs from official TestLoss(size_average=False)")
    loss.backward()
    unclipped_norm = torch.nn.utils.clip_grad_norm_(transolver.parameters(), 0.1)
    optimizer.step()
    changed = any(not torch.equal(a, b.detach()) for a, b in zip(before, transolver.parameters()))
    transolver_contract = {
        "status": "PASS_ONE_DECODED_PHYSICAL_SPACE_OPTIMIZER_STEP",
        "loss_semantics": "decode_prediction_and_target_to_physical_deltaT_K_then_official_TestLoss_size_average_false",
        "official_TestLoss_exact_equal": True,
        "loss_finite": bool(torch.isfinite(loss)),
        "gradient_norm_before_clip": float(unclipped_norm),
        "gradient_clip_norm": 0.1,
        "parameters_changed": changed,
        "optimizer_steps": 1,
        "parameter_count": sum(p.numel() for p in transolver.parameters() if p.requires_grad),
    }
    status = "PASS" if changed and transolver_contract["loss_finite"] else "FAIL"
    receipt = {
        "schema_version": "heat3d_v7_g2_p4_common_baseline_regression_v1",
        "status": status,
        "host": "local_mac_cpu",
        "sample_id": meta["sample_id"],
        "sample_role": meta["role"],
        "formal_accuracy_observed": False,
        "p1i_test_or_sealed_access": False,
        "formal_or_long_training_started": False,
        "GINO": gino_contract,
        "Transolver": transolver_contract,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
