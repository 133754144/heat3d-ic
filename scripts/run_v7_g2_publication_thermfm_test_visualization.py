#!/usr/bin/env python3
"""Fixed test_iid visualization-only Therm-FM inference.

The sample id is supplied by the pre-frozen input-only figure-selection
receipt.  This script loads a frozen Poseidon-T checkpoint, performs one
forward pass, and writes physical-K arrays for plotting.  It never trains,
selects a checkpoint, or computes a selection metric.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


GRID = (65, 65, 57)
NODE_COUNT = int(np.prod(GRID))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--replay-script", type=Path, required=True)
    parser.add_argument("--therm-repo", type=Path, required=True)
    parser.add_argument("--poseidon-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--full-field-archive", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise SystemExit("FAIL_CLOSED: visualization inference requires CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adapter = load_module(args.adapter, "v7_g2_p22_thermfm_adapter_testviz")
    replay = load_module(args.replay_script, "v7_g2_p23_thermfm_replay_testviz")
    observed_checkpoint_sha = sha256_file(args.checkpoint)
    if observed_checkpoint_sha != args.checkpoint_sha256:
        raise SystemExit(f"checkpoint SHA mismatch: {observed_checkpoint_sha}")
    source = adapter.P1iThermFMSource(
        args.split_manifest,
        args.sample_root,
        args.full_field_archive,
        roles=("test_iid",),
    )
    if args.sample_id not in source.ids_by_role["test_iid"]:
        raise ValueError(f"sample is not in frozen test_iid role: {args.sample_id}")
    inputs = source.raw_input(args.sample_id)
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    input_mean = np.asarray(stats["input"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
    input_std = np.asarray(stats["input"]["std"], dtype=np.float32).reshape(-1, 1, 1)
    output_mean = np.asarray(stats["output"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
    output_std = np.asarray(stats["output"]["std"], dtype=np.float32).reshape(-1, 1, 1)
    input_std = np.where(input_std > 0.0, input_std, 1.0)
    output_std = np.where(output_std > 0.0, output_std, 1.0)
    model = replay.initialize_model(args.therm_repo, args.poseidon_dir, torch.device("cuda"))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError("Therm-FM checkpoint lacks model state")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray((inputs - input_mean) / input_std))[None].cuda()
        output = model(pixel_values=x).output
        torch.cuda.synchronize()
        zxy = output.detach().cpu().numpy()[0].astype(np.float64) * output_std + output_mean
    prediction = np.asarray(zxy[...].transpose(1, 2, 0), dtype=np.float32).reshape(-1)
    with h5py.File(args.full_field_archive, "r") as archive:
        ids = [v.decode() if isinstance(v, bytes) else str(v) for v in archive["samples/sample_id"][:]]
        roles = [v.decode() if isinstance(v, bytes) else str(v) for v in archive["samples/split_role"][:]]
        row = ids.index(args.sample_id)
        if roles[row] != "test_iid":
            raise ValueError("full-field role mismatch")
        truth = np.asarray(archive["samples/deltaT_K"][row], dtype=np.float32).reshape(-1)
        coords = np.asarray(archive["shared/coords_m"][:], dtype=np.float64)
        layer_id = np.asarray(archive["shared/layer_id"][:], dtype=np.int32)
    if prediction.shape != (NODE_COUNT,) or truth.shape != (NODE_COUNT,):
        raise ValueError("native-65 output/truth shape mismatch")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("non-finite test visualization artifact")
    np.savez_compressed(
        args.output,
        sample_id=np.asarray(args.sample_id),
        role=np.asarray("test_iid"),
        coords=coords,
        layer_id=layer_id,
        truth=truth,
        prediction=prediction,
    )
    receipt = {
        "schema_version": "heat3d_v7_g2_publication_thermfm_test_visualization_inference_v1",
        "status": "PASS_VISUALIZATION_ONLY_TEST_INFERENCE",
        "visualization_only": True,
        "used_for_model_selection": False,
        "used_for_claim_or_color_policy": False,
        "sealed_access": False,
        "sample_id": args.sample_id,
        "role": "test_iid",
        "checkpoint_sha256": observed_checkpoint_sha,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "normalization_sha256": sha256_file(args.stats),
        "full_field_archive_sha256": sha256_file(args.full_field_archive),
        "adapter_sha256": sha256_file(args.adapter),
        "grid": list(GRID),
        "input_layout": "layer-major zxy channels from deterministic case-definition rasterization",
        "output_layout": "canonical x_y_z flattened in shared full-field order",
        "temperature_space": "deltaT_K",
        "output_sha256": sha256_file(args.output),
        "prediction_sha256": sha256_array(prediction),
        "truth_sha256": sha256_array(truth),
        "coords_sha256": sha256_array(coords),
        "layer_id_sha256": sha256_array(layer_id),
        "elapsed_seconds": time.perf_counter() - started,
        "test_iid_read": True,
        "training_started": False,
        "optimizer_called": False,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "runner_script_sha256": sha256_file(Path(__file__)),
        "runner_repo_commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.therm_repo, text=True
        ).strip(),
        "command": sys.argv,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
