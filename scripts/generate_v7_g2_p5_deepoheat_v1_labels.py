#!/usr/bin/env python3
"""Generate frozen DeepOHeat-v1 768/128 solver labels outside Git.

This generator is resumable and fail-closed.  It reads only the released
100,000 training input-function pool at the preregistered train/valid indices;
the official 100 test fields are never opened.  Every stored field and
extraction is hashed after writing.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_FS_TRAIN_SHA256 = "a39a4f51e853f9114d86feb88f74553914b2bfc68ab1c553a3a31df25893fff7"


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def decode_indices(payload: dict[str, Any], role: str) -> np.ndarray:
    row = payload["roles"][role]
    values = np.frombuffer(base64.b64decode(row["indices_base64"]), dtype="<u4").astype(np.int64)
    if len(values) != int(row["count"]):
        raise ValueError(f"{role} subset count mismatch")
    return values


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-train", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, default=768)
    parser.add_argument("--limit-valid", type=int, default=128)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not str(output).startswith(("/tmp/", "/private/tmp/")):
        raise ValueError("large labels must remain under /tmp")
    if file_sha256(args.fs_train) != OFFICIAL_FS_TRAIN_SHA256:
        raise ValueError("official fs_train_volume.npy SHA mismatch")
    subset = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    if subset["selection"]["accuracy_or_temperature_observed"] is not False:
        raise ValueError("subset manifest is not temperature/accuracy blind")
    role_indices = {
        "train": decode_indices(subset, "train")[: args.limit_train],
        "valid": decode_indices(subset, "valid")[: args.limit_valid],
    }
    output.mkdir(parents=True, exist_ok=True)
    solver_module = load_script("run_v7_g2_deepoheat_v1_solver_fidelity.py")
    support_module = load_script("prepare_v7_g2_p5_deepoheat_v1_support.py")
    solver = solver_module.OfficialVolumetricFDSolver()
    fs_train = np.load(args.fs_train, mmap_mode="r", allow_pickle=False)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for role in ("train", "valid"):
        for ordinal, source_index in enumerate(role_indices[role]):
            source_index = int(source_index)
            sample_id = f"dhv1_volume_{role}_{source_index:05d}"
            directory = output / role / sample_id
            directory.mkdir(parents=True, exist_ok=True)
            metadata_path = directory / "receipt.json"
            if metadata_path.is_file():
                row = json.loads(metadata_path.read_text(encoding="utf-8"))
                for key in ("full_reference", "support_indices", "support_target", "u_slice_z015"):
                    artifact = directory / row["artifacts"][key]["file"]
                    if not artifact.is_file() or canonical_array_sha256(np.load(artifact, mmap_mode="r")) != row["artifacts"][key]["sha256"]:
                        raise ValueError(f"resume artifact drift: {sample_id}/{key}")
                rows.append(row)
                continue
            power = np.asarray(fs_train[source_index])
            support, strata, support_audit = support_module.select_support(
                power, source_index=source_index, role=role
            )
            field_u, solve = solver.solve(np.asarray(power, dtype=np.float64))
            if solve["gmres_info"] != 0 or solve["relative_linear_residual"] > 1.0e-8:
                raise RuntimeError(f"FAIL-CLOSED linear solve: {sample_id}")
            delta_t = np.asarray(25.0 * (field_u - 0.2), dtype=np.float32)
            support_target = delta_t.reshape(-1)[support].astype(np.float32)
            slice_truth = delta_t[:, :, 15].reshape(-1).astype(np.float32)
            artifacts = {
                "full_reference": ("deltaT_full_K.npy", delta_t),
                "support_indices": ("support_indices.npy", support.astype(np.int32)),
                "support_target": ("deltaT_support1024_K.npy", support_target),
                "u_slice_z015": ("deltaT_z015_10201_K.npy", slice_truth),
            }
            artifact_rows = {}
            for key, (filename, value) in artifacts.items():
                path = directory / filename
                atomic_save(path, value)
                stored = np.load(path, mmap_mode="r", allow_pickle=False)
                artifact_rows[key] = {
                    "file": filename,
                    "shape": list(stored.shape),
                    "dtype": str(stored.dtype),
                    "sha256": canonical_array_sha256(stored),
                }
            row = {
                "sample_id": sample_id,
                "role": role,
                "role_ordinal": ordinal,
                "source_index": source_index,
                "source_input_sha256": canonical_array_sha256(power),
                "support_sha256": support_audit["support_indices_sha256"],
                "support_strata_sha256": hashlib.sha256("\n".join(strata).encode()).hexdigest(),
                "linear_relative_residual": solve["relative_linear_residual"],
                "gmres_iterations": solve["iterations"],
                "solve_wall_seconds": solve["wall_seconds"],
                "artifacts": artifact_rows,
            }
            metadata_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rows.append(row)
            if len(rows) % 16 == 0:
                print(f"generated_or_verified={len(rows)} elapsed_s={time.perf_counter()-started:.1f}", flush=True)
    expected = int(args.limit_train) + int(args.limit_valid)
    if len(rows) != expected:
        raise AssertionError("label row count mismatch")
    ordered_row_hashes = [hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest() for row in rows]
    import pyamg
    receipt = {
        "schema_version": "heat3d_v7_g2_p5_deepoheat_v1_label_generation_v1",
        "status": "PASS_COMPLETE_768_128" if (args.limit_train, args.limit_valid) == (768, 128) else "PASS_BOUNDED_PARTIAL",
        "roles": {"train": int(args.limit_train), "valid": int(args.limit_valid)},
        "frozen_subset_manifest_sha256": file_sha256(args.subset_manifest),
        "fs_train_file_sha256": OFFICIAL_FS_TRAIN_SHA256,
        "solver": {
            "implementation": "official_hybrid_solver_notebook_matrix_CPU_port",
            "mesh": [101, 101, 56],
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pyamg": pyamg.__version__,
            "rtol": 1.0e-10, "peak_process_rss_bytes": peak_rss_bytes(),
        },
        "u_strategy_domain": {"count": 10201, "z_index": 15, "z": 0.15, "domain": "101x101 source-layer top slice"},
        "ordered_row_receipt_hashes_sha256": hashlib.sha256("\n".join(ordered_row_hashes).encode()).hexdigest(),
        "rows": rows,
        "total_wall_seconds_including_resume_verification": time.perf_counter() - started,
        "official_100_test_fields_accessed": False,
        "formal_or_long_training_started": False,
    }
    receipt_path = output / "label_generation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
