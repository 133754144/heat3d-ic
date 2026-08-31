#!/usr/bin/env python3
"""Materialize Heat3D-on-v1 train-only normalization into a small receipt."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget
from rigno.heat3d_v1_normalization import legacy_train_only_stats
from rigno.heat3d_v6_dataset import V6_DUAL_ROBIN_CONDITION_FEATURES, V6DualRobinExample


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name; spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--fs-train", type=Path, required=True); parser.add_argument("--subset-manifest", type=Path, required=True); parser.add_argument("--labels-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if not str(args.output.resolve()).startswith(("/tmp/", "/private/tmp/")): raise ValueError("materialization receipt must first be written under /tmp")
    subset = json.loads(args.subset_manifest.read_text()); row = subset["roles"]["train"]; indices = np.frombuffer(base64.b64decode(row["indices_base64"]), dtype="<u4")
    if len(indices) != 768: raise ValueError("formal train count mismatch")
    converter = load_script("convert_v7_g2_semiconductor_case.py"); fs_train = np.load(args.fs_train, mmap_mode="r", allow_pickle=False); examples = []
    for source_index_u32 in indices:
        source_index = int(source_index_u32); sample_id = f"dhv1_volume_train_{source_index:05d}"; directory = args.labels_root / "train" / sample_id
        support = np.asarray(np.load(directory / "support_indices.npy"), dtype=np.int64); target = np.asarray(np.load(directory / "deltaT_support1024_K.npy"), dtype=np.float64)
        arrays = converter.volume_v1_arrays(np.asarray(fs_train[source_index])); coords = np.asarray(arrays["coords"])[support].astype(np.float64); features = np.asarray(arrays["features"])[support].astype(np.float64)
        examples.append(V6DualRobinExample(sample_id=sample_id, condition=V1SteadyConditionInput(coords=coords, condition_features=features, condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES, k_encoding_mode="diag3"), target=V1SteadyTarget(target_u=(298.15 + target).reshape(-1, 1)), meta={"split": "train", "physics": {"ambient_K": 298.15, "footprint_m": [1.0,1.0], "layers_bottom_to_top": [{"name":"lower","thickness_m":0.1,"k_W_mK":2.0},{"name":"upper","thickness_m":0.45,"k_W_mK":0.1}]}, "package_total_power_W": 1.0, "v6_adapter": {"dataset_id":"deepoheat_v1_volumetric_method_native_1024","manifest_split_role":"train","group_id":sample_id,"reference_temperature_K":298.15,"top_T_inf_K":298.15,"bottom_T_inf_K":298.15,"bottom_boundary_semantics":"robin_not_dirichlet"}}, operator_point_weights=np.ones(1024, dtype=np.float64)))
    stats = legacy_train_only_stats(examples)
    statistics = {key: (list(value) if isinstance(value, tuple) else np.asarray(value).tolist()) for key, value in stats.items()}
    payload = {"schema_version": "heat3d_v7_g2_p5_deepoheat_v1_train_only_normalization_v1", "fit_role": "train_only_768", "support_count_per_case": 1024, "subset_manifest_sha256": sha256(args.subset_manifest), "label_generation_receipt_sha256": sha256(args.labels_root / "label_generation_receipt.json"), "statistics": statistics, "valid_or_test_used_to_fit": False}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(); payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({key:value for key,value in payload.items() if key != "statistics"}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
