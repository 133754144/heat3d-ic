#!/usr/bin/env python3
"""Read-only train/valid audit for bugged V1 and sparse-safe V2 QK features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EPS = 1.0e-12
ROLE_ALLOWLIST = ("train", "valid_iid")
FEATURE_NAMES = {
    "bugged_v1": (
        "log1p_q_relative",
        "log_inverse_kz_relative",
        "log1p_q_inverse_kz_relative",
        "q_high_inverse_kz_overlap",
        "source_z_normalized",
    ),
    "sparse_safe_v2": (
        "log1p_q_relative",
        "log_inverse_kz_relative",
        "log1p_q_inverse_kz_relative",
        "source_present_fraction",
        "region_z_normalized",
    ),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", type=Path, default=Path("data/heat3d_v4_p5_clean_nohard_v0")
    )
    parser.add_argument(
        "--split-map",
        type=Path,
        default=Path(
            "configs/heat3d_v4/"
            "candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json"
        ),
    )
    parser.add_argument("--roles", default="train,valid_iid")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _kz(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float64)
    if values.ndim == 1:
        return values
    if values.ndim != 2 or values.shape[1] not in (1, 3):
        raise ValueError(f"unsupported k_field shape: {values.shape}")
    return values[:, 0] if values.shape[1] == 1 else values[:, 2]


def _sample_features(sample_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    q = np.maximum(np.asarray(np.load(sample_dir / "q_field.npy")).reshape(-1), 0.0)
    kz = np.maximum(_kz(sample_dir / "k_field.npy").reshape(-1), EPS)
    coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
    if coords.shape != (q.size, 3) or kz.shape != q.shape:
        raise ValueError(f"{sample_dir.name}: coords/k/q shape mismatch")
    inv_kz = 1.0 / kz
    positive = q[q > EPS]
    q_reference = float(np.mean(positive)) if positive.size else 1.0
    inv_reference = max(float(np.mean(inv_kz)), EPS)
    q_relative = q / max(q_reference, EPS)
    inv_relative = inv_kz / inv_reference
    continuous = (
        np.log1p(q_relative),
        np.log(np.maximum(inv_relative, EPS)),
        np.log1p(q_relative * inv_relative),
    )
    q75 = float(np.quantile(q, 0.75))
    q_high = q >= q75
    inv_high = inv_kz >= np.quantile(inv_kz, 0.75)
    overlap = (q_high & inv_high).astype(np.float64)
    source_present = (q > EPS).astype(np.float64)
    z = coords[:, 2]
    z_norm = (z - float(np.min(z))) / max(float(np.ptp(z)), EPS)
    old_overlap_count = int(np.count_nonzero(overlap))
    old_overlap_outside_source = int(np.count_nonzero((overlap > 0.0) & (q <= EPS)))
    zero_q_count = int(np.count_nonzero(q <= EPS))
    old_q_high_false_positive = int(np.count_nonzero(q_high & (q <= EPS)))
    return (
        {
            "bugged_v1": np.stack((*continuous, overlap, z_norm), axis=-1),
            "sparse_safe_v2": np.stack(
                (*continuous, source_present, z_norm), axis=-1
            ),
        },
        {
            "node_count": int(q.size),
            "q75_is_zero": float(q75 <= EPS),
            "zero_q_node_count": zero_q_count,
            "old_q_high_false_positive_count": old_q_high_false_positive,
            "old_overlap_count": old_overlap_count,
            "old_overlap_outside_source_count": old_overlap_outside_source,
        },
    )


def _summary(rows: list[np.ndarray], counters: list[dict[str, float]]) -> dict:
    packed = {
        version: np.concatenate([row[version] for row in rows], axis=0)
        for version in FEATURE_NAMES
    }
    node_count = sum(int(row["node_count"]) for row in counters)
    zero_q_count = sum(int(row["zero_q_node_count"]) for row in counters)
    old_overlap_count = sum(int(row["old_overlap_count"]) for row in counters)
    return {
        "sample_count": len(rows),
        "node_count": node_count,
        "old_bug_coverage": {
            "q75_zero_sample_fraction": float(
                np.mean([row["q75_is_zero"] for row in counters])
            ),
            "q_high_false_positive_fraction_among_zero_q_nodes": (
                sum(int(row["old_q_high_false_positive_count"]) for row in counters)
                / max(zero_q_count, 1)
            ),
            "old_overlap_node_fraction": old_overlap_count / max(node_count, 1),
            "old_overlap_outside_source_fraction": (
                sum(
                    int(row["old_overlap_outside_source_count"])
                    for row in counters
                )
                / max(old_overlap_count, 1)
            ),
        },
        "features": {
            version: {
                "names": list(FEATURE_NAMES[version]),
                "variance": {
                    name: float(np.var(packed[version][:, index]))
                    for index, name in enumerate(FEATURE_NAMES[version])
                },
                "pearson_correlation": np.corrcoef(
                    packed[version], rowvar=False
                ).tolist(),
            }
            for version in FEATURE_NAMES
        },
    }


def main() -> int:
    args = _args()
    roles = tuple(role for role in args.roles.split(",") if role)
    if not roles or any(role not in ROLE_ALLOWLIST for role in roles):
        raise ValueError(f"roles must be a subset of {ROLE_ALLOWLIST}")
    assignments = json.loads(args.split_map.read_text(encoding="utf-8"))[
        "sample_splits"
    ]
    split = {
        role: sorted(
            sample_id
            for sample_id, assigned_role in assignments.items()
            if assigned_role == role
        )
        for role in roles
    }
    role_rows: dict[str, dict] = {}
    combined_features: list[dict[str, np.ndarray]] = []
    combined_counters: list[dict[str, float]] = []
    for role in roles:
        features: list[dict[str, np.ndarray]] = []
        counters: list[dict[str, float]] = []
        for sample_id in split[role]:
            sample_features, sample_counters = _sample_features(args.subset / sample_id)
            features.append(sample_features)
            counters.append(sample_counters)
        role_rows[role] = _summary(features, counters)
        combined_features.extend(features)
        combined_counters.extend(counters)
    payload = {
        "schema_version": "heat3d_v5_qk_sparse_feature_audit_v1",
        "status": "passed",
        "roles_accessed": list(roles),
        "forbidden_roles_accessed": [],
        "sealed_iid_accessed": False,
        "target_or_label_files_read": [],
        "feature_source_files": ["coords.npy", "k_field.npy", "q_field.npy"],
        "feature_schema": {
            version: list(names) for version, names in FEATURE_NAMES.items()
        },
        "roles": role_rows,
        "combined": _summary(combined_features, combined_counters),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
