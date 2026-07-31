#!/usr/bin/env python3
"""Target-independent split utilities for V6-P1i continuous datasets."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import ks_2samp, qmc

import generate_heat3d_v6_p1i_continuous_dataset as base_generator
import heat3d_v6_p1i_continuous_core as core


ROOT = Path(__file__).resolve().parent.parent
ROLES = ("train", "valid_iid", "test_iid")
ROLE_PAIRS = (
    ("train", "valid_iid"),
    ("train", "test_iid"),
    ("valid_iid", "test_iid"),
)

CONTINUOUS_FEATURES = (
    "continuous_severity",
    "package_total_power_W",
    "top_h_W_m2K",
    "bottom_h_W_m2K",
    "total_source_area_fraction",
    "mean_source_area_fraction",
    "source_area_cv",
    "mean_source_aspect_ratio",
    "source_centroid_spread",
    "source_upper_fraction",
    "q_proxy_mean_W_m3",
    "q_proxy_max_W_m3",
    "mean_local_k_W_mK",
    "local_k_log_std",
    "cross_family_overlap_fraction",
    "background_kz_pcb_fr4_equivalent_W_mK",
    "background_kz_bt_substrate_with_vias_W_mK",
    "background_kz_silicon_interposer_tsv_0p1_W_mK",
    "background_kz_bump_underfill_under_interposer_W_mK",
    "background_kz_silicon_die_lower_W_mK",
    "background_kz_tim_between_dies_W_mK",
    "background_kz_silicon_die_upper_W_mK",
    "background_kz_tim_to_spreader_W_mK",
    "background_kz_spreader_W_mK",
)

DISCRETE_FEATURES = (
    "source_count",
    "k_region_count",
    "upper_source_count",
    "cross_family_overlap_pair_count",
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bbox_features(blocks: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    area = []
    aspect = []
    centroids = []
    for block in blocks:
        x0, x1, y0, y1 = map(float, block["bbox_fraction_xy"])
        width, height = x1 - x0, y1 - y0
        area.append(width * height)
        aspect.append(max(width / height, height / width))
        centroids.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    area_array = np.asarray(area, dtype=np.float64)
    centroid_array = np.asarray(centroids, dtype=np.float64)
    centroid_spread = float(
        np.sqrt(np.mean(np.sum((centroid_array - centroid_array.mean(0)) ** 2, axis=1)))
    )
    return {
        "total_source_area_fraction": float(area_array.sum()),
        "mean_source_area_fraction": float(area_array.mean()),
        "source_area_cv": float(area_array.std() / area_array.mean()),
        "mean_source_aspect_ratio": float(np.mean(aspect)),
        "source_centroid_spread": centroid_spread,
    }


def record_from_case(
    case: Mapping[str, Any],
    group: Mapping[str, Any],
    physics: Mapping[str, Any],
) -> dict[str, Any]:
    q_blocks = list(group["q_blocks"])
    k_blocks = list(group["k_blocks"])
    result: dict[str, Any] = {
        "sample_id": str(case["sample_id"]),
        "sobol_index": int(case["sobol_index"]),
        "sobol_dimensions_consumed": int(case["sobol_dimensions_consumed"]),
        "continuous_severity": float(case["continuous_severity"]),
        "package_total_power_W": float(case["package_total_power_W"]),
        "top_h_W_m2K": float(case["top_h_W_m2K"]),
        "bottom_h_W_m2K": float(case["bottom_h_W_m2K"]),
        "source_count": len(q_blocks),
        "k_region_count": len(k_blocks),
        "upper_source_count": sum(
            block["layer"] == "silicon_die_upper" for block in q_blocks
        ),
        "cross_family_overlap_pair_count": int(
            group["cross_family_overlap_pair_count"]
        ),
    }
    for name in (
        "power_base_W",
        "power_top_h_factor",
        "power_independent_latent",
        "power_independent_multiplier",
    ):
        if name in case:
            result[name] = float(case[name])
    result.update(_bbox_features(q_blocks))
    result["source_upper_fraction"] = (
        float(result["upper_source_count"]) / float(result["source_count"])
    )
    result["cross_family_overlap_fraction"] = (
        float(result["cross_family_overlap_pair_count"])
        / max(float(result["source_count"] * result["k_region_count"]), 1.0)
    )
    footprint = np.prod(np.asarray(physics["footprint_m"], dtype=np.float64))
    thickness = {
        str(layer["id"]): float(layer["thickness_m"])
        for layer in physics["layers_bottom_to_top"]
    }
    volumes = []
    for block in q_blocks:
        x0, x1, y0, y1 = map(float, block["bbox_fraction_xy"])
        volumes.append(
            (x1 - x0) * (y1 - y0) * footprint * thickness[str(block["layer"])]
        )
    volume_array = np.asarray(volumes, dtype=np.float64)
    fractions = np.asarray(case["q_block_power_fractions"], dtype=np.float64)
    q_proxy = float(case["package_total_power_W"]) * fractions / volume_array
    result["q_proxy_mean_W_m3"] = float(np.mean(q_proxy))
    result["q_proxy_max_W_m3"] = float(np.max(q_proxy))
    local_k = np.asarray(case["k_block_values_W_mK"], dtype=np.float64)
    result["mean_local_k_W_mK"] = float(np.mean(local_k))
    result["local_k_log_std"] = float(np.std(np.log(local_k)))
    for layer in physics["layers_bottom_to_top"]:
        result[
            f"background_kz_{layer['id']}_W_mK"
        ] = float(layer["background_k_xyz_W_mK"][2])
    return result


def design_records(
    config_path: Path,
    *,
    case_builder: Callable[..., Any] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    config = core.load_config(config_path)
    count = int(config["sample_count"])
    sampler = qmc.Sobol(
        d=int(config["sampling"]["dimensions"]),
        scramble=bool(config["sampling"]["scramble"]),
        seed=int(config["sampling"]["seed"]),
    )
    values = sampler.random_base2(int(math.log2(count)))
    builder = case_builder or base_generator._case_from_sobol
    records = []
    for index, row in enumerate(values):
        case, group, physics, _ = builder(index, row, config, "unassigned")
        records.append(record_from_case(case, group, physics))
    return records, values


def _role_slots(counts: Mapping[str, int]) -> list[str]:
    return [
        role for role in ROLES for _ in range(int(counts[role]))
    ]


def octet_hash_assignment(
    sample_ids: Sequence[str],
    counts: Mapping[str, int],
    *,
    salt: str,
) -> dict[str, str]:
    if len(sample_ids) % 8:
        raise core.ContinuousPhysicsError("octet assignment requires multiple of 8")
    per_block = {"train": 6, "valid_iid": 1, "test_iid": 1}
    result: dict[str, str] = {}
    slots = _role_slots(per_block)
    for offset in range(0, len(sample_ids), 8):
        ordered = sorted(
            sample_ids[offset : offset + 8],
            key=lambda value: _hash(f"{salt}:{value}"),
        )
        result.update(dict(zip(ordered, slots)))
    if Counter(result.values()) != Counter(
        {role: int(counts[role]) for role in ROLES}
    ):
        raise core.ContinuousPhysicsError("octet assignment count mismatch")
    return result


def global_hash_assignment(
    sample_ids: Sequence[str],
    counts: Mapping[str, int],
    *,
    salt: str,
) -> dict[str, str]:
    ordered = sorted(sample_ids, key=lambda value: _hash(f"{salt}:{value}"))
    return dict(zip(ordered, _role_slots(counts)))


def sobol_dimension_assignment(
    sample_ids: Sequence[str],
    counts: Mapping[str, int],
    *,
    sobol_values: np.ndarray,
    dimension: int,
) -> dict[str, str]:
    values = np.asarray(sobol_values, dtype=np.float64)
    if values.shape[0] != len(sample_ids):
        raise core.ContinuousPhysicsError("reserved Sobol row mismatch")
    if not 0 <= int(dimension) < values.shape[1]:
        raise core.ContinuousPhysicsError("reserved Sobol dimension out of range")
    split_values = values[:, int(dimension)]
    ordered = [
        sample_ids[index]
        for index in np.lexsort((np.arange(len(split_values)), split_values))
    ]
    return dict(zip(ordered, _role_slots(counts)))


def _rank_bins(values: np.ndarray, bins: int) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.int64)
    ranks[order] = np.arange(values.size)
    return np.minimum((ranks * bins) // values.size, bins - 1)


def _stratification_tokens(
    records: Sequence[Mapping[str, Any]],
    *,
    continuous_bins: int = 8,
) -> tuple[list[list[str]], list[str]]:
    tokens: list[list[str]] = [[] for _ in records]
    continuous_bin_values: dict[str, np.ndarray] = {}
    for name in CONTINUOUS_FEATURES:
        values = np.asarray([float(row[name]) for row in records])
        encoded = _rank_bins(values, continuous_bins)
        continuous_bin_values[name] = encoded
        for index, value in enumerate(encoded):
            tokens[index].append(f"c:{name}:{int(value)}")
    for name in DISCRETE_FEATURES:
        for index, row in enumerate(records):
            tokens[index].append(f"d:{name}:{int(row[name])}")
    joint_pairs = (
        ("package_total_power_W", "top_h_W_m2K"),
        ("package_total_power_W", "bottom_h_W_m2K"),
        ("package_total_power_W", "total_source_area_fraction"),
        ("top_h_W_m2K", "bottom_h_W_m2K"),
        ("q_proxy_mean_W_m3", "mean_local_k_W_mK"),
    )
    for left, right in joint_pairs:
        left_bins = continuous_bin_values[left] // 2
        right_bins = continuous_bin_values[right] // 2
        for index, (a, b) in enumerate(zip(left_bins, right_bins)):
            tokens[index].append(f"j:{left}:{right}:{int(a)}:{int(b)}")
    all_tokens = sorted({token for row in tokens for token in row})
    return tokens, all_tokens


def balanced_input_assignment(
    records: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
    *,
    salt: str,
) -> dict[str, str]:
    """Greedy multilabel stratification using pre-solve inputs only."""

    tokens, token_names = _stratification_tokens(records)
    token_index = {value: index for index, value in enumerate(token_names)}
    token_totals = np.zeros(len(token_names), dtype=np.float64)
    token_rows: list[np.ndarray] = []
    for row in tokens:
        encoded = np.asarray([token_index[value] for value in row], dtype=np.int64)
        token_rows.append(encoded)
        token_totals[encoded] += 1.0
    n = len(records)
    targets = {
        role: token_totals * (float(counts[role]) / float(n))
        for role in ROLES
    }
    current = {
        role: np.zeros(len(token_names), dtype=np.float64) for role in ROLES
    }
    remaining = {role: int(counts[role]) for role in ROLES}
    rarity = np.asarray(
        [sum(1.0 / token_totals[index] for index in row) for row in token_rows]
    )
    ordered = sorted(
        range(n),
        key=lambda index: (
            -rarity[index],
            _hash(f"{salt}:{records[index]['sample_id']}"),
        ),
    )
    assignment: dict[str, str] = {}
    for sample_index in ordered:
        encoded = token_rows[sample_index]
        candidates = [role for role in ROLES if remaining[role] > 0]
        scores = {}
        for role in candidates:
            deficit = targets[role][encoded] - current[role][encoded]
            balance = remaining[role] / float(counts[role])
            scores[role] = float(
                np.sum(deficit / np.sqrt(token_totals[encoded])) + 0.25 * balance
            )
        best = max(
            candidates,
            key=lambda role: (
                scores[role],
                _hash(f"{salt}:{records[sample_index]['sample_id']}:{role}"),
            ),
        )
        assignment[str(records[sample_index]["sample_id"])] = best
        current[best][encoded] += 1.0
        remaining[best] -= 1
    if any(remaining.values()):
        raise core.ContinuousPhysicsError(f"balanced assignment count drift: {remaining}")
    candidates = [assignment]
    sample_ids = [str(row["sample_id"]) for row in records]
    for index in range(512):
        candidates.append(
            global_hash_assignment(
                sample_ids,
                counts,
                salt=f"{salt}:multistart:{index:04d}",
            )
        )
    return min(
        candidates,
        key=lambda candidate: (
            _split_objective(records, candidate),
            assignment_sha256(candidate),
        ),
    )


def _tv(left: np.ndarray, right: np.ndarray) -> float:
    categories = np.unique(np.concatenate((left, right)))
    p = np.asarray([np.mean(left == value) for value in categories])
    q = np.asarray([np.mean(right == value) for value in categories])
    return float(0.5 * np.sum(np.abs(p - q)))


def split_metrics(
    records: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, str],
) -> dict[str, Any]:
    masks = {
        role: np.asarray(
            [assignment[str(row["sample_id"])] == role for row in records]
        )
        for role in ROLES
    }
    continuous_rows = []
    for name in CONTINUOUS_FEATURES:
        values = np.asarray([float(row[name]) for row in records])
        for left, right in ROLE_PAIRS:
            result = ks_2samp(values[masks[left]], values[masks[right]])
            continuous_rows.append(
                {
                    "feature": name,
                    "left": left,
                    "right": right,
                    "ks": float(result.statistic),
                }
            )
    discrete_rows = []
    for name in DISCRETE_FEATURES:
        values = np.asarray([int(row[name]) for row in records])
        for left, right in ROLE_PAIRS:
            discrete_rows.append(
                {
                    "feature": name,
                    "left": left,
                    "right": right,
                    "tv": _tv(values[masks[left]], values[masks[right]]),
                }
            )
    rank_matrix = np.column_stack(
        [
            _rank_bins(
                np.asarray([float(row[name]) for row in records]), 32
            )
            / 31.0
            for name in CONTINUOUS_FEATURES
        ]
    )
    joint_rows = []
    for left, right in ROLE_PAIRS:
        left_values, right_values = rank_matrix[masks[left]], rank_matrix[masks[right]]
        mean_term = float(
            np.linalg.norm(left_values.mean(0) - right_values.mean(0))
            / math.sqrt(rank_matrix.shape[1])
        )
        covariance_term = float(
            np.linalg.norm(
                np.cov(left_values, rowvar=False)
                - np.cov(right_values, rowvar=False),
                ord="fro",
            )
            / rank_matrix.shape[1]
        )
        joint_rows.append(
            {
                "left": left,
                "right": right,
                "mean_discrepancy": mean_term,
                "covariance_discrepancy": covariance_term,
                "joint_discrepancy": mean_term + covariance_term,
            }
        )
    return {
        "counts": dict(Counter(assignment.values())),
        "continuous": continuous_rows,
        "discrete": discrete_rows,
        "joint": joint_rows,
        "maximum_continuous_ks": max(row["ks"] for row in continuous_rows),
        "mean_continuous_ks": float(
            np.mean([row["ks"] for row in continuous_rows])
        ),
        "maximum_discrete_tv": max(row["tv"] for row in discrete_rows),
        "mean_discrete_tv": float(
            np.mean([row["tv"] for row in discrete_rows])
        ),
        "maximum_joint_discrepancy": max(
            row["joint_discrepancy"] for row in joint_rows
        ),
    }


def _split_objective(
    records: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, str],
) -> float:
    metrics = split_metrics(records, assignment)
    return float(
        metrics["maximum_continuous_ks"]
        + metrics["maximum_discrete_tv"]
        + metrics["maximum_joint_discrepancy"]
    )


def assignment_sha256(assignment: Mapping[str, str]) -> str:
    return core.canonical_json_sha256(dict(sorted(assignment.items())))
