#!/usr/bin/env python3
"""Generate a frozen V6-RandomBlock dataset without training or inference."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import yaml

import heat3d_v6_randomblock_core as core


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6_randomblock"
DOCS_DIR = ROOT / "docs"
ARRAY_FILES = (
    "coords.npy",
    "temperature.npy",
    "deltaT.npy",
    "k_field.npy",
    "q_field.npy",
    "layer_id.npy",
    "bc_features.npy",
    "control_volume.npy",
)


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise core.RandomBlockError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _config_contract(config: Mapping[str, Any]) -> None:
    cases = list(config["cases"])
    groups = list(config["layout_groups"])
    variants = list(config["physics_variants"])
    if len(cases) != int(config["sample_count"]):
        raise core.RandomBlockError("sample count mismatch")
    if len(groups) != int(config["group_count"]):
        raise core.RandomBlockError("group count mismatch")
    if len(variants) != 8 or int(config["variants_per_group"]) != 8:
        raise core.RandomBlockError("eight-variant contract failed")
    if len(cases) != 8 * len(groups):
        raise core.RandomBlockError("group/case cardinality mismatch")
    ids = [str(row["sample_id"]) for row in cases]
    if len(set(ids)) != len(ids):
        raise core.RandomBlockError("duplicate sample ID")
    group_map = {str(row["group_id"]): row for row in groups}
    if len(group_map) != len(groups):
        raise core.RandomBlockError("duplicate group ID")
    by_group: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        group_id = str(case["group_id"])
        if group_id not in group_map:
            raise core.RandomBlockError("case references missing group")
        if str(case["split_role"]) != str(group_map[group_id]["split_role"]):
            raise core.RandomBlockError("group split leakage")
        by_group[group_id].append(case)
    for group_id, rows in by_group.items():
        if len(rows) != 8:
            raise core.RandomBlockError(f"{group_id}: not eight variants")
        if Counter(int(row["intended_temperature_bin"]) for row in rows) != {
            0: 2,
            1: 2,
            2: 2,
            3: 2,
        }:
            raise core.RandomBlockError(f"{group_id}: intended bin imbalance")
    for key, value in config["guardrails"].items():
        if key in {
            "training",
            "model_inference",
            "post_solve_temperature_filtering",
            "post_solve_sample_replacement",
            "group_split_leakage",
            "canonical_p1h_modified",
        } and value is not False:
            raise core.RandomBlockError(f"forbidden guardrail: {key}={value}")
    embedded = dict(config)
    provenance = dict(embedded["provenance"])
    expected = str(provenance.pop("protocol_sha256"))
    embedded["provenance"] = provenance
    actual = core.canonical_json_sha256(embedded)
    if actual != expected:
        raise core.RandomBlockError(
            f"protocol hash mismatch: {actual} != {expected}"
        )


def _sample_file_hashes(sample_dir: Path) -> dict[str, str]:
    result = {
        name: core.file_sha256(sample_dir / name) for name in ARRAY_FILES
    }
    result["sample_meta.json"] = core.file_sha256(
        sample_dir / "sample_meta.json"
    )
    return result


def _support_audit_rows(
    group: Mapping[str, Any],
    support: Mapping[str, Any],
    mesh: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for index, count in enumerate(support["block_coverage"]):
        family = "q" if index < len(group["q_blocks"]) else "k"
        local = index if family == "q" else index - len(group["q_blocks"])
        rows.append(
            {
                "group_id": group["group_id"],
                "family": family,
                "block_index": local,
                "support_node_count": int(count),
                "coordinate_sha256": support["coordinate_sha256"],
                "support_index_sha256": support["index_sha256"],
                "solver_node_count": int(mesh["node_count"]),
            }
        )
    return rows


def _write_sample(
    sample_dir: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
) -> dict[str, str]:
    sample_dir.mkdir(parents=True, exist_ok=False)
    for name, value in arrays.items():
        np.save(sample_dir / name, np.asarray(value), allow_pickle=False)
    _json(sample_dir / "sample_meta.json", meta)
    return _sample_file_hashes(sample_dir)


def _aggregate_audit(
    config: Mapping[str, Any],
    sample_rows: Sequence[Mapping[str, Any]],
    block_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    peaks = [float(row["peak_deltaT_K"]) for row in sample_rows]
    realized = Counter(
        "outside" if row["realized_temperature_bin"] == "" else str(
            row["realized_temperature_bin"]
        )
        for row in sample_rows
    )
    intended = Counter(str(row["intended_temperature_bin"]) for row in sample_rows)
    q_rows = [row for row in block_rows if row["family"] == "q"]
    k_rows = [row for row in block_rows if row["family"] == "k"]
    energy = [
        abs(float(row["energy_balance_relative_error"])) for row in sample_rows
    ]
    residual = [float(row["linear_residual"]) for row in sample_rows]
    support_counts = [int(row["support_node_count"]) for row in support_rows]
    in_window = sum(
        core.TEMPERATURE_BIN_EDGES_K[0]
        <= value
        <= core.TEMPERATURE_BIN_EDGES_K[-1]
        for value in peaks
    )
    physical_pass = (
        max(energy) <= 1.0e-6
        and max(residual) <= 1.0e-7
        and min(support_counts) > 0
        and all(math.isfinite(value) for value in peaks)
    )
    stage = str(config["stage"])
    temperature_pass = in_window == len(peaks)
    if stage in {"pilot128", "formal1024"}:
        expected = len(sample_rows) // 4
        temperature_pass = temperature_pass and all(
            int(realized.get(str(index), 0)) == expected
            for index in range(4)
        )
    return {
        "schema_version": "heat3d_v6_randomblock_audit_v1",
        "dataset_id": config["dataset_id"],
        "stage": stage,
        "status": (
            "passed"
            if physical_pass and temperature_pass
            else "failed_temperature_gate"
            if physical_pass
            else "failed_physical_gate"
        ),
        "sample_count": len(sample_rows),
        "group_count": int(config["group_count"]),
        "split_role_counts": dict(
            sorted(Counter(str(row["split_role"]) for row in sample_rows).items())
        ),
        "intended_temperature_bin_counts": dict(sorted(intended.items())),
        "realized_temperature_bin_counts": dict(sorted(realized.items())),
        "peak_deltaT_K": {
            "minimum": min(peaks),
            "median": float(np.median(peaks)),
            "maximum": max(peaks),
            "inside_30_150_count": in_window,
        },
        "physics_QC": {
            "maximum_energy_balance_relative_error": max(energy),
            "maximum_linear_residual": max(residual),
            "minimum_support_nodes_per_block": min(support_counts),
            "minimum_k_W_mK": min(float(row["k_x_W_mK"]) for row in k_rows),
            "maximum_k_W_mK": max(float(row["k_x_W_mK"]) for row in k_rows),
            "minimum_q_W_m3": min(float(row["q_W_m3"]) for row in q_rows),
            "maximum_q_W_m3": max(float(row["q_W_m3"]) for row in q_rows),
            "minimum_surface_power_density_W_cm2": min(
                float(row["surface_power_density_W_cm2"]) for row in q_rows
            ),
            "maximum_surface_power_density_W_cm2": max(
                float(row["surface_power_density_W_cm2"]) for row in q_rows
            ),
            "maximum_cg_iterations": max(
                int(row["cg_iterations"]) for row in sample_rows
            ),
        },
        "guardrails": {
            "training_runs": 0,
            "model_inference_runs": 0,
            "temperature_filtered_samples": 0,
            "sample_replacements": 0,
            "group_split_leakage": False,
            "canonical_p1h_modified": False,
        },
        "elapsed_seconds": float(elapsed_seconds),
    }


def _markdown(audit: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    peak = audit["peak_deltaT_K"]
    qc = audit["physics_QC"]
    lines = [
        f"# {audit['dataset_id']} audit",
        "",
        f"- stage/status: `{audit['stage']}` / `{audit['status']}`",
        f"- samples/groups: {audit['sample_count']} / {audit['group_count']}",
        f"- peak ΔT: {peak['minimum']:.6f}–{peak['maximum']:.6f} K "
        f"(median {peak['median']:.6f} K)",
        f"- realized bins: `{audit['realized_temperature_bin_counts']}`",
        f"- intended bins: `{audit['intended_temperature_bin_counts']}`",
        f"- max energy error: {qc['maximum_energy_balance_relative_error']:.3e}",
        f"- max linear residual: {qc['maximum_linear_residual']:.3e}",
        f"- k range: {qc['minimum_k_W_mK']:.6g}–"
        f"{qc['maximum_k_W_mK']:.6g} W/(m·K)",
        f"- q range: {qc['minimum_q_W_m3']:.6g}–"
        f"{qc['maximum_q_W_m3']:.6g} W/m³",
        f"- source flux range: "
        f"{qc['minimum_surface_power_density_W_cm2']:.6g}–"
        f"{qc['maximum_surface_power_density_W_cm2']:.6g} W/cm²",
        f"- manifest SHA256: `{manifest['manifest_payload_sha256']}`",
        f"- full-field archive SHA256: "
        f"`{manifest['full_field_archive']['sha256']}`",
        "",
        "所有样本均保留；没有按温度过滤、替换或重采。该阶段没有训练或模型推理。",
    ]
    return "\n".join(lines) + "\n"


def generate(
    config_path: Path,
    dataset: Path,
    artifact_dir: Path,
    docs_dir: Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    config = core.load_config(config_path)
    _config_contract(config)
    mesh = core.build_mesh(config["physics"])
    groups = {
        str(group["group_id"]): group for group in config["layout_groups"]
    }
    layout_audits = {
        group_id: core.validate_layout(group, mesh)
        for group_id, group in groups.items()
    }
    supports = {
        group_id: core.select_group_support(
            group,
            mesh,
            layout_audits[group_id],
            int(config["seeds"]["support"]),
        )
        for group_id, group in groups.items()
    }
    # Validate every amplitude assignment before any solver or write.
    for case in config["cases"]:
        group_id = str(case["group_id"])
        core.build_case_fields(
            case, groups[group_id], mesh, layout_audits[group_id]
        )
    if dry_run:
        return {
            "status": "dry_run_passed",
            "dataset_id": config["dataset_id"],
            "sample_count": int(config["sample_count"]),
            "group_count": int(config["group_count"]),
            "solver_node_count": int(mesh["node_count"]),
            "write_executed": False,
            "solve_executed": False,
            "training_executed": False,
            "model_inference_executed": False,
        }
    if dataset.exists():
        raise core.RandomBlockError(f"refusing to overwrite dataset: {dataset}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{dataset.name}.", dir=dataset.parent)
    )
    start = time.perf_counter()
    sample_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    manifest_samples: list[dict[str, Any]] = []
    for group_id, group in groups.items():
        support_rows.extend(
            _support_audit_rows(group, supports[group_id], mesh)
        )
    archive_path = temporary / "full_fields.h5"
    try:
        config_snapshot = temporary / "generation_config.yaml"
        config_snapshot.write_text(
            yaml.safe_dump(config, sort_keys=False, width=100),
            encoding="utf-8",
        )
        sample_count = int(config["sample_count"])
        node_count = int(mesh["node_count"])
        with h5py.File(archive_path, "w") as archive:
            archive.attrs["schema_version"] = (
                "heat3d_v6_randomblock_full_fields_v1"
            )
            archive.attrs["dataset_id"] = str(config["dataset_id"])
            archive.create_dataset(
                "coords",
                data=np.asarray(mesh["coords"], dtype=np.float64),
                compression="lzf",
            )
            archive.create_dataset(
                "control_volume",
                data=np.asarray(mesh["weights"], dtype=np.float64),
                compression="lzf",
            )
            archive.create_dataset(
                "layer_id",
                data=np.asarray(mesh["layer_ids"], dtype=np.int16),
                compression="lzf",
            )
            string_type = h5py.string_dtype(encoding="utf-8")
            sample_ids = archive.create_dataset(
                "sample_id", (sample_count,), dtype=string_type
            )
            full_temperature = archive.create_dataset(
                "temperature_K",
                (sample_count, node_count),
                dtype="f4",
                chunks=(1, node_count),
                compression="lzf",
            )
            full_q = archive.create_dataset(
                "q_W_m3",
                (sample_count, node_count),
                dtype="f4",
                chunks=(1, node_count),
                compression="lzf",
            )
            full_k = archive.create_dataset(
                "k_xyz_W_mK",
                (sample_count, node_count, 3),
                dtype="f4",
                chunks=(1, node_count, 3),
                compression="lzf",
            )
            for row_index, case in enumerate(config["cases"]):
                sample_start = time.perf_counter()
                group_id = str(case["group_id"])
                group = groups[group_id]
                layout = layout_audits[group_id]
                support = supports[group_id]
                k_diag, q, case_blocks = core.build_case_fields(
                    case, group, mesh, layout
                )
                temperature, solver = core.solve_case(
                    mesh,
                    k_diag,
                    q,
                    top_h=float(case["top_h_W_m2K"]),
                    bottom_h=float(case["bottom_h_W_m2K"]),
                )
                metrics = core.case_metrics(mesh, temperature, q, solver)
                indices = np.asarray(support["indices"], dtype=np.int64)
                sample_id = str(case["sample_id"])
                sample_dir = temporary / sample_id
                flags = core.boundary_flags(support["coords"], mesh)
                arrays = {
                    "coords.npy": np.asarray(support["coords"], dtype=np.float64),
                    "temperature.npy": temperature[indices, None],
                    "deltaT.npy": (
                        temperature[indices, None] - core.AMBIENT_K
                    ),
                    "k_field.npy": k_diag[indices],
                    "q_field.npy": q[indices, None],
                    "layer_id.npy": np.asarray(mesh["layer_ids"])[indices, None],
                    "bc_features.npy": flags,
                    "control_volume.npy": np.asarray(
                        support["control_volume"], dtype=np.float64
                    )[:, None],
                }
                block_meta = []
                for block in case_blocks:
                    enriched = dict(block)
                    family = str(block["family"])
                    block_index = int(block["block_index"])
                    geometry = (
                        group["q_blocks"][block_index]
                        if family == "q"
                        else group["k_blocks"][block_index]
                    )
                    enriched["bbox_fraction_xy"] = geometry[
                        "bbox_fraction_xy"
                    ]
                    enriched["block_id"] = geometry["block_id"]
                    block_meta.append(enriched)
                    block_rows.append(
                        {
                            "sample_id": sample_id,
                            "group_id": group_id,
                            "split_role": case["split_role"],
                            **enriched,
                        }
                    )
                projected_peak = float(
                    np.max(arrays["deltaT.npy"])
                )
                meta = {
                    "schema_version": "heat3d_v6_randomblock_sample_v1",
                    "dataset_id": config["dataset_id"],
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "split_role": case["split_role"],
                    "variant_id": case["variant_id"],
                    "intended_temperature_bin": int(
                        case["intended_temperature_bin"]
                    ),
                    "realized_temperature_bin": metrics[
                        "realized_temperature_bin"
                    ],
                    "boundary_conditions": {
                        "top": {
                            "type": "robin",
                            "h_W_m2K": float(case["top_h_W_m2K"]),
                            "T_inf_K": core.AMBIENT_K,
                        },
                        "bottom": {
                            "type": "robin",
                            "h_W_m2K": float(case["bottom_h_W_m2K"]),
                            "T_inf_K": core.AMBIENT_K,
                        },
                        "sides": {"type": "adiabatic"},
                    },
                    "package_total_power_W": float(
                        case["package_total_power_W"]
                    ),
                    "layers_bottom_to_top": config["physics"][
                        "layers_bottom_to_top"
                    ],
                    "blocks": block_meta,
                    "metrics": metrics,
                    "solver_peak_minus_projected_peak_K": float(
                        metrics["peak_deltaT_K"] - projected_peak
                    ),
                    "support": {
                        "selection_uses_temperature_or_labels": False,
                        "coordinate_sha256": support["coordinate_sha256"],
                        "support_index_sha256": support["index_sha256"],
                        "stratum_counts": dict(
                            sorted(Counter(support["strata"]).items())
                        ),
                        "block_coverage": support["block_coverage"],
                    },
                    "provenance": {
                        "protocol_sha256": config["provenance"][
                            "protocol_sha256"
                        ],
                        "layout_hash": group["layout_hash"],
                        "power": case["power_provenance"],
                        "bc": case["bc_provenance"],
                        "k": case["k_provenance"],
                        "q": case["q_provenance"],
                    },
                    "guardrails": {
                        "sample_temperature_filtered": False,
                        "sample_replaced": False,
                        "training": False,
                        "model_inference": False,
                    },
                }
                hashes = _write_sample(
                    sample_dir, arrays=arrays, meta=meta
                )
                sample_ids[row_index] = sample_id
                full_temperature[row_index] = temperature.astype(np.float32)
                full_q[row_index] = q.astype(np.float32)
                full_k[row_index] = k_diag.astype(np.float32)
                row = {
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "split_role": case["split_role"],
                    "variant_id": case["variant_id"],
                    "intended_temperature_bin": int(
                        case["intended_temperature_bin"]
                    ),
                    "realized_temperature_bin": (
                        ""
                        if metrics["realized_temperature_bin"] is None
                        else int(metrics["realized_temperature_bin"])
                    ),
                    "package_total_power_W": metrics[
                        "package_total_power_W"
                    ],
                    "top_h_W_m2K": float(case["top_h_W_m2K"]),
                    "bottom_h_W_m2K": float(case["bottom_h_W_m2K"]),
                    "peak_deltaT_K": metrics["peak_deltaT_K"],
                    "mean_deltaT_K": metrics["mean_deltaT_K"],
                    "cv_rms_deltaT_K": metrics["cv_rms_deltaT_K"],
                    "projected_peak_deltaT_K": projected_peak,
                    "top_heat_fraction": metrics["top_heat_fraction"],
                    "bottom_heat_fraction": metrics["bottom_heat_fraction"],
                    "energy_balance_relative_error": metrics[
                        "energy_balance_relative_error"
                    ],
                    "linear_residual": metrics["linear_residual"],
                    "cg_iterations": metrics["cg_iterations"],
                    "support_coordinate_sha256": support[
                        "coordinate_sha256"
                    ],
                    "layout_hash": group["layout_hash"],
                    "solve_and_write_seconds": time.perf_counter()
                    - sample_start,
                }
                sample_rows.append(row)
                manifest_samples.append(
                    {
                        "sample_id": sample_id,
                        "sample_dir": sample_id,
                        "group_id": group_id,
                        "split_role": case["split_role"],
                        "variant_id": case["variant_id"],
                        "full_field_archive_row": row_index,
                        "point_coordinates_sha256": support[
                            "coordinate_sha256"
                        ],
                        "support_index_sha256": support["index_sha256"],
                        "layout_hash": group["layout_hash"],
                        "file_sha256": hashes,
                    }
                )
        elapsed = time.perf_counter() - start
        archive_sha = core.file_sha256(archive_path)
        manifest = {
            "schema_version": "heat3d_v6_randomblock_manifest_v1",
            "dataset_id": config["dataset_id"],
            "stage": config["stage"],
            "sample_count": len(manifest_samples),
            "group_count": int(config["group_count"]),
            "protocol_sha256": config["provenance"]["protocol_sha256"],
            "config_sha256": core.file_sha256(config_path),
            "solver_mesh": {
                "shape": list(mesh["shape"]),
                "node_count": int(mesh["node_count"]),
                "coords_sha256": core.canonical_json_sha256(
                    np.asarray(mesh["coords"]).tolist()
                ),
            },
            "full_field_archive": {
                "path": "full_fields.h5",
                "sha256": archive_sha,
                "dtype": "float32",
                "sample_count": len(manifest_samples),
                "solver_node_count": int(mesh["node_count"]),
            },
            "samples": manifest_samples,
            "guardrails": {
                "training_runs": 0,
                "model_inference_runs": 0,
                "temperature_filtered_samples": 0,
                "sample_replacements": 0,
                "test_used_for_model_selection": False,
            },
        }
        manifest["manifest_payload_sha256"] = core.canonical_json_sha256(
            manifest
        )
        audit = _aggregate_audit(
            config, sample_rows, block_rows, support_rows, elapsed
        )
        dataset_manifest = temporary / "manifest.json"
        dataset_audit = temporary / "audit.json"
        _json(dataset_manifest, manifest)
        _json(dataset_audit, audit)
        _csv(temporary / "samples.csv", sample_rows)
        _csv(temporary / "blocks.csv", block_rows)
        _csv(temporary / "support_coverage.csv", support_rows)
        temporary.rename(dataset)

        stem = str(config["dataset_id"]).removeprefix("heat3d_")
        manifest_path = artifact_dir / f"{stem}_manifest.json"
        audit_path = artifact_dir / f"{stem}_audit.json"
        samples_path = artifact_dir / f"{stem}_samples.csv"
        blocks_path = artifact_dir / f"{stem}_blocks.csv"
        support_path = artifact_dir / f"{stem}_support_coverage.csv"
        report_path = docs_dir / f"{stem}_audit.md"
        _json(manifest_path, manifest)
        _json(audit_path, audit)
        _csv(samples_path, sample_rows)
        _csv(blocks_path, block_rows)
        _csv(support_path, support_rows)
        report_path.write_text(
            _markdown(audit, manifest), encoding="utf-8"
        )
        return {
            "status": audit["status"],
            "dataset": _relative(dataset),
            "manifest": _relative(manifest_path),
            "audit": _relative(audit_path),
            "report": _relative(report_path),
            "sample_count": len(sample_rows),
            "group_count": int(config["group_count"]),
            "elapsed_seconds": elapsed,
            "manifest_payload_sha256": manifest[
                "manifest_payload_sha256"
            ],
            "full_field_archive_sha256": archive_sha,
            "training_executed": False,
            "model_inference_executed": False,
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else ROOT / args.config
    ).resolve()
    config = core.load_config(config_path)
    dataset = args.dataset or ROOT / "data" / str(config["dataset_id"])
    dataset = (dataset if dataset.is_absolute() else ROOT / dataset).resolve()
    artifact_dir = (
        args.artifact_dir
        if args.artifact_dir.is_absolute()
        else ROOT / args.artifact_dir
    ).resolve()
    docs_dir = (
        args.docs_dir if args.docs_dir.is_absolute() else ROOT / args.docs_dir
    ).resolve()
    result = generate(
        config_path,
        dataset,
        artifact_dir,
        docs_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"passed", "dry_run_passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
