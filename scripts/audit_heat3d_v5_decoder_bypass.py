#!/usr/bin/env python3
"""Read-only frozen V4P5_02 decoder-bypass audit for V5 clean-first work.

The audit replays the frozen V4 best checkpoint with mutable intermediates,
extracts the actual post-decoder residual, and compares that prediction with
the same checkpoint output after subtracting only the residual contribution.
It never trains, mutates a checkpoint, alters a split, or writes data/output
directories.  The only writes are the explicitly requested CSV/JSON/Markdown
audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.heat3d_v1_normalization import normalized_delta_to_raw, recover_raw_condition  # noqa: E402
from rigno.heat3d_v5_bypass_audit import (  # noqa: E402
    bypass_structure_recommendation,
    classify_feature_node_variation,
    compare_bypass_metric_rows,
)
from rigno.heat3d_v5_metrics import compute_sample_metrics, control_volume_weights  # noqa: E402
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    Heat3DGraphBuilder,
    Heat3DV1NativeSupervisedDataset,
    _device_params,
    _load_params_checkpoint,
    _make_groups_with_progress,
    _resolve_decoder_bypass_model_config,
    _resolve_training_splits,
    _sample_root,
    _validate_model_config,
)
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    load_training_examples,
    stats_from_checkpoint_payload,
)


AUDIT_ID = "V5-frozen-V4P5_02-decoder-bypass-audit"
SCHEMA_VERSION = "heat3d_v5_decoder_bypass_audit_v1"
DEFAULT_ROLES = ("train", "valid_iid", "test_iid")
EXPECTED_BASELINE_ID = "V4P5_02_clean_baseline_raw_B28_e600"
EXPECTED_EPOCH = 405
EPS = 1.0e-12


class AuditError(RuntimeError):
    """Raised for a malformed frozen V4 bypass audit."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=None)
    parser.add_argument("--split-map", type=Path, default=None)
    parser.add_argument(
        "--frozen-valid-predictions",
        type=Path,
        default=None,
        help=(
            "Optional frozen V4 best raw-temperature NPZ for valid_iid replay verification. "
            "It is read only and must cover exactly the frozen valid IDs."
        ),
    )
    parser.add_argument("--role", choices=DEFAULT_ROLES, action="append")
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--expected-epoch", type=int, default=EXPECTED_EPOCH)
    parser.add_argument("--output-table", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_output_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    forbidden = {"data", "output", "checkpoints", "logs"}
    if any(part in forbidden for part in resolved.parts):
        raise AuditError(f"--{label} must not write under data/output/checkpoints/logs: {path}")
    return resolved


def _resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.output_table is None or args.output_json is None or args.output_md is None:
        raise AuditError("audit requires --output-table, --output-json, and --output-md")
    paths = tuple(
        _ensure_output_path(path, label)
        for path, label in (
            (args.output_table, "output-table"),
            (args.output_json, "output-json"),
            (args.output_md, "output-md"),
        )
    )
    if len(set(paths)) != 3:
        raise AuditError("audit output paths must be distinct")
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise AuditError(f"refusing to overwrite audit artifact(s): {existing}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def _load_examples(
    *,
    sample_root: Path,
    sample_ids: Sequence[str],
    checkpoint_stats: Mapping[str, Any],
    boundary_mask_fallback: bool,
) -> list[Any]:
    feature_names = tuple(checkpoint_stats.get("feature_names") or ())
    k_encoding_mode = "diag3" if {"k_x", "k_y", "k_z"}.issubset(feature_names) else "native"
    dataset = Heat3DV1NativeSupervisedDataset(
        sample_root,
        k_encoding_mode=k_encoding_mode,
        boundary_mask_fallback=boundary_mask_fallback,
    )
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
    if missing:
        raise AuditError(f"dataset is missing frozen split sample IDs: {missing[:10]}")
    return [dataset[index_by_id[sample_id]] for sample_id in sample_ids]


def _load_frozen_prediction_archive(
    path: Path | None,
    expected_ids: Sequence[str],
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    if not path.is_file():
        raise AuditError(f"frozen valid prediction archive does not exist: {path}")
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise AuditError(f"cannot load frozen valid prediction archive {path}: {exc}") from exc
    expected = set(expected_ids)
    found = set(archive.files)
    if found != expected:
        raise AuditError(
            "frozen valid prediction archive IDs differ from valid_iid split: "
            f"missing={sorted(expected - found)[:5]} extra={sorted(found - expected)[:5]}"
        )
    values: dict[str, np.ndarray] = {}
    for sample_id in expected_ids:
        field = np.asarray(archive[sample_id], dtype=np.float64).reshape(-1)
        if field.size == 0 or not np.all(np.isfinite(field)):
            raise AuditError(f"{path}: frozen prediction {sample_id} is invalid")
        values[sample_id] = field
    return values, {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "sample_count": len(values),
        "comparison_tolerance_K": 0.02,
    }


def _validate_checkpoint(
    checkpoint_path: Path,
    checkpoint_payload: Mapping[str, Any],
    expected_epoch: int,
) -> None:
    if not checkpoint_path.is_file():
        raise AuditError(f"checkpoint does not exist: {checkpoint_path}")
    if int(checkpoint_payload.get("epoch", -1)) != expected_epoch:
        raise AuditError(
            f"frozen checkpoint epoch mismatch: expected {expected_epoch}, "
            f"found {checkpoint_payload.get('epoch')!r}"
        )
    if EXPECTED_BASELINE_ID not in checkpoint_path.as_posix():
        raise AuditError(f"checkpoint must be under frozen baseline {EXPECTED_BASELINE_ID}")
    config = dict(checkpoint_payload.get("model_config") or {})
    if config.get("decoder_bypass_mode") != "post_decoder_residual":
        raise AuditError("frozen checkpoint must use post_decoder_residual")
    if config.get("decoder_bypass_features") != "full_condition":
        raise AuditError("frozen checkpoint must use full_condition bypass")


def _metrics_sample(
    *,
    sample_id: str,
    role: str,
    prediction_normalized: np.ndarray,
    target_normalized: np.ndarray,
    target_delta: np.ndarray,
    q: np.ndarray,
    volumes: np.ndarray,
) -> dict[str, Any]:
    prediction_delta = np.asarray(normalized_delta_to_raw(prediction_normalized, _CURRENT_STATS), dtype=np.float64).reshape(-1)
    return compute_sample_metrics(
        {
            "sample_id": sample_id,
            "split": role,
            "prediction_deltaT_K": prediction_delta,
            "target_deltaT_K": target_delta,
            "control_volumes_m3": volumes,
            "q_W_m3": q,
            "prediction_normalized": prediction_normalized,
            "target_normalized": target_normalized,
        }
    )


# This reference is set once after frozen checkpoint stats are reconstructed.
# Keeping it module-local makes the per-sample routine easy to use from a JAX
# output loop while preventing any target information from entering model inputs.
_CURRENT_STATS: dict[str, Any] = {}


def _run_audit(args: argparse.Namespace, outputs: tuple[Path, Path, Path]) -> dict[str, Any]:
    global _CURRENT_STATS
    run_config = _load_json(args.run_config)
    checkpoint_payload = _load_params_checkpoint(args.checkpoint)
    _validate_checkpoint(args.checkpoint, checkpoint_payload, args.expected_epoch)
    checkpoint_stats = dict(checkpoint_payload.get("train_only_normalization") or {})
    if not checkpoint_stats:
        raise AuditError("frozen checkpoint lacks train_only_normalization")
    install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples_for_stats = load_training_examples(run_config, checkpoint_stats)
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples_for_stats)
    _CURRENT_STATS = stats

    sample_root = _sample_root(args.subset or Path(str(run_config.get("subset") or "")))
    if not sample_root.is_dir():
        raise AuditError(f"frozen subset does not exist: {sample_root}")
    split_map = args.split_map or Path(str(run_config.get("split_map_path") or ""))
    if not split_map.is_file():
        raise AuditError(f"frozen split map does not exist: {split_map}")
    split_ids, split_source, _primary, _stress = _resolve_training_splits(sample_root, split_map)
    roles = tuple(args.role or DEFAULT_ROLES)
    for role in roles:
        if not split_ids.get(role):
            raise AuditError(f"frozen split map has no samples for role={role}")
    archived_valid, archived_valid_provenance = _load_frozen_prediction_archive(
        args.frozen_valid_predictions,
        split_ids.get("valid_iid", []),
    )

    model_config = _resolve_decoder_bypass_model_config(
        dict(checkpoint_payload.get("model_config") or {}), stats
    )
    _validate_model_config(model_config)
    feature_names = tuple(model_config.get("decoder_bypass_feature_names") or ())
    if not feature_names:
        raise AuditError("frozen bypass feature list is empty")
    graph_config = dict(run_config.get("graph_config") or {})
    graph_seed = int(run_config.get("graph_seed", 0))
    builder = Heat3DGraphBuilder(**graph_config)
    model = GraphNeuralOperator(**model_config)
    params = _device_params(checkpoint_payload["params"])
    residual_scale = float(model_config["decoder_bypass_residual_scale"])
    per_role_full: dict[str, list[dict[str, Any]]] = {role: [] for role in roles}
    per_role_without: dict[str, list[dict[str, Any]]] = {role: [] for role in roles}
    per_sample_rows: list[dict[str, Any]] = []
    raw_feature_samples: list[np.ndarray] = []
    replay_max_abs_normalized_error = 0.0
    replay_archive_max_abs_error_K: float | None = None

    for role in roles:
        examples = _load_examples(
            sample_root=sample_root,
            sample_ids=split_ids[role],
            checkpoint_stats=checkpoint_stats,
            boundary_mask_fallback=bool(run_config.get("boundary_mask_fallback", True)),
        )
        groups = _make_groups_with_progress(
            examples,
            stats,
            builder,
            role,
            False,
            "basic",
            graph_seed,
            batch_size=args.prediction_batch_size,
            drop_last=False,
        )
        coords_by_id = {example.sample_id: np.asarray(example.condition.coords, dtype=np.float64) for example in examples}
        for group in groups:
            full_normalized, mutable = model.apply(
                {"params": params},
                inputs=group["inputs"],
                graphs=group["graphs"],
                mutable=["intermediates"],
            )
            residual_tree = mutable.get("intermediates", {}).get("decoder_bypass_residual")
            if not residual_tree:
                raise AuditError(f"{group['name']}: frozen model did not expose decoder_bypass_residual")
            residual = np.asarray(residual_tree[0], dtype=np.float64)
            full = np.asarray(full_normalized, dtype=np.float64)
            target = np.asarray(group["target_normalized"], dtype=np.float64)
            target_delta = np.asarray(group["target_delta_raw"], dtype=np.float64)
            if residual.shape != full.shape or full.shape != target.shape:
                raise AuditError(f"{group['name']}: output/residual/target shapes do not align")
            without = full - residual_scale * residual
            replay_max_abs_normalized_error = max(
                replay_max_abs_normalized_error,
                float(np.max(np.abs((without + residual_scale * residual) - full))),
            )
            raw_c = np.asarray(recover_raw_condition(group["inputs"].c, stats), dtype=np.float64)
            raw_c = raw_c[:, 0, :, :]
            if raw_c.shape[-1] != len(feature_names):
                raise AuditError(f"{group['name']}: raw condition width differs from frozen bypass feature list")
            q_index = feature_names.index("q") if "q" in feature_names else None
            for index, sample_id in enumerate(group["sample_ids"]):
                volumes = control_volume_weights(coords_by_id[sample_id])
                sample_full = _metrics_sample(
                    sample_id=sample_id,
                    role=role,
                    prediction_normalized=full[index],
                    target_normalized=target[index],
                    target_delta=target_delta[index].reshape(-1),
                    q=raw_c[index, :, q_index] if q_index is not None else np.zeros(volumes.shape),
                    volumes=volumes,
                )
                sample_without = _metrics_sample(
                    sample_id=sample_id,
                    role=role,
                    prediction_normalized=without[index],
                    target_normalized=target[index],
                    target_delta=target_delta[index].reshape(-1),
                    q=raw_c[index, :, q_index] if q_index is not None else np.zeros(volumes.shape),
                    volumes=volumes,
                )
                per_role_full[role].append(sample_full)
                per_role_without[role].append(sample_without)
                raw_feature_samples.append(raw_c[index])
                raw_residual_delta = residual_scale * residual[index].reshape(-1) * float(
                    np.asarray(stats["target_delta_std"]).reshape(-1)[0]
                )
                if archived_valid is not None and role == "valid_iid":
                    raw_temperature = np.asarray(group["t_ref"][index], dtype=np.float64).reshape(-1) + np.asarray(
                        normalized_delta_to_raw(full[index], stats), dtype=np.float64
                    ).reshape(-1)
                    archived = archived_valid[sample_id]
                    if raw_temperature.shape != archived.shape:
                        raise AuditError(f"{sample_id}: frozen valid replay shape mismatch")
                    current_error = float(np.max(np.abs(raw_temperature - archived)))
                    replay_archive_max_abs_error_K = max(replay_archive_max_abs_error_K or 0.0, current_error)
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "role": role,
                    "bypass_residual_scale": residual_scale,
                    "bypass_residual_normalized_mean_abs": float(np.mean(np.abs(residual[index]))),
                    "bypass_residual_normalized_rmse": float(math.sqrt(np.mean(np.square(residual[index])))),
                    "bypass_residual_raw_delta_cv_rms_K": float(
                        math.sqrt(np.sum(np.square(raw_residual_delta) * volumes) / np.sum(volumes))
                    ),
                    "bypass_residual_raw_delta_mean_abs_K": float(np.mean(np.abs(raw_residual_delta))),
                }
                row.update({f"full_{key}": value for key, value in sample_full.items()})
                row.update({f"without_bypass_{key}": value for key, value in sample_without.items()})
                per_sample_rows.append(row)

    feature_variation = classify_feature_node_variation(feature_names, raw_feature_samples)
    decision = bypass_structure_recommendation(feature_variation)
    role_comparisons = {
        role: compare_bypass_metric_rows(per_role_full[role], per_role_without[role])
        for role in roles
    }
    all_full = [row for role in roles for row in per_role_full[role]]
    all_without = [row for role in roles for row in per_role_without[role]]
    role_comparisons["clean_all"] = compare_bypass_metric_rows(all_full, all_without)

    table_path, json_path, md_path = outputs
    _write_table(per_sample_rows, table_path)
    payload = {
        "audit_id": AUDIT_ID,
        "schema_version": SCHEMA_VERSION,
        "mode": "read_only_frozen_checkpoint_replay",
        "baseline": {
            "config_id": EXPECTED_BASELINE_ID,
            "checkpoint": args.checkpoint.as_posix(),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "checkpoint_epoch": int(checkpoint_payload["epoch"]),
            "run_config": args.run_config.as_posix(),
            "run_config_sha256": _sha256(args.run_config),
            "full_condition_bypass": {
                "mode": model_config["decoder_bypass_mode"],
                "feature_source": model_config["decoder_bypass_feature_source"],
                "residual_scale": residual_scale,
                "feature_names": list(feature_names),
            },
        },
        "dataset": {
            "subset": sample_root.as_posix(),
            "split_map": split_map.as_posix(),
            "split_source": split_source,
            "roles": list(roles),
            "role_counts": {role: len(per_role_full[role]) for role in roles},
            "clean_only": True,
        },
        "frozen_replay": {
            "output_identity_reconstruction_max_abs_normalized_error": replay_max_abs_normalized_error,
            "residual_subtraction": "without_bypass_normalized = full_normalized - decoder_bypass_residual_scale * captured_decoder_bypass_residual",
            "target_or_label_derived_model_inputs": False,
            "training_runs": 0,
            "checkpoint_writes": 0,
            "dataset_writes": 0,
            "valid_archive": archived_valid_provenance,
            "valid_archive_max_abs_temperature_error_K": replay_archive_max_abs_error_K,
            "valid_archive_replay_pass": (
                None
                if archived_valid_provenance is None
                else bool(replay_archive_max_abs_error_K is not None and replay_archive_max_abs_error_K <= 0.02)
            ),
        },
        "bypass_input_variation": feature_variation,
        "architecture_recommendation": decision,
        "metric_comparisons": role_comparisons,
        "per_sample_table": {
            "path": table_path.as_posix(),
            "sha256": _sha256(table_path),
            "row_count": len(per_sample_rows),
            "column_count": len(per_sample_rows[0]) if per_sample_rows else 0,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _write_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise AuditError("bypass audit produced no sample rows")
    fields = list(rows[0])
    for row in rows:
        if list(row) != fields:
            raise AuditError("per-sample bypass CSV schema drift")
    with path.open("w", encoding="utf-8", newline="") as handle:
        # Keep committed audit artifacts LF-only so Git's whitespace checker
        # treats the generated CSV as a portable text file on every host.
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}g}" if np.isfinite(numeric) else "n/a"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["architecture_recommendation"]
    lines = [
        "# V5 Frozen V4 Decoder-Bypass Audit",
        "",
        "## Scope",
        "",
        f"- Frozen baseline: `{payload['baseline']['config_id']}` epoch `{payload['baseline']['checkpoint_epoch']}`.",
        "- Replayed only clean `train`, `valid_iid`, and `test_iid`; no hard role, training, checkpoint write, data write, or model modification occurred.",
        "- The disabled-bypass counterfactual is exact residual subtraction in normalized DeltaT space from the captured V4 `decoder_bypass_residual` intermediate.",
        "",
        "## Input Variation Audit",
        "",
        "| feature | classification | node-varying samples | invariant samples | max within-sample range | V5 treatment |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in payload["bypass_input_variation"]:
        treatment = "retain local bypass" if row["retain_as_local_bypass_input"] else "move to Global FiLM only"
        lines.append(
            f"| {row['feature_name']} | {row['classification']} | {row['node_varying_sample_count']} | "
            f"{row['node_invariant_sample_count']} | {_fmt(row['max_within_sample_range'])} | {treatment} |"
        )
    lines.extend(
        [
            "",
            "## Architecture Decision",
            "",
            f"- Decision: `{decision['decision']}`.",
            f"- Rationale: {decision['rationale']}",
            f"- Local-capable bypass inputs: `{', '.join(decision['local_bypass_feature_names']) or 'none'}`.",
            f"- Sample-global broadcast inputs to remove from local bypass: `{', '.join(decision['global_broadcast_feature_names']) or 'none'}`.",
            "",
            "## Frozen Full Bypass Versus Residual-Disabled Counterfactual",
            "",
            "Positive error reduction means retaining the frozen full bypass lowers that error; spatial-correlation gain is full minus disabled.",
            "",
            "| role | full sample-first CV-rel % | disabled sample-first CV-rel % | bypass reduction pp | full raw CV-RMSE K | disabled raw CV-RMSE K | bypass hotspot reduction K | bypass shape-CV reduction | bypass scale-log reduction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for role, comparison in payload["metric_comparisons"].items():
        full = comparison["with_full_bypass"]
        disabled = comparison["without_bypass"]
        reduction = comparison["bypass_error_reduction_positive_is_better"]
        lines.append(
            "| "
            + " | ".join(
                (
                    role,
                    _fmt(full.get("sample_first_cv_relative_rmse_pct")),
                    _fmt(disabled.get("sample_first_cv_relative_rmse_pct")),
                    _fmt(reduction.get("sample_first_cv_relative_rmse_pct")),
                    _fmt(full.get("raw_cv_weighted_rmse_K")),
                    _fmt(disabled.get("raw_cv_weighted_rmse_K")),
                    _fmt(reduction.get("hotspot_cv_weighted_rmse_K")),
                    _fmt(reduction.get("shape_cv_rmse")),
                    _fmt(reduction.get("scale_log_rmse")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The JSON contains the full required V5 metric suite for both variants, including point-global relative RMSE, background bias/RMSE/over-ratio, top-five and strong-q RMSE, amplitude ratio, and spatial correlation.",
            "",
        ]
    )
    return "\n".join(lines)


def _dry_run(args: argparse.Namespace) -> dict[str, Any]:
    run_config = _load_json(args.run_config)
    checkpoint_payload = _load_params_checkpoint(args.checkpoint)
    _validate_checkpoint(args.checkpoint, checkpoint_payload, args.expected_epoch)
    sample_root = _sample_root(args.subset or Path(str(run_config.get("subset") or "")))
    split_map = args.split_map or Path(str(run_config.get("split_map_path") or ""))
    split_ids, split_source, _primary, _stress = _resolve_training_splits(sample_root, split_map)
    roles = tuple(args.role or DEFAULT_ROLES)
    return {
        "audit_id": AUDIT_ID,
        "mode": "dry_run",
        "read_only": True,
        "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "roles": {role: len(split_ids.get(role, [])) for role in roles},
        "frozen_valid_predictions": args.frozen_valid_predictions.as_posix() if args.frozen_valid_predictions else None,
        "split_source": split_source,
        "planned_writes": [],
        "training_runs": 0,
        "dataset_writes": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.prediction_batch_size < 1:
            raise AuditError("--prediction-batch-size must be >= 1")
        if not args.run_config.is_file():
            raise AuditError(f"run config does not exist: {args.run_config}")
        if args.dry_run:
            print(json.dumps(_dry_run(args), indent=2, sort_keys=True))
            return 0
        outputs = _resolve_outputs(args)
        payload = _run_audit(args, outputs)
    except (AuditError, ValueError) as exc:
        print(f"decoder bypass audit error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "audit_id": payload["audit_id"],
                "rows": payload["per_sample_table"]["row_count"],
                "decision": payload["architecture_recommendation"]["decision"],
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
