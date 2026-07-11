#!/usr/bin/env python3
"""Export frozen V4 processed-region mean-pooled latents for V5 Gate 4A.

This is read-only inference over a frozen checkpoint.  It does not train or
update parameters.  Every recovered raw-temperature prediction is checked
against the supplied frozen prediction archive before a latent export is
accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "rigno").is_dir() and (Path.cwd() / "rigno").is_dir():
    # Remote audits deliberately execute a copied script from /tmp while the
    # immutable source checkout remains untouched.
    REPO_ROOT = Path.cwd()
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rigno.heat3d_v1_normalization import recover_temperature_from_normalized_delta  # noqa: E402
from run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    GraphNeuralOperator,
    Heat3DGraphBuilder,
    Heat3DV1NativeSupervisedDataset,
    _device_params,
    _load_params_checkpoint,
    _make_groups_with_progress,
    _resolve_decoder_bypass_model_config,
    _sample_root,
    _validate_model_config,
)
from run_heat3d_v3_final_probe_checkpoint_smoke import (  # noqa: E402
    install_checkpoint_feature_hooks,
    load_training_examples,
    stats_from_checkpoint_payload,
)


LATENT_NAME = "rnodes_processed"
# The archived split exports were produced with heterogeneous batch compositions
# (the train/hard-train temporary exports use 16, while historical exports use
# their original split batch size).  RIGNO's batched sparse reductions differ at
# the low-tens-of-mK level across those compositions, so provenance checks use a small
# raw-temperature tolerance while the Gate 4A field path always retains the
# exact archived prediction arrays.
PREDICTION_TOL_K = 2.0e-2


class LatentExportError(RuntimeError):
    """Raised when a frozen checkpoint or latent provenance check fails."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LatentExportError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LatentExportError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_ids(path: Path) -> list[str]:
    payload = _read_json(path)
    raw = payload.get("sample_splits")
    if not isinstance(raw, Mapping) or not raw:
        raise LatentExportError("split map must contain nonempty sample_splits")
    return sorted(str(sample_id) for sample_id in raw)


def _load_frozen_predictions(paths: Sequence[Path], expected_ids: set[str]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if not paths:
        raise LatentExportError("at least one --frozen-predictions archive is required")
    merged: dict[str, np.ndarray] = {}
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise LatentExportError(f"prediction archive does not exist: {path}")
        try:
            archive = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise LatentExportError(f"cannot read prediction archive {path}: {exc}") from exc
        if not archive.files:
            raise LatentExportError(f"prediction archive is empty: {path}")
        for sample_id in archive.files:
            if sample_id not in expected_ids:
                raise LatentExportError(f"unexpected sample in frozen prediction archive: {sample_id}")
            if sample_id in merged:
                raise LatentExportError(f"duplicate frozen prediction for {sample_id}")
            prediction = np.asarray(archive[sample_id], dtype=np.float64).reshape(-1)
            if prediction.size == 0 or not np.all(np.isfinite(prediction)):
                raise LatentExportError(f"invalid frozen prediction for {sample_id}")
            merged[sample_id] = prediction
        artifacts.append({"path": path.as_posix(), "sha256": _sha256(path), "sample_count": len(archive.files)})
    if set(merged) != expected_ids:
        missing = sorted(expected_ids - set(merged))
        raise LatentExportError(f"frozen predictions do not cover split-map samples; missing={missing[:8]}")
    return merged, artifacts


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
        raise LatentExportError(f"dataset is missing split-map samples: {missing[:8]}")
    return [dataset[index_by_id[sample_id]] for sample_id in sample_ids]


def _find_rnodes_processed(value: Any) -> Any:
    if isinstance(value, Mapping):
        if LATENT_NAME in value:
            candidate = value[LATENT_NAME]
            if isinstance(candidate, (tuple, list)):
                if len(candidate) != 1:
                    raise LatentExportError(f"{LATENT_NAME} collection must contain exactly one value")
                candidate = candidate[0]
            return candidate
        matches = []
        for child in value.values():
            try:
                matches.append(_find_rnodes_processed(child))
            except KeyError:
                continue
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise LatentExportError(f"{LATENT_NAME} intermediate is ambiguous")
    raise KeyError(LATENT_NAME)


def _pooled_rnodes(value: Any, batch_size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0, :, :]
    if array.ndim != 3 or array.shape[0] != batch_size or array.shape[1] < 1 or array.shape[2] < 1:
        raise LatentExportError(f"unexpected {LATENT_NAME} shape: {array.shape}")
    if not np.all(np.isfinite(array)):
        raise LatentExportError(f"{LATENT_NAME} contains non-finite values")
    return array.mean(axis=1)


def _extract(
    *,
    checkpoint: Path,
    run_config_path: Path,
    subset: Path,
    split_map: Path,
    frozen_predictions: Mapping[str, np.ndarray],
    batch_size: int,
    progress_detail: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not checkpoint.is_file() or not run_config_path.is_file():
        raise LatentExportError("checkpoint and run-config must exist")
    run_config = _read_json(run_config_path)
    checkpoint_payload = _load_params_checkpoint(checkpoint)
    checkpoint_stats = dict(checkpoint_payload.get("train_only_normalization") or {})
    if not checkpoint_stats:
        raise LatentExportError("checkpoint lacks train_only_normalization")
    sample_root = _sample_root(subset)
    sample_ids = _split_ids(split_map)
    install_checkpoint_feature_hooks(checkpoint_stats)
    train_examples = load_training_examples(run_config, checkpoint_stats)
    examples = _load_examples(
        sample_root=sample_root,
        sample_ids=sample_ids,
        checkpoint_stats=checkpoint_stats,
        boundary_mask_fallback=bool(run_config.get("boundary_mask_fallback", True)),
    )
    stats = stats_from_checkpoint_payload(checkpoint_stats, train_examples)
    model_config = dict(checkpoint_payload.get("model_config") or run_config.get("model_config") or {})
    if not model_config:
        raise LatentExportError("checkpoint lacks model_config")
    model_config = _resolve_decoder_bypass_model_config(model_config, stats)
    _validate_model_config(model_config)
    builder = Heat3DGraphBuilder(**dict(run_config.get("graph_config") or {}))
    groups = _make_groups_with_progress(
        examples,
        stats,
        builder,
        "gate4a_pooled_latent",
        progress_detail != "off",
        progress_detail,
        int(run_config.get("graph_seed", 0)),
        batch_size=batch_size,
        drop_last=False,
    )
    model = GraphNeuralOperator(**model_config)
    params = _device_params(checkpoint_payload["params"])
    latents: dict[str, np.ndarray] = {}
    max_prediction_error = 0.0
    for group in groups:
        output, mutable = model.apply(
            {"params": params},
            inputs=group["inputs"],
            graphs=group["graphs"],
            mutable=["intermediates"],
        )
        recovered = np.asarray(recover_temperature_from_normalized_delta(output, group["t_ref"], stats), dtype=np.float64)
        pooled = _pooled_rnodes(_find_rnodes_processed(mutable), recovered.shape[0])
        for index, sample_id in enumerate(group["sample_ids"]):
            prediction = recovered[index, 0, :, :].reshape(-1)
            reference = frozen_predictions[sample_id]
            if prediction.shape != reference.shape:
                raise LatentExportError(f"{sample_id}: frozen prediction shape mismatch")
            error = float(np.max(np.abs(prediction - reference)))
            max_prediction_error = max(max_prediction_error, error)
            if error > PREDICTION_TOL_K:
                raise LatentExportError(
                    f"{sample_id}: recovered prediction drift {error:.3e} K exceeds {PREDICTION_TOL_K:.3e} K"
                )
            latents[sample_id] = pooled[index].astype(np.float64)
    if set(latents) != set(sample_ids):
        raise LatentExportError("latent sample IDs do not cover split map")
    dimensions = {value.shape for value in latents.values()}
    if len(dimensions) != 1:
        raise LatentExportError(f"pooled latent dimensions differ: {dimensions}")
    return latents, {
        "sample_count": len(latents),
        "latent_dimension": int(next(iter(dimensions))[0]),
        "max_prediction_abs_error_K": max_prediction_error,
        "prediction_tolerance_K": PREDICTION_TOL_K,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "run_config": run_config_path.as_posix(),
        "run_config_sha256": _sha256(run_config_path),
        "subset": sample_root.as_posix(),
        "split_map": split_map.as_posix(),
        "model_config": model_config,
    }


def _write_outputs(latents: Mapping[str, np.ndarray], manifest: Mapping[str, Any], output: Path, manifest_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{sample_id: latents[sample_id] for sample_id in sorted(latents)})
    payload = dict(manifest)
    payload["latent_archive"] = output.as_posix()
    payload["latent_archive_sha256"] = _sha256(output)
    payload["pooling"] = "mean over frozen rnodes_processed regional nodes"
    payload["read_only"] = True
    payload["training_runs"] = 0
    payload["model_parameter_changes"] = 0
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--split-map", type=Path, required=True)
    parser.add_argument("--frozen-predictions", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--progress-detail", choices=("off", "basic"), default="basic")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.batch_size < 1:
            raise LatentExportError("--batch-size must be >= 1")
        sample_ids = _split_ids(args.split_map)
        frozen, artifacts = _load_frozen_predictions(args.frozen_predictions, set(sample_ids))
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "mode": "dry_run",
                        "read_only": True,
                        "sample_count": len(sample_ids),
                        "frozen_prediction_archives": artifacts,
                        "planned_writes": [],
                        "training_runs": 0,
                        "model_parameter_changes": 0,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.output is None or args.manifest is None:
            raise LatentExportError("normal mode requires --output and --manifest")
        if args.output.resolve() == args.manifest.resolve():
            raise LatentExportError("--output and --manifest must differ")
        if not args.overwrite and (args.output.exists() or args.manifest.exists()):
            raise LatentExportError("refusing to overwrite latent output or manifest")
        latents, manifest = _extract(
            checkpoint=args.checkpoint,
            run_config_path=args.run_config,
            subset=args.subset,
            split_map=args.split_map,
            frozen_predictions=frozen,
            batch_size=args.batch_size,
            progress_detail=args.progress_detail,
        )
        manifest = {**manifest, "frozen_prediction_archives": artifacts}
        _write_outputs(latents, manifest, args.output, args.manifest)
        print(
            "gate4a_pooled_latents "
            f"samples={manifest['sample_count']} dim={manifest['latent_dimension']} "
            f"max_prediction_error_K={manifest['max_prediction_abs_error_K']:.3e}",
            flush=True,
        )
    except LatentExportError as exc:
        print(f"Gate 4A latent export error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
