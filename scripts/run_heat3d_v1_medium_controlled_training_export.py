"""Controlled Heat3D v1 medium training export smoke.

This runner reuses the existing v1 train/valid smoke path and writes recovered
temperature predictions to an ignored output directory for downstream
diagnostic comparison. It is not a formal training experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    MODEL_CONFIG,
    _global_norm,
    _make_groups,
    _metrics,
    _sample_root,
    _selected_steps,
    _subset_split_ids,
    _train_only_stats,
    _weighted_loss,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


DEFAULT_SUBSET = (
    REPO_DIR
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium_v2"
)
DEFAULT_OUTPUT_DIR = REPO_DIR / "output" / "heat3d_v1_medium_runs" / "export_smoke_seed0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled training export smoke for Heat3D v1 medium labels. "
            "Writes ignored predictions for diagnostic comparison only."
        )
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--report-every", type=int, default=1)
    return parser.parse_args()


def _ensure_ignored_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    output_root = (REPO_DIR / "output").resolve()
    if resolved != output_root and output_root not in resolved.parents:
        raise ValueError(f"--output-dir must be under ignored output/: {path}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _require_train_valid_splits(split_ids: dict[str, list[str]]) -> None:
    train_ids = split_ids.get("train", [])
    valid_ids = split_ids.get("valid", [])
    if not train_ids or not valid_ids:
        raise ValueError(
            "Expected non-empty train and valid splits for controlled training export, "
            f"found train={len(train_ids)} valid={len(valid_ids)}"
        )


def _should_report_epoch(epoch: int, epochs: int, report_every: int) -> bool:
    return epoch == 1 or epoch == epochs or epoch % report_every == 0


def _epoch_history_record(
    epoch: int,
    train_loss: float,
    valid_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "valid_loss": float(valid_metrics["normalized_loss"]),
        "train_raw_deltaT_mse": float(train_metrics["raw_delta_mse"]),
        "valid_raw_deltaT_mse": float(valid_metrics["raw_delta_mse"]),
        "train_recovered_T_mse": float(train_metrics["recovered_temperature_mse"]),
        "valid_recovered_T_mse": float(valid_metrics["recovered_temperature_mse"]),
    }


def _print_epoch_progress(record: dict[str, Any], epochs: int) -> None:
    print(
        f"epoch {record['epoch']:03d}/{epochs:03d} "
        f"train_loss={record['train_loss']:.8e} "
        f"valid_loss={record['valid_loss']:.8e} "
        f"train_raw_deltaT_mse={record['train_raw_deltaT_mse']:.8e} "
        f"valid_raw_deltaT_mse={record['valid_raw_deltaT_mse']:.8e} "
        f"train_recovered_T_mse={record['train_recovered_T_mse']:.8e} "
        f"valid_recovered_T_mse={record['valid_recovered_T_mse']:.8e}"
    )


def _fit_once(
    train_groups: list[dict],
    valid_groups: list[dict],
    stats: dict,
    epochs: int,
    lr: float,
    seed: int,
    report_every: int,
) -> dict:
    model = GraphNeuralOperator(**MODEL_CONFIG)
    params = model.init(
        jax.random.PRNGKey(seed),
        inputs=train_groups[0]["inputs"],
        graphs=train_groups[0]["graphs"],
    )["params"]

    def loss_fn(current_params):
        return _weighted_loss(model, current_params, train_groups)

    train_losses = [float(loss_fn(params))]
    valid_losses = [_metrics(model, params, valid_groups, stats)["normalized_loss"]]
    grad_norms = []
    grad_finite = True
    epoch_history = []
    for epoch in range(1, epochs + 1):
        _, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = _global_norm(grads)
        grad_norms.append(grad_norm)
        grad_finite = grad_finite and bool(np.isfinite(grad_norm))
        params = tree.tree_map(lambda param, grad: param - lr * grad, params, grads)
        train_loss = float(loss_fn(params))
        valid_metrics = _metrics(model, params, valid_groups, stats)
        train_losses.append(train_loss)
        valid_losses.append(valid_metrics["normalized_loss"])
        if _should_report_epoch(epoch, epochs, report_every):
            train_metrics = _metrics(model, params, train_groups, stats)
            record = _epoch_history_record(epoch, train_loss, valid_metrics, train_metrics)
            epoch_history.append(record)
            _print_epoch_progress(record, epochs)

    train_metrics = _metrics(model, params, train_groups, stats)
    valid_metrics = _metrics(model, params, valid_groups, stats)
    status_ok = (
        grad_finite
        and train_metrics["finite_ok"]
        and valid_metrics["finite_ok"]
        and train_metrics["shape_ok"]
        and valid_metrics["shape_ok"]
        and bool(np.all(np.isfinite(train_losses)))
        and bool(np.all(np.isfinite(valid_losses)))
    )
    return {
        "model": model,
        "params": params,
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "valid_losses": np.asarray(valid_losses, dtype=np.float64),
        "grad_norms": np.asarray(grad_norms, dtype=np.float64),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "epoch_history": epoch_history,
        "grad_finite": grad_finite,
        "status_ok": status_ok,
    }


def _predict_temperatures(model, params, groups: list[dict], stats: dict) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for group in groups:
        pred_normalized = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        pred_delta = pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]
        recovered = np.asarray(group["t_ref"] + pred_delta)
        if not np.all(np.isfinite(recovered)):
            raise ValueError(f"Non-finite recovered predictions in group {group['name']}")
        for batch_index, sample_id in enumerate(group["sample_ids"]):
            predictions[sample_id] = recovered[batch_index, 0, :, :].astype(np.float64)
    return predictions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _stats_payload(stats: dict) -> dict[str, Any]:
    return {
        "feature_names": list(stats["feature_names"]),
        "target_delta_mean": float(stats["target_delta_mean"].reshape(-1)[0]),
        "target_delta_std": float(stats["target_delta_std"].reshape(-1)[0]),
        "condition_mean": [float(value) for value in stats["condition_mean"].reshape(-1)],
        "condition_std": [float(value) for value in stats["condition_std"].reshape(-1)],
    }


def _metrics_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
        for key, value in metrics.items()
    }


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.report_every < 1:
        raise ValueError("--report-every must be >= 1")

    output_dir = _ensure_ignored_output_dir(args.output_dir)
    sample_root = _sample_root(args.subset)
    split_ids = _subset_split_ids(sample_root)
    _require_train_valid_splits(split_ids)

    all_ids = sorted(sample_id for ids in split_ids.values() for sample_id in ids)
    train_ids = split_ids["train"]
    valid_ids = split_ids["valid"]
    for sample_id in all_ids:
        if not (sample_root / sample_id / "temperature.npy").is_file():
            raise FileNotFoundError(f"Missing temperature.npy for {sample_id}")

    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode="diag3")
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in all_ids if sample_id not in index_by_id]
    if missing:
        raise FileNotFoundError(f"Dataset loader did not expose samples: {missing}")

    train_examples = [dataset[index_by_id[sample_id]] for sample_id in train_ids]
    valid_examples = [dataset[index_by_id[sample_id]] for sample_id in valid_ids]
    all_examples = [dataset[index_by_id[sample_id]] for sample_id in all_ids]

    builder = Heat3DGraphBuilder()
    stats = _train_only_stats(train_examples)
    train_groups = _make_groups(train_examples, stats, builder)
    valid_groups = _make_groups(valid_examples, stats, builder)
    all_groups = _make_groups(all_examples, stats, builder)

    split_counts = {split: len(ids) for split, ids in sorted(split_ids.items())}

    print("Heat3D v1 medium controlled training export smoke")
    print("  scope: research reference diagnostics only; not formal model performance")
    print(f"  subset: {sample_root}")
    print(f"  split counts: {split_counts}")
    print(f"  epochs: {args.epochs}")
    print(f"  lr: {args.lr}")
    print(f"  seed: {args.seed}")
    print(f"  output_dir: {output_dir}")
    print(f"  save_predictions: {bool(args.save_predictions)}")
    print(f"  report every: {args.report_every}")
    print(f"  feature mode: relative BC features, diag3 k encoding, zero_delta_u_bridge")
    print("  target mode: normalized DeltaT target; recovered temperature predictions exported")

    result = _fit_once(train_groups, valid_groups, stats, args.epochs, args.lr, args.seed, args.report_every)
    predictions = _predict_temperatures(result["model"], result["params"], all_groups, stats)

    run_config = {
        "diagnostic_scope": "controlled training export smoke; not formal model performance",
        "subset": str(sample_root),
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "route": "relative BC features + zero_delta_u_bridge + normalized DeltaT target",
        "output_dir": str(output_dir),
        "save_predictions": bool(args.save_predictions),
        "checkpoint_saved": False,
        "split_counts": split_counts,
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "ignored_candidate_ids": sorted(
            sample_id for split, ids in split_ids.items() if split not in {"train", "valid"} for sample_id in ids
        ),
    }
    _write_json(output_dir / "run_config.json", run_config)

    loss_summary = {
        "status_ok": bool(result["status_ok"]),
        "grad_finite": bool(result["grad_finite"]),
        "train_losses": [float(value) for value in result["train_losses"]],
        "valid_losses": [float(value) for value in result["valid_losses"]],
        "grad_norms": [float(value) for value in result["grad_norms"]],
        "train_metrics": _metrics_payload(result["train_metrics"]),
        "valid_metrics": _metrics_payload(result["valid_metrics"]),
        "train_loss_selected": _selected_steps(result["train_losses"], args.report_every),
        "valid_loss_selected": _selected_steps(result["valid_losses"], args.report_every),
        "grad_norm_selected": _selected_steps(result["grad_norms"], args.report_every),
        "epoch_history": result["epoch_history"],
        "train_only_normalization": _stats_payload(stats),
    }
    _write_json(output_dir / "loss_summary.json", loss_summary)

    predictions_path = output_dir / "predictions.npz"
    if args.save_predictions:
        np.savez_compressed(predictions_path, **predictions)

    print("")
    print("summary")
    print(f"  train loss initial/final: {result['train_losses'][0]:.8e} -> {result['train_losses'][-1]:.8e}")
    print(f"  valid loss initial/final: {result['valid_losses'][0]:.8e} -> {result['valid_losses'][-1]:.8e}")
    print(f"  final train raw DeltaT MSE: {result['train_metrics']['raw_delta_mse']:.8e}")
    print(f"  final valid raw DeltaT MSE: {result['valid_metrics']['raw_delta_mse']:.8e}")
    print(f"  final train recovered temperature MSE: {result['train_metrics']['recovered_temperature_mse']:.8e}")
    print(f"  final valid recovered temperature MSE: {result['valid_metrics']['recovered_temperature_mse']:.8e}")
    print(f"  gradient finite check: {result['grad_finite']}")
    print(f"  predictions saved: {bool(args.save_predictions)}")
    print(f"  predictions path: {predictions_path if args.save_predictions else 'not_written'}")
    print(f"  prediction sample count: {len(predictions)}")
    print("  checkpoint saved: False")
    print(f"  export smoke ok: {result['status_ok']}")
    return 0 if result["status_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
