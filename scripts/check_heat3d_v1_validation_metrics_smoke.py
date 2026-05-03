import argparse
from collections import defaultdict
from pathlib import Path
import sys

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
    DEFAULT_MANIFEST,
    DEFAULT_SUBSET,
    MODEL_CONFIG,
    _global_norm,
    _load_manifest,
    _make_groups,
    _manifest_split_ids,
    _metrics,
    _sample_root,
    _train_only_stats,
    _weighted_loss,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_metrics import (  # noqa: E402
    hotspot_coord_distance,
    hotspot_index,
    mae,
    max_abs_error,
    mse,
    peak_T_abs_error,
    rmse,
    top_k_hotspot_overlap,
)
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation metrics smoke for the Heat3D v1 supervised-small subset. "
            "This is diagnostic-only and is not a formal benchmark."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", "--runs", "--repeat-runs", dest="runs", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument(
        "--include-diagnostic-tests",
        action="store_true",
        help="Also print diagnostic-only rows for test_smoke/test_ood_* samples. Not a benchmark.",
    )
    return parser.parse_args()


def _fit_once(train_groups: list[dict], valid_groups: list[dict], stats: dict, steps: int, lr: float, seed: int) -> dict:
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
    for _ in range(steps):
        _, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = _global_norm(grads)
        grad_norms.append(grad_norm)
        grad_finite = grad_finite and bool(np.isfinite(grad_norm))
        params = tree.tree_map(lambda param, grad: param - lr * grad, params, grads)
        train_losses.append(float(loss_fn(params)))
        valid_losses.append(_metrics(model, params, valid_groups, stats)["normalized_loss"])

    train_metrics = _metrics(model, params, train_groups, stats)
    valid_metrics = _metrics(model, params, valid_groups, stats)
    return {
        "model": model,
        "params": params,
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "valid_losses": np.asarray(valid_losses, dtype=np.float64),
        "grad_norms": np.asarray(grad_norms, dtype=np.float64),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "grad_finite": grad_finite,
        "status_ok": (
            grad_finite
            and train_metrics["finite_ok"]
            and valid_metrics["finite_ok"]
            and train_metrics["shape_ok"]
            and valid_metrics["shape_ok"]
            and bool(np.all(np.isfinite(train_losses)))
            and bool(np.all(np.isfinite(valid_losses)))
        ),
    }


def _label_meta_count(sample_root: Path, sample_ids: list[str]) -> int:
    return sum(1 for sample_id in sample_ids if (sample_root / sample_id / "label_meta.json").is_file())


def _allclose_sequence(results: list[dict], key: str, atol: float, rtol: float) -> bool:
    return all(np.allclose(results[0][key], result[key], atol=atol, rtol=rtol) for result in results[1:])


def _run_repeatability(train_groups, valid_groups, stats, steps, lr, seed, runs, atol, rtol) -> tuple[dict, dict]:
    results = [_fit_once(train_groups, valid_groups, stats, steps, lr, seed) for _ in range(runs)]
    repeatability = {
        "train_loss_ok": _allclose_sequence(results, "train_losses", atol, rtol) if runs > 1 else True,
        "valid_loss_ok": _allclose_sequence(results, "valid_losses", atol, rtol) if runs > 1 else True,
        "grad_norm_ok": _allclose_sequence(results, "grad_norms", atol, rtol) if runs > 1 else True,
    }
    repeatability["ok"] = all(repeatability.values())
    return results[0], repeatability


def _predict_group(model, params, group: dict, stats: dict) -> np.ndarray:
    pred_normalized = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
    pred_delta = pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]
    return np.asarray(pred_delta)


def _metric_row(sample_id: str, split: str, predictor: str, pred_delta, true_delta, t_ref, coords, top_k: int) -> dict:
    pred_temperature = np.asarray(t_ref) + np.asarray(pred_delta)
    true_temperature = np.asarray(t_ref) + np.asarray(true_delta)
    pred_temperature_flat = pred_temperature.reshape(-1, 1)
    true_temperature_flat = true_temperature.reshape(-1, 1)
    return {
        "sample_id": sample_id,
        "split": split,
        "predictor": predictor,
        "raw_deltaT_mse": mse(pred_delta, true_delta),
        "recovered_T_rmse": rmse(pred_temperature, true_temperature),
        "recovered_T_mae": mae(pred_temperature, true_temperature),
        "recovered_T_max_abs_err": max_abs_error(pred_temperature, true_temperature),
        "true_peak_T": float(np.max(true_temperature_flat)),
        "pred_peak_T": float(np.max(pred_temperature_flat)),
        "peak_T_abs_err": peak_T_abs_error(pred_temperature_flat, true_temperature_flat),
        "true_hotspot_index": hotspot_index(true_temperature_flat),
        "pred_hotspot_index": hotspot_index(pred_temperature_flat),
        "hotspot_coord_distance": hotspot_coord_distance(pred_temperature_flat, true_temperature_flat, coords),
        "top_k_hotspot_overlap": top_k_hotspot_overlap(pred_temperature_flat, true_temperature_flat, k=top_k),
    }


def _collect_rows(groups: list[dict], example_by_id: dict, stats: dict, model=None, params=None, top_k: int = 5) -> list[dict]:
    rows = []
    for group in groups:
        trained_pred_delta = _predict_group(model, params, group, stats) if model is not None else None
        true_delta = np.asarray(group["target_delta_raw"])
        t_ref = np.asarray(group["t_ref"])
        for batch_index, sample_id in enumerate(group["sample_ids"]):
            example = example_by_id[sample_id]
            split = str(example.meta.get("split"))
            sample_true_delta = true_delta[batch_index]
            sample_t_ref = t_ref[batch_index]
            zero_delta = np.zeros_like(sample_true_delta)
            rows.append(
                _metric_row(
                    sample_id=sample_id,
                    split=split,
                    predictor="zero_delta_baseline",
                    pred_delta=zero_delta,
                    true_delta=sample_true_delta,
                    t_ref=sample_t_ref,
                    coords=example.condition.coords,
                    top_k=top_k,
                )
            )
            if trained_pred_delta is not None:
                rows.append(
                    _metric_row(
                        sample_id=sample_id,
                        split=split,
                        predictor="tiny_trained_prediction",
                        pred_delta=trained_pred_delta[batch_index],
                        true_delta=sample_true_delta,
                        t_ref=sample_t_ref,
                        coords=example.condition.coords,
                        top_k=top_k,
                    )
                )
    return rows


def _summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["predictor"])].append(row)

    summaries = []
    for (split, predictor), split_rows in sorted(grouped.items()):
        summaries.append(
            {
                "split": split,
                "predictor": predictor,
                "sample_count": len(split_rows),
                "mean_raw_deltaT_mse": float(np.mean([row["raw_deltaT_mse"] for row in split_rows])),
                "mean_recovered_T_rmse": float(np.mean([row["recovered_T_rmse"] for row in split_rows])),
                "mean_recovered_T_mae": float(np.mean([row["recovered_T_mae"] for row in split_rows])),
                "mean_recovered_T_max_abs_err": float(
                    np.mean([row["recovered_T_max_abs_err"] for row in split_rows])
                ),
                "mean_peak_T_abs_err": float(np.mean([row["peak_T_abs_err"] for row in split_rows])),
                "mean_hotspot_coord_distance": float(
                    np.mean([row["hotspot_coord_distance"] for row in split_rows])
                ),
                "mean_top_k_hotspot_overlap": float(
                    np.mean([row["top_k_hotspot_overlap"] for row in split_rows])
                ),
            }
        )
    return summaries


def _print_rows(rows: list[dict]) -> None:
    print("\nper-sample metrics")
    header = (
        "sample_id split predictor raw_deltaT_mse recovered_T_rmse "
        "recovered_T_mae recovered_T_max_abs_err true_peak_T pred_peak_T "
        "peak_T_abs_err true_hotspot_index pred_hotspot_index "
        "hotspot_coord_distance top_k_hotspot_overlap"
    )
    print(header)
    for row in rows:
        print(
            f"{row['sample_id']} {row['split']} {row['predictor']} "
            f"{row['raw_deltaT_mse']:.8e} {row['recovered_T_rmse']:.8e} "
            f"{row['recovered_T_mae']:.8e} {row['recovered_T_max_abs_err']:.8e} "
            f"{row['true_peak_T']:.8e} {row['pred_peak_T']:.8e} "
            f"{row['peak_T_abs_err']:.8e} {row['true_hotspot_index']} "
            f"{row['pred_hotspot_index']} {row['hotspot_coord_distance']:.8e} "
            f"{row['top_k_hotspot_overlap']:.8e}"
        )


def _print_summary(summaries: list[dict]) -> None:
    print("\nsplit summary")
    for summary in summaries:
        print(
            f"  {summary['split']} / {summary['predictor']}: "
            f"n={summary['sample_count']}, "
            f"mean_raw_deltaT_mse={summary['mean_raw_deltaT_mse']:.8e}, "
            f"mean_recovered_T_rmse={summary['mean_recovered_T_rmse']:.8e}, "
            f"mean_recovered_T_mae={summary['mean_recovered_T_mae']:.8e}, "
            f"mean_max_abs={summary['mean_recovered_T_max_abs_err']:.8e}, "
            f"mean_peak_err={summary['mean_peak_T_abs_err']:.8e}, "
            f"mean_hotspot_dist={summary['mean_hotspot_coord_distance']:.8e}, "
            f"mean_top_k_overlap={summary['mean_top_k_hotspot_overlap']:.8e}"
        )


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.runs < 1:
        raise ValueError("--repeat/--runs must be >= 1")

    manifest = _load_manifest(args.manifest)
    split_ids = _manifest_split_ids(manifest)
    train_ids = split_ids.get("train", [])
    valid_ids = split_ids.get("valid", [])
    test_ids = sorted(
        sample_id
        for split, ids in split_ids.items()
        if split not in {"train", "valid"}
        for sample_id in ids
    )
    eval_ids = train_ids + valid_ids
    if args.include_diagnostic_tests:
        eval_ids += test_ids

    sample_root = _sample_root(args.subset)
    label_meta_count = _label_meta_count(sample_root, eval_ids)
    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode="diag3")
    index_by_id = dataset.sample_index_by_id()
    examples = [dataset[index_by_id[sample_id]] for sample_id in eval_ids]
    example_by_id = {example.sample_id: example for example in examples}
    train_examples = [dataset[index_by_id[sample_id]] for sample_id in train_ids]
    valid_examples = [dataset[index_by_id[sample_id]] for sample_id in valid_ids]

    builder = Heat3DGraphBuilder()
    stats = _train_only_stats(train_examples)
    train_groups = _make_groups(train_examples, stats, builder)
    valid_groups = _make_groups(valid_examples, stats, builder)
    eval_groups = _make_groups(examples, stats, builder)

    first, repeatability = _run_repeatability(
        train_groups=train_groups,
        valid_groups=valid_groups,
        stats=stats,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        runs=args.runs,
        atol=args.atol,
        rtol=args.rtol,
    )
    rows = _collect_rows(
        eval_groups,
        example_by_id=example_by_id,
        stats=stats,
        model=first["model"],
        params=first["params"],
        top_k=args.top_k,
    )
    summaries = _summarize(rows)

    print("Heat3D v1 validation metrics smoke")
    print("  diagnostic only: not a formal benchmark, not model performance evidence")
    print(f"  subset path: {sample_root}")
    print(f"  manifest path: {args.manifest}")
    print(f"  label_meta files in evaluated samples: {label_meta_count}/{len(eval_ids)}")
    print(
        "  label source mode: "
        f"{'v2-label smoke diagnostics only' if label_meta_count else 'legacy supervised smoke labels'}"
    )
    print(f"  evaluated splits: {'train+valid+diagnostic_tests' if args.include_diagnostic_tests else 'train+valid'}")
    print(f"  train sample ids: {train_ids}")
    print(f"  valid sample ids: {valid_ids}")
    print(f"  diagnostic-only test sample ids ignored by default: {test_ids}")
    print("  route: relative BC features + zero_delta_u_bridge + normalized DeltaT target")
    print(f"  steps: {args.steps}")
    print(f"  lr: {args.lr}")
    print(f"  seed: {args.seed}")
    print(f"  repeat runs: {args.runs}")
    print(f"  repeatability ok: {repeatability['ok']}")
    print(f"  train loss initial/final: {first['train_losses'][0]:.8e} -> {first['train_losses'][-1]:.8e}")
    print(f"  valid loss initial/final: {first['valid_losses'][0]:.8e} -> {first['valid_losses'][-1]:.8e}")
    print(f"  gradient finite check: {first['grad_finite']}")

    _print_rows(rows)
    _print_summary(summaries)

    status_ok = (
        first["status_ok"]
        and repeatability["ok"]
        and bool(np.all(np.isfinite([row["raw_deltaT_mse"] for row in rows])))
        and bool(np.all(np.isfinite([row["recovered_T_rmse"] for row in rows])))
    )
    print("\nsummary")
    print(f"  per-sample metric rows: {len(rows)}")
    print(f"  split summaries: {len(summaries)}")
    print("  checkpoint saved: False")
    print("  output file written: False")
    print("  formal benchmark: False")
    print(f"  validation metrics smoke ok: {status_ok}")
    return 0 if status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
