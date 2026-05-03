import argparse
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
    _make_groups as _make_shape_groups,
    _metrics as _group_metrics,
    _train_only_stats as _group_train_only_stats,
    _weighted_loss as _group_weighted_loss,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset
from rigno.heat3d_v1_supervised import default_v1_supervised_samples_dir
from rigno.models.operator import Inputs
from rigno.models.rigno import RIGNO as GraphNeuralOperator


TARGET_SAMPLE_IDS = ("sample_000", "sample_005")
EPS = 1e-8
MODEL_CONFIG = {
    "num_outputs": 1,
    "processor_steps": 2,
    "node_latent_size": 16,
    "edge_latent_size": 16,
    "mlp_hidden_layers": 1,
    "concatenate_tau": False,
    "concatenate_t": False,
    "conditioned_normalization": False,
    "cond_norm_hidden_size": 16,
    "p_edge_masking": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tiny training smoke for v1 zero-delta bridge with relative BC features. "
            "This is not a real training experiment."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_supervised_samples_dir(REPO_DIR),
        help="Supervised smoke samples directory.",
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=None,
        help="Optional supervised subset root or samples directory. Preserves positional path compatibility.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional sample ids. Defaults to the legacy two-sample tiny-training smoke.",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-7)
    return parser.parse_args()


def _sample_by_id(dataset: Heat3DV1NativeSupervisedDataset, sample_id: str):
    index_by_id = dataset.sample_index_by_id()
    if sample_id not in index_by_id:
        raise ValueError(f"Missing sample {sample_id!r}")
    return dataset[index_by_id[sample_id]]


def _sample_root(path: Path) -> Path:
    samples = path / "samples"
    if samples.is_dir():
        return samples
    return path


def _select_examples(path: Path, sample_ids: tuple[str, ...]):
    dataset = Heat3DV1NativeSupervisedDataset(path, k_encoding_mode="diag3")
    return [_sample_by_id(dataset, sample_id) for sample_id in sample_ids]


def _build_batch_metadata(builder: Heat3DGraphBuilder, coords_list: list[np.ndarray]):
    metadata_list = [builder.build_metadata(coords) for coords in coords_list]
    same_coords = all(np.array_equal(coords_list[0], coords) for coords in coords_list[1:])
    if same_coords:
        return tree.tree_map(
            lambda value: jnp.repeat(value, repeats=len(coords_list), axis=0),
            metadata_list[0],
        ), True
    return tree.tree_map(lambda *values: jnp.concatenate(values, axis=0), *metadata_list), False


def _safe_standardize(values: jnp.ndarray):
    mean = jnp.mean(values, axis=(0, 1, 2), keepdims=True)
    std = jnp.std(values, axis=(0, 1, 2), keepdims=True)
    safe_std = jnp.where(std < EPS, 1.0, std)
    normalized = (values - mean) / safe_std
    return normalized, mean, std, safe_std


def _normalize_coords(coords: jnp.ndarray):
    coord_min = jnp.min(coords, axis=(0, 1, 2), keepdims=True)
    coord_max = jnp.max(coords, axis=(0, 1, 2), keepdims=True)
    coord_span = jnp.where((coord_max - coord_min) < EPS, 1.0, coord_max - coord_min)
    normalized = 2.0 * ((coords - coord_min) / coord_span) - 1.0
    return normalized, coord_min, coord_max


def _target_excluded(feature_names: tuple[str, ...]) -> bool:
    blocked = {"temperature", "target_temperature", "target_u"}
    return all(name not in blocked for name in feature_names)


def _raw_bc_excluded(feature_names: tuple[str, ...]) -> bool:
    return "top_T_inf" not in feature_names and "bottom_T_fixed" not in feature_names


def _make_contract(examples):
    bridges = [
        example.build_temperature_rise_legacy_inputs_from_relative_features(
            bridge_policy="zero_delta_u_bridge",
        )
        for example in examples
    ]
    feature_names = bridges[0].condition_feature_names
    for bridge in bridges[1:]:
        if bridge.condition_feature_names != feature_names:
            raise ValueError("Relative condition feature-name contract mismatch across samples")

    raw_c = jnp.concatenate([bridge.legacy_inputs.c for bridge in bridges], axis=0)
    raw_u = jnp.concatenate([bridge.legacy_inputs.u for bridge in bridges], axis=0)
    raw_coords = jnp.concatenate([bridge.legacy_inputs.x_inp for bridge in bridges], axis=0)
    target_delta = jnp.concatenate([bridge.target_delta_u for bridge in bridges], axis=0)
    t_ref = jnp.concatenate([bridge.t_ref for bridge in bridges], axis=0)
    target_temperature = t_ref + target_delta

    normalized_c, c_mean, c_std, c_safe_std = _safe_standardize(raw_c)
    normalized_target, target_mean, target_std, target_safe_std = _safe_standardize(target_delta)
    normalized_coords, coord_min, coord_max = _normalize_coords(raw_coords)

    inputs = Inputs(
        u=raw_u,
        c=normalized_c,
        x_inp=normalized_coords,
        x_out=normalized_coords,
        t=None,
        tau=None,
    )
    stats = {
        "feature_names": feature_names,
        "raw_c": raw_c,
        "raw_u": raw_u,
        "c_mean": c_mean,
        "c_std": c_std,
        "c_safe_std": c_safe_std,
        "target_delta": target_delta,
        "target_delta_mean": target_mean,
        "target_delta_std": target_std,
        "target_delta_safe_std": target_safe_std,
        "normalized_target_delta": normalized_target,
        "t_ref": t_ref,
        "target_temperature": target_temperature,
        "coord_min": coord_min,
        "coord_max": coord_max,
        "bridges": bridges,
    }
    return inputs, normalized_target, stats


def _global_norm(tree_value) -> float:
    leaves = tree.tree_leaves(tree_value)
    total = sum(float(jnp.sum(jnp.square(leaf))) for leaf in leaves)
    return float(np.sqrt(total))


def _run_once(path: Path, sample_ids: tuple[str, ...], steps: int, lr: float, seed: int) -> dict:
    examples = _select_examples(path, sample_ids)
    builder = Heat3DGraphBuilder()
    stats = _group_train_only_stats(examples)
    groups = _make_shape_groups(examples, stats, builder)
    model = GraphNeuralOperator(**MODEL_CONFIG)
    params = model.init(
        jax.random.PRNGKey(seed),
        inputs=groups[0]["inputs"],
        graphs=groups[0]["graphs"],
    )["params"]

    def loss_fn(current_params):
        return _group_weighted_loss(model, current_params, groups)

    losses = [float(loss_fn(params))]
    grad_norms = []
    for _ in range(steps):
        _, grads = jax.value_and_grad(loss_fn)(params)
        grad_norms.append(_global_norm(grads))
        params = tree.tree_map(lambda param, grad: param - lr * grad, params, grads)
        losses.append(float(loss_fn(params)))

    metrics = _group_metrics(model, params, groups, stats)
    raw_delta_mse = metrics["raw_delta_mse"]
    recovered_temperature_mse = metrics["recovered_temperature_mse"]

    finite_ok = bool(
        jnp.all(jnp.isfinite(jnp.asarray(losses)))
        and jnp.all(jnp.isfinite(jnp.asarray(grad_norms)))
        and metrics["finite_ok"]
        and jnp.isfinite(raw_delta_mse)
        and jnp.isfinite(recovered_temperature_mse)
    )
    shape_ok = metrics["shape_ok"] and all(
        group["inputs"].u.shape[0] == len(group["sample_ids"])
        and group["inputs"].c.shape[-1] == len(stats["feature_names"])
        for group in groups
    )
    target_excluded = _target_excluded(stats["feature_names"])
    raw_bc_excluded = _raw_bc_excluded(stats["feature_names"])
    zero_u_ok = all(bool(jnp.max(jnp.abs(group["inputs"].u)) <= EPS) for group in groups)
    loss_stable = finite_ok and max(losses) < 1e6
    status_ok = (
        finite_ok
        and shape_ok
        and target_excluded
        and raw_bc_excluded
        and zero_u_ok
        and loss_stable
    )

    return {
        "examples": examples,
        "groups": groups,
        "stats": stats,
        "losses": np.asarray(losses, dtype=np.float64),
        "grad_norms": np.asarray(grad_norms, dtype=np.float64),
        "raw_delta_mse": raw_delta_mse,
        "recovered_temperature_mse": recovered_temperature_mse,
        "finite_ok": finite_ok,
        "shape_ok": shape_ok,
        "target_excluded": target_excluded,
        "raw_bc_excluded": raw_bc_excluded,
        "zero_u_ok": zero_u_ok,
        "loss_stable": loss_stable,
        "status_ok": status_ok,
    }


def _max_abs_delta(values: list[np.ndarray | float]) -> float:
    base = values[0]
    return max(float(np.max(np.abs(value - base))) for value in values[1:]) if len(values) > 1 else 0.0


def _print_stats(result: dict, sample_ids: tuple[str, ...], steps: int, lr: float, seed: int) -> None:
    stats = result["stats"]
    groups = result["groups"]
    c_mean = np.asarray(stats["condition_mean"]).reshape(-1)
    c_std = np.asarray(stats["condition_std"]).reshape(-1)
    target_delta = np.concatenate(
        [np.asarray(group["target_delta_raw"]).reshape(-1, 1) for group in groups],
        axis=0,
    )
    normalized_target = np.concatenate(
        [np.asarray(group["target_normalized"]).reshape(-1, 1) for group in groups],
        axis=0,
    )
    target_delta_mean = float(np.asarray(stats["target_delta_mean"]).reshape(-1)[0])
    target_delta_std = float(np.asarray(stats["target_delta_std"]).reshape(-1)[0])
    normalized_target_mean = float(np.mean(normalized_target))
    normalized_target_std = float(np.std(normalized_target))

    print("v1 zero-delta relative-BC tiny training smoke")
    print("  purpose: forward/backward/optimizer/loss/recovery smoke only, not a real experiment")
    print(f"  sample_ids: {sample_ids}")
    print("  k_encoding_mode: diag3")
    print("  bridge: zero_delta_u_bridge")
    print("  target: DeltaT = target_temperature - T_ref")
    print("  recovery: T_pred = T_ref + denormalized_DeltaT_pred")
    print("  loss contract: MSE on normalized DeltaT")
    print(f"  steps: {steps}")
    print(f"  lr: {lr}")
    print(f"  seed: {seed}")

    print("\ncontract")
    print(f"  feature_names: {stats['feature_names']}")
    print(
        "  graph-shape groups: "
        f"{[(group['name'], group['sample_ids'], tuple(group['inputs'].u.shape), tuple(group['target_normalized'].shape)) for group in groups]}"
    )
    print(f"  target excluded from inputs: {result['target_excluded']}")
    print(f"  raw absolute BC temperatures excluded: {result['raw_bc_excluded']}")
    print(f"  legacy_inputs.u is zero_delta_field: {result['zero_u_ok']}")

    print("\ncondition feature stats")
    for name, mean, std in zip(stats["feature_names"], c_mean, c_std, strict=True):
        print(f"  {name}: mean={float(mean):.6e}, std={float(std):.6e}")

    print("\ntarget stats")
    print(
        "  target_delta_u min/max/mean/std: "
        f"{float(np.min(target_delta)):.6f}, {float(np.max(target_delta)):.6f}, "
        f"{float(np.mean(target_delta)):.6f}, {float(np.std(target_delta)):.6f}"
    )
    print(f"  target_delta normalization mean/std: {target_delta_mean:.6f}, {target_delta_std:.6f}")
    print(
        "  normalized target_delta_u mean/std: "
        f"{normalized_target_mean:.6e}, {normalized_target_std:.6e}"
    )

    print("\ntiny training smoke")
    print(f"  model_config: {MODEL_CONFIG}")
    print(f"  normalized losses: {[float(value) for value in result['losses']]}")
    print(f"  grad norms: {[float(value) for value in result['grad_norms']]}")
    print(f"  raw DeltaT MSE after smoke steps: {result['raw_delta_mse']}")
    print(f"  recovered temperature MSE after smoke steps: {result['recovered_temperature_mse']}")
    print(f"  finite ok: {result['finite_ok']}")
    print(f"  shape ok: {result['shape_ok']}")
    print(f"  loss stable: {result['loss_stable']}")
    print(f"  tiny training smoke ok: {result['status_ok']}")


def _print_repeatability(results: list[dict], atol: float, rtol: float) -> bool:
    loss_arrays = [result["losses"] for result in results]
    grad_arrays = [result["grad_norms"] for result in results]
    raw_delta_mses = [result["raw_delta_mse"] for result in results]
    recovered_mses = [result["recovered_temperature_mse"] for result in results]

    exact_losses = all(np.array_equal(loss_arrays[0], value) for value in loss_arrays[1:])
    exact_grad_norms = all(np.array_equal(grad_arrays[0], value) for value in grad_arrays[1:])
    exact_raw_delta = all(raw_delta_mses[0] == value for value in raw_delta_mses[1:])
    exact_recovered = all(recovered_mses[0] == value for value in recovered_mses[1:])

    close_losses = all(np.allclose(loss_arrays[0], value, atol=atol, rtol=rtol) for value in loss_arrays[1:])
    close_grad_norms = all(np.allclose(grad_arrays[0], value, atol=atol, rtol=rtol) for value in grad_arrays[1:])
    close_raw_delta = all(np.isclose(raw_delta_mses[0], value, atol=atol, rtol=rtol) for value in raw_delta_mses[1:])
    close_recovered = all(np.isclose(recovered_mses[0], value, atol=atol, rtol=rtol) for value in recovered_mses[1:])
    status_ok = close_losses and close_grad_norms and close_raw_delta and close_recovered

    print("\ncheckpoint-free repeatability")
    for index, result in enumerate(results):
        print(f"  run_{index}:")
        print(f"    losses: {[float(value) for value in result['losses']]}")
        print(f"    grad_norms: {[float(value) for value in result['grad_norms']]}")
        print(f"    raw_delta_mse: {result['raw_delta_mse']}")
        print(f"    recovered_temperature_mse: {result['recovered_temperature_mse']}")

    print("  comparison:")
    print(f"    exact losses: {exact_losses}")
    print(f"    exact grad norms: {exact_grad_norms}")
    print(f"    exact raw DeltaT MSE: {exact_raw_delta}")
    print(f"    exact recovered temperature MSE: {exact_recovered}")
    print(f"    max loss delta: {_max_abs_delta(loss_arrays):.6e}")
    print(f"    max grad norm delta: {_max_abs_delta(grad_arrays):.6e}")
    print(f"    max raw DeltaT MSE delta: {_max_abs_delta(raw_delta_mses):.6e}")
    print(f"    max recovered temperature MSE delta: {_max_abs_delta(recovered_mses):.6e}")
    print(f"    within tolerance losses: {close_losses}")
    print(f"    within tolerance grad norms: {close_grad_norms}")
    print(f"    within tolerance raw DeltaT MSE: {close_raw_delta}")
    print(f"    within tolerance recovered temperature MSE: {close_recovered}")
    print(f"    repeatability smoke ok: {status_ok}")
    return status_ok


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    target_path = _sample_root(args.subset if args.subset is not None else args.path)
    sample_ids = (
        tuple(args.sample_ids)
        if args.sample_ids is not None and len(args.sample_ids) > 0
        else TARGET_SAMPLE_IDS
    )
    results = [
        _run_once(target_path, sample_ids, args.steps, args.lr, args.seed)
        for _ in range(args.runs)
    ]
    _print_stats(results[0], sample_ids=sample_ids, steps=args.steps, lr=args.lr, seed=args.seed)

    repeatability_ok = True
    if args.runs >= 2:
        repeatability_ok = _print_repeatability(results, atol=args.atol, rtol=args.rtol)

    all_runs_ok = all(result["status_ok"] for result in results)
    status_ok = all_runs_ok and repeatability_ok

    print("\nsummary")
    print(f"  all runs tiny training smoke ok: {all_runs_ok}")
    print(f"  repeatability ok: {repeatability_ok}")
    print("  zero_delta bridge replaces legacy u=k_x for this smoke path: True")
    print("  formal training experiment: False")
    print(f"  zero_delta relative-BC tiny training smoke ok: {status_ok}")
    return 0 if status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
