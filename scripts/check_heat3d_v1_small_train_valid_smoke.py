import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v1_schema import find_sample_dirs, load_sample_meta
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset
from rigno.models.operator import Inputs
from rigno.models.rigno import RIGNO as GraphNeuralOperator


DEFAULT_MANIFEST = REPO_DIR / "configs" / "heat3d_v1_supervised_small_manifest.json"
DEFAULT_SUBSET = (
    REPO_DIR
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_supervised_small"
)
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
            "Train/valid smoke for the 16-sample Heat3D v1 supervised-small subset. "
            "This is not a formal training experiment."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional full-batch grouped smoke epochs. When set, this overrides "
            "--steps and runs one optimizer update per epoch."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", "--repeat-runs", dest="runs", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument(
        "--radius-policy",
        choices=("legacy_kdtree_mean4", "discrete_physical_coverage"),
        default="legacy_kdtree_mean4",
    )
    parser.add_argument(
        "--coverage-repair-policy",
        choices=("none", "nearest_rnode"),
        default="none",
    )
    parser.add_argument("--repair-p2r", dest="repair_p2r", action="store_true", default=True)
    parser.add_argument("--no-repair-p2r", dest="repair_p2r", action="store_false")
    parser.add_argument("--repair-r2p", dest="repair_r2p", action="store_true", default=True)
    parser.add_argument("--no-repair-r2p", dest="repair_r2p", action="store_false")
    parser.add_argument("--min-physical-coverage", type=int, default=1)
    return parser.parse_args()


def _sample_root(path: Path) -> Path:
    samples = path / "samples"
    if samples.is_dir():
        return samples
    return path


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _manifest_split_ids(manifest: dict) -> dict[str, list[str]]:
    split_ids: dict[str, list[str]] = {}
    for sample in manifest.get("samples", []):
        split_ids.setdefault(str(sample.get("split")), []).append(str(sample.get("sample_id")))
    return {split: sorted(ids) for split, ids in split_ids.items()}


def _subset_split_ids(sample_root: Path) -> dict[str, list[str]]:
    split_ids: dict[str, list[str]] = {}
    for sample_dir in find_sample_dirs(sample_root):
        meta = load_sample_meta(sample_dir)
        split_ids.setdefault(str(meta.get("split")), []).append(str(meta.get("sample_id")))
    return {split: sorted(ids) for split, ids in split_ids.items()}


def _resolve_split_ids(manifest: dict, sample_root: Path) -> tuple[dict[str, list[str]], str]:
    manifest_ids = _manifest_split_ids(manifest)
    manifest_sample_ids = {
        sample_id
        for ids in manifest_ids.values()
        for sample_id in ids
    }
    if manifest_sample_ids and all((sample_root / sample_id).is_dir() for sample_id in manifest_sample_ids):
        return manifest_ids, "manifest"
    subset_ids = _subset_split_ids(sample_root)
    if subset_ids:
        return subset_ids, "subset_sample_meta"
    return manifest_ids, "manifest"


def _metadata_shape_signature(metadata) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(value.shape)
        for value in tree.tree_leaves(metadata)
        if hasattr(value, "shape")
    )


def _bridge_for(example):
    return example.build_temperature_rise_legacy_inputs_from_relative_features(
        bridge_policy="zero_delta_u_bridge"
    )


def _safe_stats(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(array, axis=0, keepdims=True)
    std = np.std(array, axis=0, keepdims=True)
    return mean, np.where(std < EPS, 1.0, std)


def _train_only_stats(examples) -> dict:
    c_values = []
    delta_values = []
    coord_values = []
    feature_names = None
    for example in examples:
        bridge = _bridge_for(example)
        names = bridge.condition_feature_names
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Relative condition feature-name mismatch in train split")

        c_values.append(np.asarray(bridge.legacy_inputs.c).reshape(-1, len(names)))
        delta_values.append(np.asarray(bridge.target_delta_u).reshape(-1, 1))
        coord_values.append(np.asarray(bridge.legacy_inputs.x_inp).reshape(-1, 3))

    c_all = np.concatenate(c_values, axis=0)
    delta_all = np.concatenate(delta_values, axis=0)
    coord_all = np.concatenate(coord_values, axis=0)
    c_mean, c_std = _safe_stats(c_all)
    delta_mean, delta_std = _safe_stats(delta_all)
    coord_min = np.min(coord_all, axis=0, keepdims=True)
    coord_max = np.max(coord_all, axis=0, keepdims=True)
    coord_span = np.where((coord_max - coord_min) < EPS, 1.0, coord_max - coord_min)
    return {
        "feature_names": tuple(feature_names or ()),
        "condition_mean": c_mean.reshape(1, 1, 1, -1),
        "condition_std": c_std.reshape(1, 1, 1, -1),
        "target_delta_mean": delta_mean.reshape(1, 1, 1, 1),
        "target_delta_std": delta_std.reshape(1, 1, 1, 1),
        "coord_min": coord_min.reshape(1, 1, 1, 3),
        "coord_span": coord_span.reshape(1, 1, 1, 3),
    }


def _normalize_coords(coords, stats: dict):
    return 2.0 * ((coords - stats["coord_min"]) / stats["coord_span"]) - 1.0


def _build_batch_metadata(builder: Heat3DGraphBuilder, coords_list: list[np.ndarray]):
    metadata_list = [builder.build_metadata(coords) for coords in coords_list]
    same_coords = all(np.array_equal(coords_list[0], coords) for coords in coords_list[1:])
    if same_coords:
        return tree.tree_map(
            lambda value: jnp.repeat(value, repeats=len(coords_list), axis=0),
            metadata_list[0],
        ), True
    return tree.tree_map(lambda *values: jnp.concatenate(values, axis=0), *metadata_list), False


def _make_batch_group(group_name: str, examples, stats: dict, builder: Heat3DGraphBuilder) -> dict:
    bridges = [_bridge_for(example) for example in examples]
    feature_names = bridges[0].condition_feature_names
    for bridge in bridges[1:]:
        if bridge.condition_feature_names != feature_names:
            raise ValueError(f"Feature-name mismatch in {group_name}")

    raw_u = jnp.concatenate([bridge.legacy_inputs.u for bridge in bridges], axis=0)
    raw_c = jnp.concatenate([bridge.legacy_inputs.c for bridge in bridges], axis=0)
    raw_coords = jnp.concatenate([bridge.legacy_inputs.x_inp for bridge in bridges], axis=0)
    target_delta = jnp.concatenate([bridge.target_delta_u for bridge in bridges], axis=0)
    t_ref = jnp.concatenate([bridge.t_ref for bridge in bridges], axis=0)

    c = (raw_c - stats["condition_mean"]) / stats["condition_std"]
    target = (target_delta - stats["target_delta_mean"]) / stats["target_delta_std"]
    coords = _normalize_coords(raw_coords, stats)
    inputs = Inputs(u=raw_u, c=c, x_inp=coords, x_out=coords, t=None, tau=None)
    metadata, shared = _build_batch_metadata(
        builder=builder,
        coords_list=[example.condition.coords for example in examples],
    )
    graphs = builder.build_graphs(metadata)
    return {
        "name": group_name,
        "sample_ids": tuple(example.sample_id for example in examples),
        "split": examples[0].meta.get("split"),
        "inputs": inputs,
        "target_normalized": target,
        "target_delta_raw": target_delta,
        "target_temperature": t_ref + target_delta,
        "t_ref": t_ref,
        "graphs": graphs,
        "metadata": metadata,
        "shared_metadata": shared,
        "feature_names": feature_names,
    }


def _group_examples(examples, builder: Heat3DGraphBuilder) -> dict:
    groups: dict[tuple[int, tuple[str, ...], tuple[tuple[int, ...], ...]], list] = {}
    for example in examples:
        bridge = _bridge_for(example)
        signature = _metadata_shape_signature(builder.build_metadata(example.condition.coords))
        key = (
            example.condition.coords.shape[0],
            bridge.condition_feature_names,
            signature,
        )
        groups.setdefault(key, []).append(example)
    result = {}
    for idx, ((n_points, feature_names, _signature), group_examples) in enumerate(groups.items(), start=1):
        result[f"group_{idx}_N{n_points}_F{len(feature_names)}"] = group_examples
    return result


def _make_groups(examples, stats: dict, builder: Heat3DGraphBuilder) -> list[dict]:
    grouped = _group_examples(examples, builder)
    return [
        _make_batch_group(name, group_examples, stats, builder)
        for name, group_examples in grouped.items()
    ]


def _weighted_loss(model, params, groups: list[dict]):
    weighted = 0.0
    count = 0
    for group in groups:
        pred = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        group_loss = jnp.mean(jnp.square(pred - group["target_normalized"]))
        n = group["target_normalized"].shape[0]
        weighted = weighted + group_loss * n
        count += int(n)
    return weighted / max(count, 1)


def _metrics(model, params, groups: list[dict], stats: dict) -> dict:
    normalized_losses = []
    raw_delta_mses = []
    recovered_mses = []
    finite_ok = True
    shape_ok = True
    for group in groups:
        pred_normalized = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        normalized_losses.append(jnp.mean(jnp.square(pred_normalized - group["target_normalized"])))
        pred_delta = pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]
        recovered = group["t_ref"] + pred_delta
        raw_delta_mses.append(jnp.mean(jnp.square(pred_delta - group["target_delta_raw"])))
        recovered_mses.append(jnp.mean(jnp.square(recovered - group["target_temperature"])))
        finite_ok = (
            finite_ok
            and bool(jnp.all(jnp.isfinite(pred_normalized)))
            and bool(jnp.all(jnp.isfinite(pred_delta)))
            and bool(jnp.all(jnp.isfinite(recovered)))
        )
        shape_ok = shape_ok and pred_normalized.shape == group["target_normalized"].shape

    return {
        "normalized_loss": float(jnp.mean(jnp.asarray(normalized_losses))),
        "raw_delta_mse": float(jnp.mean(jnp.asarray(raw_delta_mses))),
        "recovered_temperature_mse": float(jnp.mean(jnp.asarray(recovered_mses))),
        "finite_ok": finite_ok,
        "shape_ok": shape_ok,
    }


def _global_norm(tree_value) -> float:
    leaves = tree.tree_leaves(tree_value)
    total = sum(float(jnp.sum(jnp.square(leaf))) for leaf in leaves)
    return float(np.sqrt(total))


def _run_once(train_groups: list[dict], valid_groups: list[dict], stats: dict, steps: int, lr: float, seed: int) -> dict:
    model = GraphNeuralOperator(**MODEL_CONFIG)
    first_group = train_groups[0]
    params = model.init(
        jax.random.PRNGKey(seed),
        inputs=first_group["inputs"],
        graphs=first_group["graphs"],
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


def _max_abs_delta(values) -> float:
    base = values[0]
    return max(float(np.max(np.abs(value - base))) for value in values[1:]) if len(values) > 1 else 0.0


def _selected_steps(values: np.ndarray, report_every: int) -> list[tuple[int, float]]:
    if report_every < 1:
        raise ValueError("--report-every must be >= 1")
    selected = {0, len(values) - 1}
    selected.update(range(report_every, len(values), report_every))
    return [(index, float(values[index])) for index in sorted(selected)]


def _repeatability(results: list[dict], atol: float, rtol: float) -> dict:
    train_loss_ok = all(
        np.allclose(results[0]["train_losses"], result["train_losses"], atol=atol, rtol=rtol)
        for result in results[1:]
    )
    valid_loss_ok = all(
        np.allclose(results[0]["valid_losses"], result["valid_losses"], atol=atol, rtol=rtol)
        for result in results[1:]
    )
    grad_ok = all(
        np.allclose(results[0]["grad_norms"], result["grad_norms"], atol=atol, rtol=rtol)
        for result in results[1:]
    )
    train_raw_ok = all(
        np.isclose(
            results[0]["train_metrics"]["raw_delta_mse"],
            result["train_metrics"]["raw_delta_mse"],
            atol=atol,
            rtol=rtol,
        )
        for result in results[1:]
    )
    valid_raw_ok = all(
        np.isclose(
            results[0]["valid_metrics"]["raw_delta_mse"],
            result["valid_metrics"]["raw_delta_mse"],
            atol=atol,
            rtol=rtol,
        )
        for result in results[1:]
    )
    return {
        "train_loss_ok": train_loss_ok,
        "valid_loss_ok": valid_loss_ok,
        "grad_ok": grad_ok,
        "train_raw_ok": train_raw_ok,
        "valid_raw_ok": valid_raw_ok,
        "max_train_loss_delta": _max_abs_delta([result["train_losses"] for result in results]),
        "max_valid_loss_delta": _max_abs_delta([result["valid_losses"] for result in results]),
        "max_grad_delta": _max_abs_delta([result["grad_norms"] for result in results]),
        "ok": train_loss_ok and valid_loss_ok and grad_ok and train_raw_ok and valid_raw_ok,
    }


def _print_group_summary(label: str, groups: list[dict]) -> None:
    print(f"{label} graph-shape groups:")
    for group in groups:
        print(
            f"  {group['name']}: samples={group['sample_ids']}, "
            f"u={tuple(group['inputs'].u.shape)}, c={tuple(group['inputs'].c.shape)}, "
            f"target={tuple(group['target_normalized'].shape)}, shared_metadata={group['shared_metadata']}"
        )


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.report_every < 1:
        raise ValueError("--report-every must be >= 1")

    manifest = _load_manifest(args.manifest)
    sample_root = _sample_root(args.subset)
    split_ids, split_source = _resolve_split_ids(manifest, sample_root)
    train_ids = split_ids.get("train", [])
    valid_ids = split_ids.get("valid", [])
    ignored_ids = sorted(
        sample_id
        for split, ids in split_ids.items()
        if split not in {"train", "valid"}
        for sample_id in ids
    )
    if not train_ids or not valid_ids:
        raise ValueError(f"Expected non-empty train and valid splits, found {len(train_ids)}/{len(valid_ids)}")

    missing_temperature = [
        sample_id
        for sample_id in train_ids + valid_ids
        if not (sample_root / sample_id / "temperature.npy").is_file()
    ]
    if missing_temperature:
        raise FileNotFoundError(f"Missing temperature.npy for samples: {missing_temperature}")

    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode="diag3")
    index_by_id = dataset.sample_index_by_id()
    train_examples = [dataset[index_by_id[sample_id]] for sample_id in train_ids]
    valid_examples = [dataset[index_by_id[sample_id]] for sample_id in valid_ids]
    builder = Heat3DGraphBuilder(
        radius_policy=args.radius_policy,
        coverage_repair_policy=args.coverage_repair_policy,
        repair_p2r=args.repair_p2r,
        repair_r2p=args.repair_r2p,
        min_physical_coverage=args.min_physical_coverage,
    )
    stats = _train_only_stats(train_examples)
    train_groups = _make_groups(train_examples, stats, builder)
    valid_groups = _make_groups(valid_examples, stats, builder)
    training_steps = args.epochs if args.epochs is not None else args.steps

    results = [_run_once(train_groups, valid_groups, stats, training_steps, args.lr, args.seed) for _ in range(args.runs)]
    repeatability = _repeatability(results, args.atol, args.rtol) if args.runs >= 2 else {"ok": True}
    first = results[0]

    print("Heat3D v1 small train/valid smoke")
    print("  this is smoke only, not model performance")
    print(f"  subset path: {sample_root}")
    print(f"  manifest path: {args.manifest}")
    print(f"  split source: {split_source}")
    split_counts = {split: len(ids) for split, ids in split_ids.items()}
    print(f"  split counts: {split_counts}")
    print(f"  train sample ids: {train_ids}")
    print(f"  valid sample ids: {valid_ids}")
    print(f"  ignored test sample ids: {ignored_ids}")
    print(f"  feature names: {stats['feature_names']}")
    print("  target name: DeltaT")
    print("  route: relative BC features + zero_delta_u_bridge + normalized DeltaT target")
    print(f"  steps: {training_steps}")
    print(f"  epochs: {args.epochs if args.epochs is not None else 'not_set'}")
    print(f"  lr: {args.lr}")
    print(f"  seed: {args.seed}")
    print(f"  repeat runs: {args.runs}")
    print(f"  report every: {args.report_every}")
    print(f"  graph builder config: {builder.config}")

    print("\ntrain-only normalization stats")
    c_mean = stats["condition_mean"].reshape(-1)
    c_std = stats["condition_std"].reshape(-1)
    for name, mean, std in zip(stats["feature_names"], c_mean, c_std, strict=True):
        print(f"  {name}: mean={float(mean):.6e}, safe_std={float(std):.6e}")
    print(
        "  target_delta: "
        f"mean={float(stats['target_delta_mean'].reshape(-1)[0]):.6e}, "
        f"safe_std={float(stats['target_delta_std'].reshape(-1)[0]):.6e}"
    )

    print("")
    _print_group_summary("train", train_groups)
    _print_group_summary("valid", valid_groups)

    print("\nrun_0 smoke results")
    print(f"  train normalized loss selected steps: {_selected_steps(first['train_losses'], args.report_every)}")
    print(f"  valid normalized loss selected steps: {_selected_steps(first['valid_losses'], args.report_every)}")
    print(f"  grad norm selected steps: {_selected_steps(first['grad_norms'], args.report_every)}")
    print(f"  final train raw DeltaT MSE: {first['train_metrics']['raw_delta_mse']}")
    print(f"  final valid raw DeltaT MSE: {first['valid_metrics']['raw_delta_mse']}")
    print(f"  final train recovered temperature MSE: {first['train_metrics']['recovered_temperature_mse']}")
    print(f"  final valid recovered temperature MSE: {first['valid_metrics']['recovered_temperature_mse']}")
    print(f"  finite check: {first['train_metrics']['finite_ok'] and first['valid_metrics']['finite_ok']}")
    print(f"  shape check: {first['train_metrics']['shape_ok'] and first['valid_metrics']['shape_ok']}")
    print(f"  gradient finite check: {first['grad_finite']}")

    print("\nrepeatability")
    for index, result in enumerate(results):
        print(f"  run_{index}:")
        print(f"    train losses selected: {_selected_steps(result['train_losses'], args.report_every)}")
        print(f"    valid losses selected: {_selected_steps(result['valid_losses'], args.report_every)}")
        print(f"    grad norms selected: {_selected_steps(result['grad_norms'], args.report_every)}")
        print(f"    train raw DeltaT MSE: {result['train_metrics']['raw_delta_mse']}")
        print(f"    valid raw DeltaT MSE: {result['valid_metrics']['raw_delta_mse']}")
        print(f"    train recovered temperature MSE: {result['train_metrics']['recovered_temperature_mse']}")
        print(f"    valid recovered temperature MSE: {result['valid_metrics']['recovered_temperature_mse']}")
    print(f"  max train loss delta: {repeatability.get('max_train_loss_delta', 0.0):.6e}")
    print(f"  max valid loss delta: {repeatability.get('max_valid_loss_delta', 0.0):.6e}")
    print(f"  max grad norm delta: {repeatability.get('max_grad_delta', 0.0):.6e}")
    print(f"  repeatability smoke ok: {repeatability['ok']}")

    status_ok = all(result["status_ok"] for result in results) and bool(repeatability["ok"])
    train_loss_decreased = bool(first["train_losses"][-1] < first["train_losses"][0])
    valid_loss_decreased = bool(first["valid_losses"][-1] < first["valid_losses"][0])
    print("\nsummary")
    print(f"  all run status ok: {all(result['status_ok'] for result in results)}")
    print(f"  repeatability ok: {repeatability['ok']}")
    print(f"  train loss decreased initial_to_final: {train_loss_decreased}")
    print(f"  valid loss decreased initial_to_final: {valid_loss_decreased}")
    print("  checkpoint saved: False")
    print("  log file written: False")
    print("  formal training experiment: False")
    print(f"  small train/valid smoke ok: {status_ok}")
    return 0 if status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
