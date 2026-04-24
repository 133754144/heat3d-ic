import argparse
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
from rigno.heat3d_v1_supervised import Heat3DV1SupervisedDataset, default_v1_supervised_samples_dir
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
        description="Very tiny v1 supervised training smoke. This is not a real experiment."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_supervised_samples_dir(REPO_DIR),
        help="Supervised smoke samples directory.",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _select_examples(dataset: Heat3DV1SupervisedDataset):
    index_by_id = {sample["sample_id"]: idx for idx, sample in enumerate(dataset.samples)}
    missing = [sample_id for sample_id in TARGET_SAMPLE_IDS if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"Required supervised smoke samples are missing: {missing}")
    indices = [index_by_id[sample_id] for sample_id in TARGET_SAMPLE_IDS]
    return [dataset.samples[idx] for idx in indices], [dataset.get_supervised_example(idx) for idx in indices]


def _build_batch_metadata(builder: Heat3DGraphBuilder, coords_list: list[np.ndarray]):
    metadata_list = [builder.build_metadata(coords) for coords in coords_list]
    same_coords = all(np.array_equal(coords_list[0], coords) for coords in coords_list[1:])
    if same_coords:
        return tree.tree_map(
            lambda value: jnp.repeat(value, repeats=len(coords_list), axis=0),
            metadata_list[0],
        ), True
    return tree.tree_map(lambda *values: jnp.concatenate(values, axis=0), *metadata_list), False


def _combine_features(examples):
    feature_names = examples[0].full_feature_names
    for example in examples[1:]:
        if example.full_feature_names != feature_names:
            raise ValueError("Feature-name contract mismatch across supervised smoke samples")

    features = []
    for example in examples:
        if example.inputs.c is None:
            feature = example.inputs.u
        else:
            feature = jnp.concatenate([example.inputs.u, example.inputs.c], axis=-1)
        features.append(feature)
    return jnp.concatenate(features, axis=0), feature_names


def _make_normalized_inputs(examples) -> tuple[Inputs, jnp.ndarray, dict]:
    raw_features, feature_names = _combine_features(examples)
    raw_target = jnp.concatenate([example.target_temperature for example in examples], axis=0)
    raw_coords = jnp.concatenate([example.inputs.x_inp for example in examples], axis=0)

    feature_mean = jnp.mean(raw_features, axis=(0, 1, 2), keepdims=True)
    feature_std = jnp.std(raw_features, axis=(0, 1, 2), keepdims=True)
    feature_safe_std = jnp.where(feature_std < EPS, 1.0, feature_std)

    target_mean = jnp.mean(raw_target, axis=(0, 1, 2), keepdims=True)
    target_std = jnp.std(raw_target, axis=(0, 1, 2), keepdims=True)
    target_safe_std = jnp.where(target_std < EPS, 1.0, target_std)

    coord_min = jnp.min(raw_coords, axis=(0, 1, 2), keepdims=True)
    coord_max = jnp.max(raw_coords, axis=(0, 1, 2), keepdims=True)
    coord_span = jnp.where((coord_max - coord_min) < EPS, 1.0, coord_max - coord_min)

    normalized_features = (raw_features - feature_mean) / feature_safe_std
    normalized_target = (raw_target - target_mean) / target_safe_std
    normalized_coords = 2.0 * ((raw_coords - coord_min) / coord_span) - 1.0

    inputs = Inputs(
        u=normalized_features[..., :1],
        c=normalized_features[..., 1:],
        x_inp=normalized_coords,
        x_out=normalized_coords,
        t=None,
        tau=None,
    )
    stats = {
        "feature_names": feature_names,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_safe_std": feature_safe_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_safe_std": target_safe_std,
        "raw_target": raw_target,
    }
    return inputs, normalized_target, stats


def _global_norm(tree_value) -> float:
    leaves = tree.tree_leaves(tree_value)
    total = sum(float(jnp.sum(jnp.square(leaf))) for leaf in leaves)
    return float(np.sqrt(total))


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    dataset = Heat3DV1SupervisedDataset(args.path, k_encoding_mode="diag3")
    samples, examples = _select_examples(dataset)
    inputs, target_normalized, stats = _make_normalized_inputs(examples)

    builder = Heat3DGraphBuilder()
    batch_metadata, shared_metadata = _build_batch_metadata(
        builder=builder,
        coords_list=[sample["coords"] for sample in samples],
    )
    graphs = builder.build_graphs(batch_metadata)

    model = GraphNeuralOperator(**MODEL_CONFIG)
    params = model.init(jax.random.PRNGKey(args.seed), inputs=inputs, graphs=graphs)["params"]

    def loss_fn(current_params):
        pred_normalized = model.apply({"params": current_params}, inputs=inputs, graphs=graphs)
        return jnp.mean(jnp.square(pred_normalized - target_normalized))

    initial_loss = float(loss_fn(params))
    losses = [initial_loss]
    grad_norms = []

    for _ in range(args.steps):
        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = _global_norm(grads)
        params = tree.tree_map(lambda param, grad: param - args.lr * grad, params, grads)
        losses.append(float(loss_fn(params)))
        grad_norms.append(grad_norm)

    pred_normalized = model.apply({"params": params}, inputs=inputs, graphs=graphs)
    pred_raw = pred_normalized * stats["target_safe_std"] + stats["target_mean"]
    raw_mse = float(jnp.mean(jnp.square(pred_raw - stats["raw_target"])))

    finite_ok = bool(
        jnp.all(jnp.isfinite(jnp.asarray(losses)))
        and jnp.all(jnp.isfinite(jnp.asarray(grad_norms)))
        and jnp.all(jnp.isfinite(pred_normalized))
    )
    shape_ok = pred_normalized.shape == target_normalized.shape
    loss_stable = finite_ok and max(losses) < 1e6
    status_ok = finite_ok and shape_ok and loss_stable

    print("tiny supervised training smoke")
    print("  purpose: optimizer/backward/loss-contract smoke only, not a real experiment")
    print(f"  sample_ids: {TARGET_SAMPLE_IDS}")
    print(f"  k_encoding_mode: {dataset.k_encoding_mode}")
    print(f"  feature_names: {stats['feature_names']}")
    print("  input semantics: coords + encoded_k_field + q_field + BC encoding")
    print("  target semantics: temperature.npy as normalized supervised target")
    print("  loss contract: MSE(pred_normalized_temperature, normalized_temperature)")
    print("  adapter note: u/c split is compatibility-only")

    print("\nbatch contract")
    print(f"  u shape: {tuple(inputs.u.shape)}")
    print(f"  c shape: {tuple(inputs.c.shape)}")
    print(f"  x shape: {tuple(inputs.x_inp.shape)}")
    print(f"  target shape: {tuple(target_normalized.shape)}")
    print(f"  graph metadata shared repeat: {shared_metadata}")
    print(f"  target mean/std: {float(stats['target_mean'].reshape(-1)[0]):.6f}, {float(stats['target_std'].reshape(-1)[0]):.6f}")

    print("\ntraining smoke")
    print(f"  model_config: {MODEL_CONFIG}")
    print(f"  steps: {args.steps}")
    print(f"  lr: {args.lr}")
    print(f"  normalized losses: {[float(loss) for loss in losses]}")
    print(f"  grad norms: {[float(norm) for norm in grad_norms]}")
    print(f"  raw-temperature MSE after smoke steps: {raw_mse}")
    print(f"  finite ok: {finite_ok}")
    print(f"  shape ok: {shape_ok}")
    print(f"  loss stable: {loss_stable}")
    print(f"  tiny training smoke ok: {status_ok}")

    return 0 if status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
