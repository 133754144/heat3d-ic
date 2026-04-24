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

from check_heat3d_v1_tiny_training import (  # noqa: E402
    MODEL_CONFIG,
    TARGET_SAMPLE_IDS,
    _build_batch_metadata,
    _global_norm,
    _make_normalized_inputs,
    _select_examples,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v1_supervised import Heat3DV1SupervisedDataset, default_v1_supervised_samples_dir
from rigno.models.rigno import RIGNO as GraphNeuralOperator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpoint-free repeatability check for the v1 tiny supervised training smoke."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_supervised_samples_dir(REPO_DIR),
        help="Supervised smoke samples directory.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-7)
    return parser.parse_args()


def _run_once(path: Path, steps: int, lr: float, seed: int) -> dict:
    dataset = Heat3DV1SupervisedDataset(path, k_encoding_mode="diag3")
    samples, examples = _select_examples(dataset)
    inputs, target_normalized, stats = _make_normalized_inputs(examples)

    builder = Heat3DGraphBuilder()
    batch_metadata, shared_metadata = _build_batch_metadata(
        builder=builder,
        coords_list=[sample["coords"] for sample in samples],
    )
    graphs = builder.build_graphs(batch_metadata)

    model = GraphNeuralOperator(**MODEL_CONFIG)
    params = model.init(jax.random.PRNGKey(seed), inputs=inputs, graphs=graphs)["params"]

    def loss_fn(current_params):
        pred_normalized = model.apply({"params": current_params}, inputs=inputs, graphs=graphs)
        return jnp.mean(jnp.square(pred_normalized - target_normalized))

    losses = [float(loss_fn(params))]
    grad_norms = []

    for _ in range(steps):
        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        grad_norms.append(_global_norm(grads))
        params = tree.tree_map(lambda param, grad: param - lr * grad, params, grads)
        losses.append(float(loss_fn(params)))

    pred_normalized = model.apply({"params": params}, inputs=inputs, graphs=graphs)
    pred_raw = pred_normalized * stats["target_safe_std"] + stats["target_mean"]
    raw_mse = float(jnp.mean(jnp.square(pred_raw - stats["raw_target"])))

    return {
        "losses": np.asarray(losses, dtype=np.float64),
        "grad_norms": np.asarray(grad_norms, dtype=np.float64),
        "raw_mse": raw_mse,
        "feature_names": stats["feature_names"],
        "input_shapes": {
            "u": tuple(inputs.u.shape),
            "c": tuple(inputs.c.shape) if inputs.c is not None else None,
            "x": tuple(inputs.x_inp.shape),
            "target": tuple(target_normalized.shape),
        },
        "target_mean": float(stats["target_mean"].reshape(-1)[0]),
        "target_std": float(stats["target_std"].reshape(-1)[0]),
        "shared_metadata": shared_metadata,
    }


def _max_abs_delta(values: list[np.ndarray | float]) -> float:
    base = values[0]
    return max(float(np.max(np.abs(value - base))) for value in values[1:]) if len(values) > 1 else 0.0


def main() -> int:
    args = parse_args()
    if args.runs < 2:
        raise ValueError("--runs must be >= 2")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    results = [_run_once(args.path, args.steps, args.lr, args.seed) for _ in range(args.runs)]

    loss_arrays = [result["losses"] for result in results]
    grad_arrays = [result["grad_norms"] for result in results]
    raw_mses = [result["raw_mse"] for result in results]

    exact_losses = all(np.array_equal(loss_arrays[0], value) for value in loss_arrays[1:])
    exact_grad_norms = all(np.array_equal(grad_arrays[0], value) for value in grad_arrays[1:])
    exact_raw_mse = all(raw_mses[0] == value for value in raw_mses[1:])

    close_losses = all(np.allclose(loss_arrays[0], value, atol=args.atol, rtol=args.rtol) for value in loss_arrays[1:])
    close_grad_norms = all(np.allclose(grad_arrays[0], value, atol=args.atol, rtol=args.rtol) for value in grad_arrays[1:])
    close_raw_mse = all(np.isclose(raw_mses[0], value, atol=args.atol, rtol=args.rtol) for value in raw_mses[1:])

    status_ok = close_losses and close_grad_norms and close_raw_mse

    print("checkpoint-free tiny training repeatability")
    print("  purpose: deterministic smoke validation only, not a training result")
    print(f"  sample_ids: {TARGET_SAMPLE_IDS}")
    print("  canonical mode: diag3")
    print("  input semantics: coords + encoded_k_field + q_field + BC encoding")
    print("  target semantics: temperature.npy as normalized supervised target")
    print("  adapter note: u/c split is compatibility-only")
    print(f"  runs: {args.runs}")
    print(f"  steps: {args.steps}")
    print(f"  lr: {args.lr}")
    print(f"  seed: {args.seed}")
    print(f"  tolerances: atol={args.atol}, rtol={args.rtol}")

    print("\ncontract")
    print(f"  feature_names: {results[0]['feature_names']}")
    print(f"  input_shapes: {results[0]['input_shapes']}")
    print(f"  graph metadata shared repeat: {results[0]['shared_metadata']}")
    print(f"  target mean/std: {results[0]['target_mean']:.6f}, {results[0]['target_std']:.6f}")

    print("\nruns")
    for index, result in enumerate(results):
        print(f"  run_{index}:")
        print(f"    losses: {[float(value) for value in result['losses']]}")
        print(f"    grad_norms: {[float(value) for value in result['grad_norms']]}")
        print(f"    raw_mse: {result['raw_mse']}")

    print("\ncomparison")
    print(f"  exact losses: {exact_losses}")
    print(f"  exact grad norms: {exact_grad_norms}")
    print(f"  exact raw mse: {exact_raw_mse}")
    print(f"  max loss delta: {_max_abs_delta(loss_arrays):.6e}")
    print(f"  max grad norm delta: {_max_abs_delta(grad_arrays):.6e}")
    print(f"  max raw mse delta: {_max_abs_delta(raw_mses):.6e}")
    print(f"  within tolerance losses: {close_losses}")
    print(f"  within tolerance grad norms: {close_grad_norms}")
    print(f"  within tolerance raw mse: {close_raw_mse}")
    print(f"  repeatability smoke ok: {status_ok}")

    return 0 if status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
