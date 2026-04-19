from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
  sys.path.insert(0, str(REPO_DIR))

# -----------------------------------------------------------------------------
# Backend warning switch
# -----------------------------------------------------------------------------
# JAX/XLA may print many low-level GPU timing messages such as:
#   "cuda_timer.cc:87] Delay kernel timed out ..."
# These messages are usually about profiling / timer accuracy and do NOT mean
# that training is wrong. They make the terminal very noisy, so we hide them by
# default for this project.
#
# If you want to see the original backend warnings again, use either method:
# 1. Edit the line below and set SHOW_XLA_WARNINGS = True
# 2. Or run from shell with:
#      HEAT3D_SHOW_XLA_WARNINGS=1 python3 scripts/train_heat3d_operator.py
#
# Important: this switch must be handled before importing jax/flax/optax,
# otherwise the backend logger may already be initialized.
# -----------------------------------------------------------------------------
SHOW_XLA_WARNINGS = False

if os.environ.get("HEAT3D_SHOW_XLA_WARNINGS", "").lower() in {"1", "true", "yes", "on"}:
  SHOW_XLA_WARNINGS = True

if not SHOW_XLA_WARNINGS:
  # `TF_CPP_MIN_LOG_LEVEL=3` suppresses TensorFlow/XLA C++ backend logs,
  # including the noisy cuda_timer warnings seen during GPU execution.
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rigno.dataset_Heat3D import Heat3DDataset
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_pipeline import (
  HeatSteadyOutputStepper,
  compute_heat3d_stats,
  get_batch_inputs,
  iterate_batch_indices,
  median_relative_l1_error,
  prediction_metrics,
  save_checkpoint,
  save_json,
  split_dataset_indices,
)
from rigno.models.operator import Inputs
from rigno.models.rigno import RIGNO as GraphNeuralOperator


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Train a steady 3D heat graph neural operator and save the best checkpoint.",
  )
  parser.add_argument("--data-dir", "--datadir", dest="data_dir", type=Path, default=(REPO_DIR / "dataset_3d_heat"))
  parser.add_argument("--output-dir", type=Path, default=(REPO_DIR / "output" / "heat3d_ic"))
  parser.add_argument("--checkpoint-name", type=str, default="heat3d_operator_best.pkl")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=1e-6)
  parser.add_argument("--n-train", type=int, default=160)
  parser.add_argument("--n-valid", type=int, default=20)
  parser.add_argument("--n-test", type=int, default=20)
  parser.add_argument("--processor-steps", type=int, default=8)
  parser.add_argument("--node-latent-size", type=int, default=64)
  parser.add_argument("--edge-latent-size", type=int, default=64)
  parser.add_argument("--mlp-hidden-layers", type=int, default=2)
  parser.add_argument("--p-edge-masking", type=float, default=0.0)
  parser.add_argument("--rmesh-levels", type=int, default=3)
  parser.add_argument("--subsample-factor", type=float, default=4.0)
  parser.add_argument("--overlap-factor-p2r", type=float, default=1.5)
  parser.add_argument("--overlap-factor-r2p", type=float, default=2.0)
  parser.add_argument("--node-coordinate-freqs", type=int, default=4)
  parser.add_argument("--remesh-every-epoch", action="store_true")
  return parser.parse_args()


def count_parameters(params) -> int:
  leaves = jax.tree_util.tree_leaves(params)
  return int(sum(np.prod(leaf.shape) for leaf in leaves))


def make_model_config(args: argparse.Namespace, num_outputs: int) -> dict:
  return {
    "num_outputs": num_outputs,
    "processor_steps": args.processor_steps,
    "node_latent_size": args.node_latent_size,
    "edge_latent_size": args.edge_latent_size,
    "mlp_hidden_layers": args.mlp_hidden_layers,
    "concatenate_tau": False,
    "concatenate_t": False,
    "conditioned_normalization": False,
    "cond_norm_hidden_size": 16,
    "p_edge_masking": args.p_edge_masking,
  }


def main() -> None:
  args = parse_args()

  dataset = Heat3DDataset(str(args.data_dir))
  splits = split_dataset_indices(
    num_samples=len(dataset),
    n_train=args.n_train,
    n_valid=args.n_valid,
    n_test=args.n_test,
    seed=args.seed,
  )

  builder = Heat3DGraphBuilder(
    rmesh_levels=args.rmesh_levels,
    subsample_factor=args.subsample_factor,
    overlap_factor_p2r=args.overlap_factor_p2r,
    overlap_factor_r2p=args.overlap_factor_r2p,
    node_coordinate_freqs=args.node_coordinate_freqs,
  )
  dataset.build_graph_metadata(builder)

  stats = compute_heat3d_stats(dataset, splits["train"])
  model_config = make_model_config(args, num_outputs=dataset.samples[0]["u"].shape[-1])
  model = GraphNeuralOperator(**model_config)
  stepper = HeatSteadyOutputStepper(model)

  dummy_indices = [int(splits["train"][0])] * args.batch_size
  _, dummy_u_inp, dummy_c_inp, dummy_x, dummy_g = get_batch_inputs(dataset, dummy_indices)
  dummy_graphs = builder.build_graphs(dummy_g)
  dummy_inputs = Inputs(
    u=dummy_u_inp,
    c=dummy_c_inp,
    x_inp=dummy_x,
    x_out=dummy_x,
    t=None,
    tau=None,
  )

  key = jax.random.PRNGKey(args.seed)
  key, init_key = jax.random.split(key)
  params = model.init(init_key, inputs=dummy_inputs, graphs=dummy_graphs)["params"]
  tx = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

  print(f"Model parameters: {count_parameters(state.params):,}")
  print(f"Train/valid/test split: {len(splits['train'])}/{len(splits['valid'])}/{len(splits['test'])}")

  @jax.jit
  def train_step(state, u_tgt, u_inp, c_inp, x, graphs, rng):
    inputs = Inputs(
      u=u_inp,
      c=c_inp,
      x_inp=x,
      x_out=x,
      t=None,
      tau=None,
    )

    def loss_fn(params):
      u_tgt_nrm, u_prd_nrm = stepper.get_loss_inputs(
        variables={"params": params},
        stats=stats,
        u_tgt=u_tgt,
        inputs=inputs,
        graphs=graphs,
        key=rng,
      )
      return jnp.mean(jnp.square(u_prd_nrm - u_tgt_nrm))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

  @jax.jit
  def predict_step(params, u_inp, c_inp, x, graphs):
    inputs = Inputs(
      u=u_inp,
      c=c_inp,
      x_inp=x,
      x_out=x,
      t=None,
      tau=None,
    )
    return stepper.apply(
      variables={"params": params},
      stats=stats,
      inputs=inputs,
      graphs=graphs,
      key=None,
    )

  def evaluate(indices) -> dict[str, float]:
    u_tgt_all = []
    u_prd_all = []
    for batch_indices in iterate_batch_indices(indices, batch_size=args.batch_size, shuffle=False):
      u_tgt, u_inp, c_inp, x, g = get_batch_inputs(dataset, batch_indices)
      graphs = builder.build_graphs(g)
      u_prd = predict_step(state.params, u_inp, c_inp, x, graphs)
      u_tgt_all.append(u_tgt)
      u_prd_all.append(u_prd)

    u_tgt_all = jnp.concatenate(u_tgt_all, axis=0)
    u_prd_all = jnp.concatenate(u_prd_all, axis=0)
    metrics = prediction_metrics(u_tgt_all, u_prd_all)
    metrics["median_relative_l1"] = median_relative_l1_error(u_tgt_all, u_prd_all)
    return metrics

  best_valid_mse = float("inf")
  best_payload = None
  history = []

  for epoch in range(1, args.epochs + 1):
    if args.remesh_every_epoch:
      key, remesh_key = jax.random.split(key)
      dataset.build_graph_metadata(builder, key=remesh_key)

    train_losses = []
    for batch_indices in iterate_batch_indices(
      splits["train"],
      batch_size=args.batch_size,
      shuffle=True,
      seed=args.seed + epoch,
    ):
      u_tgt, u_inp, c_inp, x, g = get_batch_inputs(dataset, batch_indices)
      graphs = builder.build_graphs(g)
      key, batch_key = jax.random.split(key)
      state, loss = train_step(state, u_tgt, u_inp, c_inp, x, graphs, batch_key)
      train_losses.append(float(loss))

    valid_metrics = evaluate(splits["valid"])
    record = {
      "epoch": epoch,
      "train_loss": float(np.mean(train_losses)),
      "valid_mse": valid_metrics["mse"],
      "valid_rmse": valid_metrics["rmse"],
      "valid_mae": valid_metrics["mae"],
      "valid_rel_l1": valid_metrics["rel_l1"],
      "valid_rel_l2": valid_metrics["rel_l2"],
      "valid_median_relative_l1": valid_metrics["median_relative_l1"],
      "valid_r2": valid_metrics["r2"],
    }
    history.append(record)

    print(
      f"Epoch {epoch:03d} | "
      f"train_loss={record['train_loss']:.6e} | "
      f"valid_mse={record['valid_mse']:.6e} | "
      f"valid_rmse={record['valid_rmse']:.6e} | "
      f"valid_med_rel_l1={record['valid_median_relative_l1']:.6e} | "
      f"valid_rel_l2={record['valid_rel_l2']:.6e} | "
      f"valid_r2={record['valid_r2']:.6e}"
    )

    if valid_metrics["mse"] < best_valid_mse:
      best_valid_mse = valid_metrics["mse"]
      best_payload = {
        "params": jax.device_get(state.params),
        "stats": jax.device_get(stats),
        "model_config": model_config,
        "builder_config": builder.config,
        "splits": {key_name: value.tolist() for key_name, value in splits.items()},
        "train_args": vars(args),
        "best_epoch": epoch,
        "best_valid_metrics": valid_metrics,
      }

  if best_payload is None:
    raise RuntimeError("Training did not produce a checkpoint payload.")

  checkpoint_path = args.output_dir / args.checkpoint_name
  save_checkpoint(checkpoint_path, best_payload)
  save_json(args.output_dir / "train_history.json", {"history": history})

  print("\nTraining finished.")
  print(f"Best epoch: {best_payload['best_epoch']}")
  print(f"Best validation metrics: {best_payload['best_valid_metrics']}")
  print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
  main()
