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
# They are typically profiling-related warnings rather than correctness errors.
# To keep evaluation output readable, we suppress them by default.
#
# To manually re-enable the raw backend warnings:
# 1. Change SHOW_XLA_WARNINGS below to True
# 2. Or run:
#      HEAT3D_SHOW_XLA_WARNINGS=1 python3 scripts/evaluate_heat3d_operator.py
#
# This must happen before importing jax, otherwise the logger may already be
# initialized and the suppression will be ineffective.
# -----------------------------------------------------------------------------
SHOW_XLA_WARNINGS = False

if os.environ.get("HEAT3D_SHOW_XLA_WARNINGS", "").lower() in {"1", "true", "yes", "on"}:
  SHOW_XLA_WARNINGS = True

if not SHOW_XLA_WARNINGS:
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import numpy as np

from rigno.dataset_Heat3D import Heat3DDataset
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_paths import CANONICAL_DATA_SUBDIR, resolve_heat3d_data_dir
from rigno.heat3d_pipeline import (
  HeatSteadyOutputStepper,
  get_batch_inputs,
  iterate_batch_indices,
  load_checkpoint,
  median_relative_l1_error,
  per_sample_relative_l1_error,
  prediction_metrics,
  save_json,
)
from rigno.models.operator import Inputs
from rigno.models.rigno import RIGNO as GraphNeuralOperator


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Load a trained 3D heat graph neural operator checkpoint and evaluate it on the test split.",
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=(REPO_DIR / "output" / "heat3d_ic" / "heat3d_operator_best.pkl"),
  )
  parser.add_argument(
    "--data-dir",
    "--datadir",
    dest="data_dir",
    type=Path,
    default=None,
    help=f"Directory containing sample_xxx folders. Defaults to {CANONICAL_DATA_SUBDIR}, with legacy fallback.",
  )
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--output-dir", type=Path, default=None)
  parser.add_argument("--preview-samples", type=int, default=3)
  parser.add_argument(
    "--report-sample-index",
    type=int,
    default=None,
    help="Dataset sample index to report with the author-style Relative L1 error. Defaults to the first test sample.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  payload = load_checkpoint(args.checkpoint)
  data_dir = resolve_heat3d_data_dir(args.data_dir, REPO_DIR)

  output_dir = args.output_dir if (args.output_dir is not None) else args.checkpoint.parent
  output_dir.mkdir(parents=True, exist_ok=True)

  dataset = Heat3DDataset(str(data_dir))
  builder = Heat3DGraphBuilder(**payload["builder_config"])
  dataset.build_graph_metadata(builder)

  model = GraphNeuralOperator(**payload["model_config"])
  stepper = HeatSteadyOutputStepper(model)
  stats = jax.tree_util.tree_map(
    lambda value: None if value is None else jnp.asarray(value),
    payload["stats"],
  )
  params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
  test_indices = np.asarray(payload["splits"]["test"], dtype=int)

  @jax.jit
  def predict_step(u_inp, c_inp, x, graphs):
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

  u_tgt_all = []
  u_prd_all = []
  x_all = []
  sample_index_all = []

  for batch_indices in iterate_batch_indices(test_indices, batch_size=args.batch_size, shuffle=False):
    u_tgt, u_inp, c_inp, x, g = get_batch_inputs(dataset, batch_indices)
    graphs = builder.build_graphs(g)
    u_prd = predict_step(u_inp, c_inp, x, graphs)

    u_tgt_all.append(u_tgt)
    u_prd_all.append(u_prd)
    x_all.append(x)
    sample_index_all.extend(batch_indices.tolist())

  u_tgt_all = jnp.concatenate(u_tgt_all, axis=0)
  u_prd_all = jnp.concatenate(u_prd_all, axis=0)
  x_all = jnp.concatenate(x_all, axis=0)

  metrics = prediction_metrics(u_tgt_all, u_prd_all)
  metrics["best_epoch"] = int(payload["best_epoch"])

  # ---------------------------------------------------------------------------
  # Author-style Relative L1 error from the upstream reference example.
  # ---------------------------------------------------------------------------
  # In the original example, the authors compute:
  #   rel_lp_error_mean(gtr, prd, p=1)
  # and then take the median over test samples.
  #
  # For our 3D steady heat task there is only one time step and one variable,
  # so this becomes the relative L1 error of the temperature field over all
  # spatial nodes for each sample.
  # ---------------------------------------------------------------------------
  per_sample_relative_l1 = per_sample_relative_l1_error(u_tgt_all, u_prd_all)
  median_relative_l1 = median_relative_l1_error(u_tgt_all, u_prd_all)
  if args.report_sample_index is None:
    report_sample_index = int(sample_index_all[0])
  else:
    report_sample_index = int(args.report_sample_index)

  if report_sample_index not in sample_index_all:
    raise ValueError(
      f"report sample index {report_sample_index} is not in the stored test split: {sample_index_all}"
    )

  report_local_idx = sample_index_all.index(report_sample_index)
  report_sample_relative_l1 = float(per_sample_relative_l1[report_local_idx])
  metrics["median_relative_l1"] = median_relative_l1
  metrics["report_sample_index"] = report_sample_index
  metrics["report_sample_relative_l1"] = report_sample_relative_l1

  preview_count = min(args.preview_samples, len(sample_index_all))
  if preview_count > 0:
    preview_payload = {
      "sample_indices": np.asarray(sample_index_all[:preview_count]),
      "x": np.asarray(x_all[:preview_count]),
      "u_target": np.asarray(u_tgt_all[:preview_count]),
      "u_prediction": np.asarray(u_prd_all[:preview_count]),
    }
    np.savez(output_dir / "test_preview_predictions.npz", **preview_payload)

  save_json(output_dir / "test_metrics.json", metrics)

  print("Evaluation finished.")
  print(f"Checkpoint: {args.checkpoint}")
  print(f"Best epoch: {payload['best_epoch']}")
  print(f"Median relative L1 test error over all test samples: {median_relative_l1 * 100:.2f}%")
  print(
    f"Relative L1 test error of SAMPLE #{report_sample_index:03d}: "
    f"{report_sample_relative_l1 * 100:.2f}%"
  )
  print(
    "Test metrics | "
    f"mse={metrics['mse']:.6e} | "
    f"rel_l1={metrics['rel_l1']:.6e} | "
    f"rel_l2={metrics['rel_l2']:.6e}"
  )
  if preview_count > 0:
    print(f"Preview predictions saved to: {output_dir / 'test_preview_predictions.npz'}")
  print(f"Metrics saved to: {output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
  main()
