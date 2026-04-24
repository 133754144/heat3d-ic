import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset
from rigno.heat3d_v1_supervised import default_v1_supervised_samples_dir
from rigno.models.rigno import RIGNO as GraphNeuralOperator


TARGET_SAMPLE_IDS = ("sample_000", "sample_005")
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
        description="Smoke-test the v1 temperature-rise legacy bridge without training."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_supervised_samples_dir(REPO_DIR),
        help="Supervised smoke samples directory.",
    )
    return parser.parse_args()


def _shape_of(value):
    if value is None:
        return None
    if hasattr(value, "shape"):
        return tuple(value.shape)
    return type(value).__name__


def main() -> int:
    args = parse_args()
    dataset = Heat3DV1NativeSupervisedDataset(args.path, k_encoding_mode="diag3")
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in TARGET_SAMPLE_IDS if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"Required supervised smoke samples are missing: {missing}")

    builder = Heat3DGraphBuilder()
    model = GraphNeuralOperator(**MODEL_CONFIG)
    params = None
    status = {}

    print("v1 temperature-rise bridge smoke")
    print("  native task: coords + condition_features -> target_temperature")
    print("  bridge policy: legacy_inputs.u = non-leaking T_ref")
    print("  bridge policy: legacy_inputs.c = condition_features")
    print("  supervised bridge target: target_delta_u = target_temperature - T_ref")
    print("  this is not training, not optimization, and not coarse-to-fine")

    for sample_id in TARGET_SAMPLE_IDS:
      example = dataset[index_by_id[sample_id]]
      bridge = example.build_temperature_rise_legacy_inputs()
      target_temperature = jnp.asarray(
          example.target.target_u.reshape(1, 1, example.target.target_u.shape[0], 1)
      )
      reconstructed = bridge.t_ref + bridge.target_delta_u

      target_not_in_input = (
          "temperature" not in example.condition.condition_feature_names
          and "target_temperature" not in example.condition.condition_feature_names
          and "target_u" not in example.condition.condition_feature_names
      )
      kx_not_legacy_u = not bool(
          np.allclose(
              np.asarray(bridge.legacy_inputs.u).reshape(-1),
              example.condition.condition_features[:, 0],
          )
      )
      reconstruction_ok = bool(jnp.allclose(reconstructed, target_temperature))
      condition_shape_ok = bridge.legacy_inputs.c is not None and (
          bridge.legacy_inputs.c.shape[-1] == len(example.condition.condition_feature_names)
      )

      metadata = builder.build_metadata(example.condition.coords)
      graphs = builder.build_graphs(metadata)
      forward_ok = False
      loss_input_ok = False
      output_shape = None
      mse_smoke = None
      error_message = None

      try:
          if params is None:
              params = model.init(
                  jax.random.PRNGKey(0),
                  inputs=bridge.legacy_inputs,
                  graphs=graphs,
              )["params"]
          output = model.apply({"params": params}, inputs=bridge.legacy_inputs, graphs=graphs)
          output_shape = tuple(output.shape)
          forward_ok = True
          mse_smoke = float(jnp.mean(jnp.square(output - bridge.target_delta_u)))
          loss_input_ok = True
      except Exception as exc:
          error_message = f"{type(exc).__name__}: {exc}"

      sample_ok = (
          target_not_in_input
          and kx_not_legacy_u
          and reconstruction_ok
          and condition_shape_ok
          and forward_ok
          and loss_input_ok
      )
      status[sample_id] = sample_ok

      delta = bridge.target_delta_u
      print(f"\n{sample_id}")
      print(f"  condition_features shape: {example.condition.condition_features.shape}")
      print(f"  condition_feature_names: {example.condition.condition_feature_names}")
      print(f"  target_temperature shape: {target_temperature.shape}")
      print(f"  T_ref value: {bridge.t_ref_value}")
      print(f"  T_ref source: {bridge.t_ref_source}")
      print(f"  T_ref role: non-leaking metadata-derived baseline, not ground truth")
      print(f"  target_delta_u shape: {delta.shape}")
      print(
          "  target_delta_u min/max/mean: "
          f"{float(jnp.min(delta)):.6f}, {float(jnp.max(delta)):.6f}, {float(jnp.mean(delta)):.6f}"
      )
      print(f"  legacy_inputs.u shape: {_shape_of(bridge.legacy_inputs.u)}")
      print(f"  legacy_inputs.c shape: {_shape_of(bridge.legacy_inputs.c)}")
      print(f"  target_temperature in inputs: {not target_not_in_input}")
      print(f"  k_x used as canonical legacy u: {not kx_not_legacy_u}")
      print(f"  target_temperature == T_ref + target_delta_u: {reconstruction_ok}")
      print(f"  forward ok: {forward_ok}")
      print(f"  output shape: {output_shape}")
      print(f"  loss-input smoke ok: {loss_input_ok}")
      print(f"  delta MSE smoke: {mse_smoke}")
      if error_message:
          print(f"  error: {error_message}")
      print(f"  sample bridge ok: {sample_ok}")

    all_ok = all(status.get(sample_id, False) for sample_id in TARGET_SAMPLE_IDS)
    print("\nsummary")
    print(f"  sample_000 bridge smoke: {status.get('sample_000', False)}")
    print(f"  sample_005 bridge smoke: {status.get('sample_005', False)}")
    print(f"  temperature-rise bridge smoke ok: {all_ok}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
