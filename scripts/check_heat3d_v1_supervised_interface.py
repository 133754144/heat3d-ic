import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v1_supervised import Heat3DV1SupervisedDataset, default_v1_supervised_samples_dir
from rigno.models.rigno import RIGNO as GraphNeuralOperator


TARGET_SAMPLE_IDS = ("sample_000", "sample_005")
MODEL_CONFIG = {
    "num_outputs": 1,
    "processor_steps": 8,
    "node_latent_size": 64,
    "edge_latent_size": 64,
    "mlp_hidden_layers": 2,
    "concatenate_tau": False,
    "concatenate_t": False,
    "conditioned_normalization": False,
    "cond_norm_hidden_size": 16,
    "p_edge_masking": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Very small supervised interface smoke for steady v1 samples."
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
    dataset = Heat3DV1SupervisedDataset(args.path, k_encoding_mode="diag3")
    builder = Heat3DGraphBuilder()
    model = GraphNeuralOperator(**MODEL_CONFIG)
    params = None
    status = {}

    sample_index_by_id = {sample["sample_id"]: idx for idx, sample in enumerate(dataset.samples)}
    missing = [sample_id for sample_id in TARGET_SAMPLE_IDS if sample_id not in sample_index_by_id]
    if missing:
        raise ValueError(f"Required supervised smoke samples are missing: {missing}")

    for sample_id in TARGET_SAMPLE_IDS:
        idx = sample_index_by_id[sample_id]
        sample = dataset.samples[idx]
        supervised = dataset.get_supervised_example(idx)

        metadata = builder.build_metadata(sample["coords"])
        graphs = builder.build_graphs(metadata)

        forward_ok = False
        loss_ok = False
        output_shape = None
        loss_value = None
        error_message = None

        try:
            if params is None:
                init_key = jax.random.PRNGKey(0)
                params = model.init(init_key, inputs=supervised.inputs, graphs=graphs)["params"]
            output = model.apply({"params": params}, inputs=supervised.inputs, graphs=graphs)
            output_shape = tuple(output.shape)
            forward_ok = True
            loss_value = float(jnp.mean(jnp.square(output - supervised.target_temperature)))
            loss_ok = True
            status[sample_id] = True
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            status[sample_id] = False

        print(f"\n{sample_id}")
        print(f"  raw k_field shape: {sample['k_field'].shape}")
        print(f"  canonical features shape: {sample['physics_input'].features.shape}")
        print(f"  target temperature shape: {supervised.target_temperature.shape}")
        print(f"  graph metadata build: True")
        print(f"  graphs build: True")
        print(
            "  model-facing input summary: "
            f"u={_shape_of(supervised.inputs.u)}, "
            f"c={_shape_of(supervised.inputs.c)}, "
            f"x_inp={_shape_of(supervised.inputs.x_inp)}"
        )
        print(f"  feature_names: {supervised.full_feature_names}")
        print(f"  target_role: {supervised.target_role}")
        print(f"  forward ok: {forward_ok}")
        print(f"  output shape: {output_shape}")
        print(f"  loss-input smoke ok: {loss_ok}")
        print(f"  mse smoke: {loss_value}")
        if error_message is not None:
            print(f"  error: {error_message}")

    print("\nsummary")
    print(f"  sample_000 supervised smoke: {status.get('sample_000', False)}")
    print(f"  sample_005 supervised smoke: {status.get('sample_005', False)}")
    print(
        "  same canonical condition contract: "
        f"{dataset.get_supervised_example(sample_index_by_id['sample_000']).full_feature_names == dataset.get_supervised_example(sample_index_by_id['sample_005']).full_feature_names}"
    )
    return 0 if all(status.get(sample_id, False) for sample_id in TARGET_SAMPLE_IDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
