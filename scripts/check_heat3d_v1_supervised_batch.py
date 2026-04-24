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
        description="Tiny supervised batch smoke for v1 steady temperature prediction."
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


def _edge_count(typed_graph):
    edge_key = list(typed_graph.edges.keys())[0]
    return typed_graph.edges[edge_key].n_edge


def _build_batch_metadata(builder: Heat3DGraphBuilder, coords_list: list[np.ndarray]):
    metadata_list = [builder.build_metadata(coords) for coords in coords_list]
    same_coords = all(np.array_equal(coords_list[0], coords) for coords in coords_list[1:])
    if same_coords:
        return tree.tree_map(
            lambda value: jnp.repeat(value, repeats=len(coords_list), axis=0),
            metadata_list[0],
        ), True
    return tree.tree_map(lambda *values: jnp.concatenate(values, axis=0), *metadata_list), False


def main() -> int:
    args = parse_args()
    dataset = Heat3DV1SupervisedDataset(args.path, k_encoding_mode="diag3")
    builder = Heat3DGraphBuilder()
    model = GraphNeuralOperator(**MODEL_CONFIG)

    sample_index_by_id = {sample["sample_id"]: idx for idx, sample in enumerate(dataset.samples)}
    missing = [sample_id for sample_id in TARGET_SAMPLE_IDS if sample_id not in sample_index_by_id]
    if missing:
        raise ValueError(f"Required supervised smoke samples are missing: {missing}")

    indices = [sample_index_by_id[sample_id] for sample_id in TARGET_SAMPLE_IDS]
    samples = [dataset.samples[idx] for idx in indices]
    examples = [dataset.get_supervised_example(idx) for idx in indices]

    feature_contract_ok = all(
        example.full_feature_names == examples[0].full_feature_names
        for example in examples[1:]
    )
    if not feature_contract_ok:
        raise ValueError("Feature-name contract mismatch across supervised smoke samples")

    batched_inputs = Inputs(
        u=jnp.concatenate([example.inputs.u for example in examples], axis=0),
        c=jnp.concatenate([example.inputs.c for example in examples], axis=0)
        if examples[0].inputs.c is not None
        else None,
        x_inp=jnp.concatenate([example.inputs.x_inp for example in examples], axis=0),
        x_out=jnp.concatenate([example.inputs.x_out for example in examples], axis=0),
        t=None,
        tau=None,
    )
    batched_target = jnp.concatenate([example.target_temperature for example in examples], axis=0)

    batch_metadata, uses_shared_metadata = _build_batch_metadata(
        builder=builder,
        coords_list=[sample["coords"] for sample in samples],
    )
    graphs = builder.build_graphs(batch_metadata)

    params = model.init(jax.random.PRNGKey(0), inputs=batched_inputs, graphs=graphs)["params"]
    output = model.apply({"params": params}, inputs=batched_inputs, graphs=graphs)
    mse = float(jnp.mean(jnp.square(output - batched_target)))

    print("samples")
    for sample, example in zip(samples, examples):
        print(
            f"  {sample['sample_id']}: split={sample['meta']['split']}, "
            f"raw_k={sample['k_field'].shape}, features={sample['physics_input'].features.shape}, "
            f"target={tuple(example.target_temperature.shape)}"
        )

    print("\nbatch")
    print(f"  sample_ids: {TARGET_SAMPLE_IDS}")
    print(f"  pure_physics default: {dataset.input_mode == 'pure_physics'}")
    print(f"  k_encoding_mode: {dataset.k_encoding_mode}")
    print(f"  feature_names: {examples[0].full_feature_names}")
    print(f"  feature contract ok: {feature_contract_ok}")
    print(f"  graph metadata shared repeat: {uses_shared_metadata}")
    print(f"  batched u shape: {_shape_of(batched_inputs.u)}")
    print(f"  batched c shape: {_shape_of(batched_inputs.c)}")
    print(f"  batched x_inp shape: {_shape_of(batched_inputs.x_inp)}")
    print(f"  batched x_out shape: {_shape_of(batched_inputs.x_out)}")
    print(f"  batched target shape: {tuple(batched_target.shape)}")
    print(f"  batch metadata x_pnodes_inp: {tuple(batch_metadata.x_pnodes_inp.shape)}")
    print(f"  batch metadata x_rnodes: {tuple(batch_metadata.x_rnodes.shape)}")
    print(f"  p2r n_edge shape: {tuple(_edge_count(graphs.p2r).shape)}")
    print(f"  r2r n_edge shape: {tuple(_edge_count(graphs.r2r).shape)}")
    print(f"  r2p n_edge shape: {tuple(_edge_count(graphs.r2p).shape)}")
    print(f"  forward ok: True")
    print(f"  batched output shape: {tuple(output.shape)}")
    print(f"  batch loss-input smoke ok: True")
    print(f"  mse smoke: {mse}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
