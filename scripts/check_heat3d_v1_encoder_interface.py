import argparse
from pathlib import Path
import sys
from typing import Any

import jax


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.dataset_Heat3D_v1 import Heat3DV1MetadataDataset
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v1_schema import default_v1_samples_dir
from rigno.models.rigno import RIGNO as GraphNeuralOperator


CANONICAL_K_MODE = "diag3"
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
        description="Model-facing encoder-interface / forward smoke for Heat3D v1 in canonical diag3 mode."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_samples_dir(REPO_DIR),
        help="Sample directory, samples/ directory, or subset directory.",
    )
    return parser.parse_args()


def _shape_of(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, dict):
        return {str(key): _shape_of(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shape_of(item) for item in value]
    return type(value).__name__


def _summarize_inputs(inputs: Any) -> dict[str, Any]:
    return {
        "u": _shape_of(inputs.u),
        "c": _shape_of(inputs.c),
        "x_inp": _shape_of(inputs.x_inp),
        "x_out": _shape_of(inputs.x_out),
        "t": _shape_of(inputs.t),
        "tau": _shape_of(inputs.tau),
    }


def main() -> int:
    args = parse_args()
    dataset = Heat3DV1MetadataDataset(args.path, k_encoding_mode=CANONICAL_K_MODE)
    builder = Heat3DGraphBuilder()

    sample_index_by_id = {sample["sample_id"]: index for index, sample in enumerate(dataset.samples)}
    missing = [sample_id for sample_id in TARGET_SAMPLE_IDS if sample_id not in sample_index_by_id]
    if missing:
        raise ValueError(f"Required smoke samples are missing: {missing}")

    model = GraphNeuralOperator(**MODEL_CONFIG)
    params = None
    forward_ok = {}

    print(f"canonical k_encoding_mode: {CANONICAL_K_MODE}")
    print(f"target samples: {TARGET_SAMPLE_IDS}")
    print(f"model_config: {MODEL_CONFIG}")

    for sample_id in TARGET_SAMPLE_IDS:
        index = sample_index_by_id[sample_id]
        sample = dataset.samples[index]
        model_input = dataset.get_model_input(index)
        operator_inputs = dataset.get_operator_interface_inputs(index)

        metadata_ok = True
        graphs_ok = True
        forward_stage = "not_started"
        metadata = None
        graphs = None
        output_shape = None
        error_message = None

        try:
            metadata = builder.build_metadata(sample["coords"])
            forward_stage = "graph_metadata_built"
        except Exception as exc:
            metadata_ok = False
            graphs_ok = False
            error_message = f"graph metadata build failed: {type(exc).__name__}: {exc}"

        if metadata_ok:
            try:
                graphs = builder.build_graphs(metadata)
                forward_stage = "graphs_built"
            except Exception as exc:
                graphs_ok = False
                error_message = f"graph build failed: {type(exc).__name__}: {exc}"

        if metadata_ok and graphs_ok:
            try:
                if params is None:
                    init_key = jax.random.PRNGKey(0)
                    params = model.init(init_key, inputs=operator_inputs.inputs, graphs=graphs)["params"]
                    forward_stage = "model_initialized"

                output = model.apply({"params": params}, inputs=operator_inputs.inputs, graphs=graphs)
                output_shape = tuple(output.shape)
                forward_stage = "forward_ok"
                forward_ok[sample_id] = True
            except Exception as exc:
                error_message = f"forward failed at stage {forward_stage}: {type(exc).__name__}: {exc}"
                forward_ok[sample_id] = False
        else:
            forward_ok[sample_id] = False

        print(f"\n{sample_id}")
        print(f"  raw k_field shape: {sample['k_field'].shape}")
        print(f"  diag3 encoded features shape: {model_input['features'].shape}")
        print(f"  feature_names: {model_input['feature_names']}")
        print(f"  graph metadata build: {metadata_ok}")
        print(f"  graphs build: {graphs_ok}")
        print(f"  model-facing input summary: {_summarize_inputs(operator_inputs.inputs)}")
        print(f"  u_feature_names: {operator_inputs.u_feature_names}")
        print(f"  c_feature_names: {operator_inputs.c_feature_names}")
        print(f"  adapter_note: {operator_inputs.adapter_note}")
        print(f"  forward stage: {forward_stage}")
        print(f"  forward output shape: {output_shape}")
        print(f"  forward ok: {forward_ok[sample_id]}")
        if error_message is not None:
            print(f"  error: {error_message}")

    print("\nsummary")
    print(f"  sample_000 passed: {forward_ok.get('sample_000', False)}")
    print(f"  sample_005 passed: {forward_ok.get('sample_005', False)}")
    print(
        "  same canonical contract: "
        f"{dataset.get_model_input(sample_index_by_id['sample_000'])['feature_names'] == dataset.get_model_input(sample_index_by_id['sample_005'])['feature_names']}"
    )

    return 0 if all(forward_ok.get(sample_id, False) for sample_id in TARGET_SAMPLE_IDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
