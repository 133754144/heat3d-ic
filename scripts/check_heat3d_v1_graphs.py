import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.dataset_Heat3D_v1 import Heat3DV1MetadataDataset
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v1_schema import default_v1_samples_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-sample graph-construction smoke check for Heat3D v1 metadata-first samples."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_samples_dir(REPO_DIR),
        help="Sample directory, samples/ directory, or subset directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "native", "diag3"),
        default="both",
        help="Which k encoding mode(s) to test.",
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


def _summarize_metadata(metadata: Any) -> dict[str, Any]:
    return {field: _shape_of(getattr(metadata, field)) for field in metadata._fields}


def _summarize_typed_graph(graph: Any) -> dict[str, Any]:
    node_summary = {}
    for node_name, node_set in graph.nodes.items():
        node_summary[node_name] = {
            "n_node": _shape_of(node_set.n_node),
            "features": _shape_of(node_set.features),
        }

    edge_summary = {}
    for edge_key, edge_set in graph.edges.items():
        edge_summary[edge_key.name] = {
            "n_edge": _shape_of(edge_set.n_edge),
            "senders": _shape_of(edge_set.indices.senders),
            "receivers": _shape_of(edge_set.indices.receivers),
            "features": _shape_of(edge_set.features),
            "node_sets": edge_key.node_sets,
        }

    return {
        "context": {
            "n_graph": _shape_of(graph.context.n_graph),
            "features": _shape_of(graph.context.features),
        },
        "nodes": node_summary,
        "edges": edge_summary,
    }


def _q_nonzero_layers(sample: dict[str, Any]) -> list[str]:
    q_field = sample["q_field"]
    layer_id = sample["layer_id"]
    meta = sample["meta"]
    layer_lookup = {
        layer.get("id"): layer.get("name")
        for layer in meta.get("layers", [])
        if isinstance(layer, dict)
    }
    ids = sorted({int(value) for value in layer_id[q_field[:, 0] != 0.0]})
    return [layer_lookup.get(layer_id_value, f"layer_{layer_id_value}") for layer_id_value in ids]


def run_mode(path: Path, mode: str) -> int:
    dataset = Heat3DV1MetadataDataset(path, k_encoding_mode=mode)
    builder = Heat3DGraphBuilder()

    info = dataset.describe()
    feature_dims = set()
    all_ok = True
    main_samples_ok = True
    diagnostic_ok = True
    exercised_k_shapes = set()

    print(f"\n=== k_encoding_mode={mode} ===")
    print(f"sample_count: {info['sample_count']}")
    print(f"supported_k_shapes: {info['supported_k_shapes']}")
    print(f"temperature_supported: {info['temperature_supported']}")
    print(f"solver_supported: {info['solver_supported']}")
    print(f"training_pipeline_integration: {info['training_pipeline_integration']}")

    for index, sample in enumerate(dataset.samples):
        model_input = dataset.get_model_input(index)
        raw_k_shape = sample["k_field"].shape
        exercised_k_shapes.add(f"(N,{raw_k_shape[1]})")
        graph_stage_pnode_features = np.expand_dims(model_input["features"], axis=0)

        metadata_ok = True
        graphs_ok = True
        metadata_summary = {}
        graph_summary = {}

        try:
            metadata = builder.build_metadata(sample["coords"])
            metadata_summary = _summarize_metadata(metadata)
        except Exception as exc:
            metadata_ok = False
            graphs_ok = False
            metadata_summary = {"error": f"{type(exc).__name__}: {exc}"}

        if metadata_ok:
            try:
                graphs = builder.build_graphs(metadata)
                graph_summary = {
                    graph_name: _summarize_typed_graph(getattr(graphs, graph_name))
                    for graph_name in graphs._fields
                }
            except Exception as exc:
                graphs_ok = False
                graph_summary = {"error": f"{type(exc).__name__}: {exc}"}

        sample_ok = metadata_ok and graphs_ok
        all_ok = all_ok and sample_ok
        if sample["sample_id"] != "sample_005":
            main_samples_ok = main_samples_ok and sample_ok
        else:
            diagnostic_ok = diagnostic_ok and sample_ok

        feature_dims.add(model_input["features"].shape[1])

        print(f"\n{sample['sample_id']}")
        print(f"  split: {sample['meta'].get('split')}")
        print(f"  raw k_field shape: {sample['k_field'].shape}")
        print(f"  encoded features shape: {model_input['features'].shape}")
        print(f"  feature_names: {model_input['feature_names']}")
        print(f"  node count: {sample['coords'].shape[0]}")
        print(f"  graph stage pnode_features shape: {graph_stage_pnode_features.shape}")
        print(f"  graph metadata build: {metadata_ok}")
        print(f"  graphs build: {graphs_ok}")
        print(f"  q_field nonzero layers: {_q_nonzero_layers(sample)}")
        print(f"  pure_physics mode: {model_input['input_mode'] == 'pure_physics'}")
        print(f"  metadata summary: {metadata_summary}")
        print(f"  graph summary: {graph_summary}")

    n6_status = (
        "not exercised in current dataset; native loader contract would preserve (N,6), "
        "but diag3 mode is not implemented for (N,6), and no (N,6) sample is generated"
    )
    print("\nmode summary")
    print(f"  feature dimensions seen: {sorted(feature_dims)}")
    print(f"  exercised k_field shapes: {sorted(exercised_k_shapes)}")
    print(f"  main (N,1) samples passed: {main_samples_ok}")
    print(f"  sample_005 passed: {diagnostic_ok}")
    print(f"  all samples passed: {all_ok}")
    print(f"  (N,6) graph-stage status: {n6_status}")

    return 0 if all_ok else 1


def main() -> int:
    args = parse_args()
    modes = ("native", "diag3") if args.mode == "both" else (args.mode,)
    exit_code = 0
    for mode in modes:
        try:
            status = run_mode(args.path, mode)
            exit_code = max(exit_code, status)
        except Exception as exc:
            print(f"\n=== k_encoding_mode={mode} FAILED ===")
            print(f"{type(exc).__name__}: {exc}")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
