import argparse
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.dataset_Heat3D_v1 import Heat3DV1MetadataDataset
from rigno.heat3d_v1_schema import default_v1_samples_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-check Heat3D v1 metadata-first loader in native and diag3 k modes."
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


def q_nonzero_layers(sample: dict) -> list[str]:
    q_field = sample["q_field"]
    layer_id = sample["layer_id"]
    meta = sample["meta"]
    layer_lookup = {
        layer.get("id"): layer.get("name")
        for layer in meta.get("layers", [])
        if isinstance(layer, dict)
    }
    layer_ids = sorted({int(value) for value in layer_id[q_field[:, 0] != 0.0]})
    return [layer_lookup.get(layer_id_value, f"layer_{layer_id_value}") for layer_id_value in layer_ids]


def run_mode(path: Path, mode: str) -> int:
    dataset = Heat3DV1MetadataDataset(path, k_encoding_mode=mode)
    info = dataset.describe()
    feature_dim = None

    print(f"\n=== k_encoding_mode={mode} ===")
    print(f"sample_count: {info['sample_count']}")
    print(f"supported_k_shapes: {info['supported_k_shapes']}")
    print(f"temperature_supported: {info['temperature_supported']}")
    print(f"solver_supported: {info['solver_supported']}")
    print(f"training_pipeline_integration: {info['training_pipeline_integration']}")

    for index, sample in enumerate(dataset.samples):
        model_input = dataset.get_model_input(index)
        feature_dim = model_input["features"].shape[1]
        print(f"\n{sample['sample_id']}")
        print(f"  split: {sample['meta'].get('split')}")
        print(f"  coords shape: {sample['coords'].shape}")
        print(f"  raw k_field shape: {sample['k_field'].shape}")
        print(f"  encoded k_field shape: {sample['encoded_k_field'].shape}")
        print(f"  q_field shape: {sample['q_field'].shape}")
        print(f"  bc_encoding shape: {sample['bc_encoding'].shape}")
        print(f"  features shape: {model_input['features'].shape}")
        print(f"  feature_names: {model_input['feature_names']}")
        print(f"  q_field nonzero layers: {q_nonzero_layers(sample)}")
        print(f"  pure_physics default: {model_input['input_mode'] == 'pure_physics'}")

    print("\nmode summary")
    print(f"  feature dimension: {feature_dim}")
    return 0


def main() -> int:
    args = parse_args()
    modes = ("native", "diag3") if args.mode == "both" else (args.mode,)
    exit_code = 0
    for mode in modes:
        try:
            run_mode(args.path, mode)
        except Exception as exc:
            print(f"\n=== k_encoding_mode={mode} FAILED ===")
            print(f"{type(exc).__name__}: {exc}")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
