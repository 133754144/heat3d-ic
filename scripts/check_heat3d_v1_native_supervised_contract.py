import argparse
from pathlib import Path
import sys

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset
from rigno.heat3d_v1_supervised import default_v1_supervised_samples_dir


TARGET_SAMPLE_IDS = ("sample_000", "sample_005")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the v1-native steady supervised input/target contract."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_supervised_samples_dir(REPO_DIR),
        help="Supervised smoke samples directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = Heat3DV1NativeSupervisedDataset(args.path, k_encoding_mode="diag3")
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in TARGET_SAMPLE_IDS if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"Required supervised smoke samples are missing: {missing}")

    examples = [dataset[index_by_id[sample_id]] for sample_id in TARGET_SAMPLE_IDS]
    reference_names = examples[0].condition.condition_feature_names
    same_contract = all(
        example.condition.condition_feature_names == reference_names
        for example in examples[1:]
    )

    target_not_in_conditions = all(
        "temperature" not in example.condition.condition_feature_names
        and "target_temperature" not in example.condition.condition_feature_names
        and "target_u" not in example.condition.condition_feature_names
        for example in examples
    )
    shapes_ok = all(
        example.condition.coords.shape[0] == example.condition.condition_features.shape[0]
        and example.condition.coords.shape[0] == example.target.target_u.shape[0]
        and example.target.target_u.ndim == 2
        and example.target.target_u.shape[1] == 1
        for example in examples
    )
    finite_ok = all(
        np.all(np.isfinite(example.condition.condition_features))
        and np.all(np.isfinite(example.target.target_u))
        and np.all(np.isfinite(example.condition.coords))
        for example in examples
    )
    status_ok = same_contract and target_not_in_conditions and shapes_ok and finite_ok

    print("v1-native supervised contract")
    print("  task: steady supervised operator learning for temperature prediction")
    print("  canonical mode: diag3")
    print("  condition semantics: coords + encoded_k_field + q_field + BC encoding")
    print("  target semantics: target_u / temperature.npy")
    print("  legacy bridge: old Inputs(u,c,...) packing is not canonical")

    for example in examples:
        print(f"\n{example.sample_id}")
        print(f"  split: {example.meta.get('split')}")
        print(f"  coords shape: {example.condition.coords.shape}")
        print(f"  condition_features shape: {example.condition.condition_features.shape}")
        print(f"  condition_feature_names: {example.condition.condition_feature_names}")
        print(f"  k_encoding_mode: {example.condition.k_encoding_mode}")
        print(f"  target_u shape: {example.target.target_u.shape}")
        print(f"  target_name: {example.target.target_name}")
        print(f"  target_role: {example.target.target_role}")
        print("  temperature in condition_features: False")

    print("\nsummary")
    print(f"  same condition feature contract: {same_contract}")
    print(f"  target absent from condition feature names: {target_not_in_conditions}")
    print(f"  shape checks ok: {shapes_ok}")
    print(f"  finite checks ok: {finite_ok}")
    print(f"  native supervised contract ok: {status_ok}")

    return 0 if status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
