import argparse
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_supervised import Heat3DV1SupervisedDataset, default_v1_supervised_samples_dir


TARGET_SAMPLE_IDS = ("sample_000", "sample_005")
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check v1 supervised normalization and target contract without training."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_supervised_samples_dir(REPO_DIR),
        help="Supervised smoke samples directory.",
    )
    return parser.parse_args()


def _select_examples(dataset: Heat3DV1SupervisedDataset):
    index_by_id = {sample["sample_id"]: idx for idx, sample in enumerate(dataset.samples)}
    missing = [sample_id for sample_id in TARGET_SAMPLE_IDS if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"Required supervised smoke samples are missing: {missing}")
    return [dataset.get_supervised_example(index_by_id[sample_id]) for sample_id in TARGET_SAMPLE_IDS]


def _combine_condition_features(examples) -> tuple[jnp.ndarray, tuple[str, ...]]:
    feature_names = examples[0].full_feature_names
    for example in examples[1:]:
        if example.full_feature_names != feature_names:
            raise ValueError("Feature-name contract mismatch across supervised smoke samples")

    features = []
    for example in examples:
        if example.inputs.c is None:
            feature = example.inputs.u
        else:
            feature = jnp.concatenate([example.inputs.u, example.inputs.c], axis=-1)
        features.append(feature)
    return jnp.concatenate(features, axis=0), feature_names


def _stats(array: jnp.ndarray, axes: tuple[int, ...]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean = jnp.mean(array, axis=axes, keepdims=True)
    std = jnp.std(array, axis=axes, keepdims=True)
    safe_std = jnp.where(std < EPS, 1.0, std)
    return mean, std, safe_std


def main() -> int:
    args = parse_args()
    dataset = Heat3DV1SupervisedDataset(args.path, k_encoding_mode="diag3")
    examples = _select_examples(dataset)

    condition_features, feature_names = _combine_condition_features(examples)
    target = jnp.concatenate([example.target_temperature for example in examples], axis=0)
    coords = jnp.concatenate([example.inputs.x_inp for example in examples], axis=0)

    cond_mean, cond_std, cond_safe_std = _stats(condition_features, axes=(0, 1, 2))
    target_mean, target_std, target_safe_std = _stats(target, axes=(0, 1, 2))

    normalized_conditions = (condition_features - cond_mean) / cond_safe_std
    restored_conditions = normalized_conditions * cond_safe_std + cond_mean
    normalized_target = (target - target_mean) / target_safe_std
    restored_target = normalized_target * target_safe_std + target_mean

    coord_min = jnp.min(coords, axis=(0, 1, 2), keepdims=True)
    coord_max = jnp.max(coords, axis=(0, 1, 2), keepdims=True)
    coord_span = jnp.where((coord_max - coord_min) < EPS, 1.0, coord_max - coord_min)
    normalized_coords = 2.0 * ((coords - coord_min) / coord_span) - 1.0

    cond_recon_error = float(jnp.max(jnp.abs(restored_conditions - condition_features)))
    cond_recon_scale = float(jnp.maximum(1.0, jnp.max(jnp.abs(condition_features))))
    cond_recon_rel_error = cond_recon_error / cond_recon_scale
    target_recon_error = float(jnp.max(jnp.abs(restored_target - target)))
    zero_std_features = [
        name
        for name, std_value in zip(feature_names, np.asarray(cond_std).reshape(-1))
        if float(std_value) < EPS
    ]

    finite_ok = all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in (
            condition_features,
            target,
            cond_mean,
            cond_std,
            normalized_conditions,
            target_mean,
            target_std,
            normalized_target,
            normalized_coords,
        )
    )
    shape_ok = (
        normalized_conditions.shape == condition_features.shape
        and normalized_target.shape == target.shape
        and normalized_coords.shape == coords.shape
    )
    recon_ok = cond_recon_rel_error <= 1e-6 and target_recon_error <= 1e-6
    target_nonconstant = bool(float(jnp.max(target_std)) > EPS)
    status_ok = finite_ok and shape_ok and recon_ok and target_nonconstant

    print("normalization contract")
    print("  mode: v1-specific steady supervised smoke")
    print("  input semantics: coords + encoded_k_field + q_field + BC encoding")
    print("  target semantics: temperature.npy as supervised steady temperature label")
    print("  adapter note: u/c split remains a model-interface compatibility detail")
    print("  recommended loss contract: MSE on normalized temperature target; report denormalized/raw metrics separately")

    print("\ncondition features")
    print(f"  feature_names: {feature_names}")
    print(f"  raw shape: {tuple(condition_features.shape)}")
    print(f"  normalized shape: {tuple(normalized_conditions.shape)}")
    print(f"  zero-std features using safe std=1.0: {zero_std_features}")
    for name, mean_value, std_value in zip(
        feature_names,
        np.asarray(cond_mean).reshape(-1),
        np.asarray(cond_std).reshape(-1),
    ):
        print(f"  {name}: mean={float(mean_value):.6g}, std={float(std_value):.6g}")

    print("\ntarget temperature")
    print(f"  raw shape: {tuple(target.shape)}")
    print(f"  normalized shape: {tuple(normalized_target.shape)}")
    print(f"  mean: {float(target_mean.reshape(-1)[0]):.6f}")
    print(f"  std: {float(target_std.reshape(-1)[0]):.6f}")
    print(f"  min/max: {float(jnp.min(target)):.6f}, {float(jnp.max(target)):.6f}")

    print("\ncoords")
    print(f"  raw shape: {tuple(coords.shape)}")
    print(f"  normalized shape: {tuple(normalized_coords.shape)}")
    print(f"  normalized min/max: {float(jnp.min(normalized_coords)):.6f}, {float(jnp.max(normalized_coords)):.6f}")

    print("\nchecks")
    print(f"  finite ok: {finite_ok}")
    print(f"  shape ok: {shape_ok}")
    print(f"  condition reconstruction max abs error: {cond_recon_error:.6e}")
    print(f"  condition reconstruction relative error: {cond_recon_rel_error:.6e}")
    print(f"  target reconstruction max abs error: {target_recon_error:.6e}")
    print(f"  target nonconstant: {target_nonconstant}")
    print(f"  normalization smoke ok: {status_ok}")

    return 0 if status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
