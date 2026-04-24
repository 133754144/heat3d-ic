import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

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
BASE_T = 300.0
SHIFTED_T = 350.0
TOL = 1e-8

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
        description=(
            "Forward/loss smoke for v1 zero-delta temperature-rise bridge with relative BC features."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_supervised_samples_dir(REPO_DIR),
        help="Supervised smoke samples directory.",
    )
    return parser.parse_args()


def _copy_metadata_shifted_sample(source_dir: Path, temp_root: Path) -> Path:
    target_dir = temp_root / source_dir.name
    shutil.copytree(source_dir, target_dir)

    meta_path = target_dir / "sample_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["description"] = (
        f"{meta.get('description', '').strip()} "
        "Temporary diagnostic copy for zero-delta bridge input-invariance smoke only."
    ).strip()
    meta["boundary_params"]["bottom"]["fixed_temperature_K"] = SHIFTED_T
    meta["boundary_params"]["top"]["ambient_temperature_K"] = SHIFTED_T
    generation_config = dict(meta.get("generation_config", {}))
    generation_config.update({
        "diagnostic_only": True,
        "diagnostic_name": "zero_delta_bridge_input_invariance",
        "original_boundary_temperature_K": BASE_T,
        "shifted_boundary_temperature_K": SHIFTED_T,
        "formal_dataset_sample": False,
        "temperature_target_note": (
            "temperature.npy is copied only because the supervised loader requires it; "
            "the shifted diagnostic compares model-facing inputs only"
        ),
    })
    meta["generation_config"] = generation_config
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return target_dir


def _shape_of(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(value.shape)


def _range(values: np.ndarray) -> tuple[float, float]:
    return float(np.min(values)), float(np.max(values))


def _summary(values: np.ndarray) -> tuple[float, float, float]:
    return float(np.min(values)), float(np.max(values)), float(np.mean(values))


def _sample_by_id(dataset: Heat3DV1NativeSupervisedDataset, sample_id: str):
    index_by_id = dataset.sample_index_by_id()
    if sample_id not in index_by_id:
        raise ValueError(f"Missing sample {sample_id!r}")
    return dataset[index_by_id[sample_id]]


def _target_excluded(feature_names: tuple[str, ...]) -> bool:
    blocked = {"temperature", "target_temperature", "target_u"}
    return all(name not in blocked for name in feature_names)


def _raw_temperature_features_excluded(feature_names: tuple[str, ...]) -> bool:
    return "top_T_inf" not in feature_names and "bottom_T_fixed" not in feature_names


def _bridge_for(example):
    return example.build_temperature_rise_legacy_inputs_from_relative_features(
        bridge_policy="zero_delta_u_bridge",
    )


def _print_sample_smoke(sample_id: str, example, builder, model, params):
    bridge = _bridge_for(example)
    legacy_u = np.asarray(bridge.legacy_inputs.u)
    legacy_c = np.asarray(bridge.legacy_inputs.c)
    target_delta = np.asarray(bridge.target_delta_u)
    target_temperature = np.asarray(
        example.target.target_u.reshape(1, 1, example.target.target_u.shape[0], 1),
        dtype=np.float32,
    )
    reconstructed = np.asarray(bridge.t_ref) + target_delta
    reconstruction_ok = bool(np.allclose(reconstructed, target_temperature))
    zero_u_ok = bool(np.max(np.abs(legacy_u)) <= TOL)
    target_not_in_input = _target_excluded(bridge.condition_feature_names)
    raw_bc_excluded = _raw_temperature_features_excluded(bridge.condition_feature_names)
    kx_not_canonical_u = not bool(
        np.allclose(
            legacy_u.reshape(-1),
            example.condition.condition_features[:, 0],
        )
    )

    metadata = builder.build_metadata(example.condition.coords)
    graphs = builder.build_graphs(metadata)
    forward_ok = False
    loss_input_ok = False
    output_shape = None
    delta_mse_smoke = None
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
        delta_mse_smoke = float(jnp.mean(jnp.square(output - bridge.target_delta_u)))
        loss_input_ok = output.shape == bridge.target_delta_u.shape
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

    delta_min, delta_max, delta_mean = _summary(target_delta)
    legacy_u_min, legacy_u_max = _range(legacy_u)
    sample_ok = (
        reconstruction_ok
        and zero_u_ok
        and target_not_in_input
        and raw_bc_excluded
        and kx_not_canonical_u
        and forward_ok
        and loss_input_ok
    )

    print(f"\n{sample_id}")
    print(f"  relative condition feature names: {bridge.condition_feature_names}")
    print(f"  relative condition feature shape: {legacy_c.reshape(legacy_c.shape[2], -1).shape}")
    print(f"  T_ref value: {bridge.t_ref_value}")
    print(f"  T_ref source: {bridge.t_ref_source}")
    print(f"  legacy_inputs.u shape: {_shape_of(bridge.legacy_inputs.u)}")
    print(f"  legacy_inputs.u min/max: {legacy_u_min:.6f}, {legacy_u_max:.6f}")
    print(f"  legacy_inputs.c shape: {_shape_of(bridge.legacy_inputs.c)}")
    print(f"  target_delta_u shape: {tuple(target_delta.shape)}")
    print(f"  target_delta_u min/max/mean: {delta_min:.6f}, {delta_max:.6f}, {delta_mean:.6f}")
    print(f"  target_temperature in inputs: {not target_not_in_input}")
    print(f"  raw absolute BC temperatures in relative features: {not raw_bc_excluded}")
    print(f"  legacy_inputs.u is zero_delta_field: {zero_u_ok}")
    print(f"  k_x used as canonical u: {not kx_not_canonical_u}")
    print(f"  T_ref + target_delta_u == target_temperature: {reconstruction_ok}")
    print(f"  graph metadata/graphs built: True")
    print(f"  forward ok: {forward_ok}")
    print(f"  output shape: {output_shape}")
    print(f"  loss-input shape contract ok: {loss_input_ok}")
    print(f"  delta MSE smoke: {delta_mse_smoke}")
    if error_message:
        print(f"  error: {error_message}")
    print(f"  sample smoke ok: {sample_ok}")
    return params, sample_ok


def _check_shifted_input_invariance(sample_root: Path) -> bool:
    with tempfile.TemporaryDirectory(prefix="heat3d_v1_zero_delta_shift_") as temp_name:
        temp_root = Path(temp_name)
        for sample_id in TARGET_SAMPLE_IDS:
            _copy_metadata_shifted_sample(sample_root / sample_id, temp_root)

        base_dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode="diag3")
        shifted_dataset = Heat3DV1NativeSupervisedDataset(temp_root, k_encoding_mode="diag3")
        all_ok = True

        print("\nbaseline-shift model-facing input check")
        print("  shifted case changes bottom fixed temperature and top ambient temperature: 300K -> 350K")
        print("  shifted temperature targets are not used; this check compares model-facing inputs only")
        for sample_id in TARGET_SAMPLE_IDS:
            base_bridge = _bridge_for(_sample_by_id(base_dataset, sample_id))
            shifted_bridge = _bridge_for(_sample_by_id(shifted_dataset, sample_id))
            u_diff = float(
                np.max(
                    np.abs(
                        np.asarray(shifted_bridge.legacy_inputs.u)
                        - np.asarray(base_bridge.legacy_inputs.u)
                    )
                )
            )
            c_diff = float(
                np.max(
                    np.abs(
                        np.asarray(shifted_bridge.legacy_inputs.c)
                        - np.asarray(base_bridge.legacy_inputs.c)
                    )
                )
            )
            sample_ok = u_diff <= TOL and c_diff <= TOL
            all_ok = all_ok and sample_ok
            print(f"  {sample_id}")
            print(f"    max_abs(u_shifted - u_base): {u_diff:.9e}")
            print(f"    max_abs(c_shifted - c_base): {c_diff:.9e}")
            print(f"    model-facing inputs invariant: {sample_ok}")
        return all_ok


def main() -> int:
    args = parse_args()
    sample_root = args.path
    missing = [sample_id for sample_id in TARGET_SAMPLE_IDS if not (sample_root / sample_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing supervised smoke samples: {missing}")

    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode="diag3")
    builder = Heat3DGraphBuilder()
    model = GraphNeuralOperator(**MODEL_CONFIG)
    params = None
    status = {}

    print("v1 zero-delta temperature-rise bridge forward/loss smoke")
    print("  native task: coords + condition_features -> target_temperature")
    print("  bridge policy: legacy_inputs.u = zero_delta_field")
    print("  bridge policy: legacy_inputs.c = relative_condition_features")
    print("  target: target_delta_u = target_temperature - T_ref")
    print("  recovery: T_pred = T_ref + DeltaT_pred")
    print("  this is forward/loss smoke only: no training, optimizer, or update")

    for sample_id in TARGET_SAMPLE_IDS:
        example = _sample_by_id(dataset, sample_id)
        params, sample_ok = _print_sample_smoke(sample_id, example, builder, model, params)
        status[sample_id] = sample_ok

    shift_ok = _check_shifted_input_invariance(sample_root)
    all_ok = all(status.get(sample_id, False) for sample_id in TARGET_SAMPLE_IDS) and shift_ok

    print("\nsummary")
    print(f"  sample_000 zero-delta bridge smoke: {status.get('sample_000', False)}")
    print(f"  sample_005 zero-delta bridge smoke: {status.get('sample_005', False)}")
    print(f"  baseline-shift model-facing input invariance: {shift_ok}")
    print("  raw absolute BC temperatures enter model-facing inputs: False")
    print("  T_ref enters model-facing inputs: False")
    print("  T_ref role: target_delta construction and final temperature recovery only")
    print(f"  zero-delta forward/loss smoke ok: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
