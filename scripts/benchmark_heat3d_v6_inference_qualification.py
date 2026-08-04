#!/usr/bin/env python3
"""Valid-only qualification timing for P1i and frozen V6 random-block transfer.

The same file acts as a single-process worker and as an orchestrator.  Every
family/route/state is executed in a separate process.  Cold measurements use
one fresh process per selected valid sample; new-case and fully-cached states
use one persistent process and never derive wall time by adding stage timers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import importlib.util
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_p1i_resolution as prior  # noqa: E402
import evaluate_heat3d_v6_common_valid_probe as common  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from rigno.heat3d_v1_native_supervised import V1SteadyConditionInput, V1SteadyTarget  # noqa: E402
from rigno.heat3d_v6_dataset import V6_DUAL_ROBIN_CONDITION_FEATURES, V6DualRobinExample  # noqa: E402
from rigno.heat3d_v6_full_field import build_reconstruction_map  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
from run_heat3d_v3_final_probe_checkpoint_smoke import install_checkpoint_feature_hooks  # noqa: E402
from evaluate_heat3d_v6_p1i_randomblock_transfer import layer_aware_fallback_map  # noqa: E402


ROUTES = ("model_support", "production_reconstruction", "fvm")
STATES = (
    "process_cold",
    "jit_cached_new_topology",
    "known_topology_new_physics",
    "fully_cached",
)
EDGE_FIELDS = ("p2r_edge_indices", "r2r_edge_indices", "r2r_edge_domains", "r2p_edge_indices")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Sequence[float], *, require_formal_count: bool = True) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if require_formal_count and array.size < 20:
        raise RuntimeError(f"formal timing requires >=20 measurements, got {array.size}")
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": array.tolist(),
    }


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def device_memory() -> dict[str, Any]:
    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    return {
        "device": str(device),
        "device_kind": device.device_kind,
        "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
        "bytes_limit": int(stats.get("bytes_limit", 0)),
    }


def canonical_role(family: str, raw: str) -> str:
    if family == "p1i":
        return raw
    return {"train": "train", "valid": "valid_iid", "test": "test_iid"}[raw]


class FamilyData:
    def __init__(
        self,
        *,
        family: str,
        dataset_root: Path,
        manifest_path: Path,
        full_fields_path: Path,
        randomblock_config: Path | None,
    ) -> None:
        self.family = family
        self.dataset_root = dataset_root
        self.manifest_path = manifest_path
        self.full_fields_path = full_fields_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.rows = list(self.manifest["samples"])
        self.valid_rows = [row for row in self.rows if canonical_role(family, str(row["split_role"])) == "valid_iid"]
        if len(self.rows) != 1024 or len(self.valid_rows) != 128:
            raise RuntimeError("frozen 1024/128 manifest contract failed")
        self.row_by_id = {str(row["sample_id"]): row for row in self.rows}
        self.randomblock_config = (
            yaml.safe_load(randomblock_config.read_text(encoding="utf-8"))
            if randomblock_config is not None else None
        )
        if family == "randomblock" and self.randomblock_config is None:
            raise RuntimeError("random-block physics config is required")
        self.randomblock_core = None
        if family == "randomblock":
            core_path = randomblock_config.parents[2] / "scripts" / "heat3d_v6_randomblock_core.py"
            spec = importlib.util.spec_from_file_location("frozen_heat3d_v6_randomblock_core", core_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load frozen random-block core: {core_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.randomblock_core = module
            self.randomblock_core_sha256 = sha256(core_path)

    def sample_dir(self, row: Mapping[str, Any]) -> Path:
        relative = str(row.get("relative_path") or row.get("sample_dir") or row["sample_id"])
        return self.dataset_root / relative

    def selected_rows(self, count: int) -> list[dict[str, Any]]:
        if count < 32:
            raise RuntimeError("qualification requires at least 32 valid samples")
        if self.family == "randomblock":
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in self.valid_rows:
                grouped[str(row["group_id"])].append(row)
            selected = []
            for group_id in sorted(grouped):
                selected.extend(sorted(grouped[group_id], key=lambda item: str(item["sample_id"]))[:2])
            if len(selected) != 32:
                raise RuntimeError("random-block qualification requires 16 groups x 2 variants")
            return selected
        ranked = sorted(
            self.valid_rows,
            key=lambda row: hashlib.sha256(str(row["sample_id"]).encode()).hexdigest(),
        )
        return ranked[:count]

    def warmup_rows(self, selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        selected_ids = {str(row["sample_id"]) for row in selected}
        if self.family == "p1i":
            return [next(row for row in self.valid_rows if str(row["sample_id"]) not in selected_ids)]
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.valid_rows:
            by_group[str(row["group_id"])].append(row)
        rows = []
        for group_id in sorted({str(row["group_id"]) for row in selected}):
            rows.append(next(row for row in sorted(by_group[group_id], key=lambda item: str(item["sample_id"])) if str(row["sample_id"]) not in selected_ids))
        return rows

    def load_example(self, row: Mapping[str, Any]) -> tuple[V6DualRobinExample, dict[str, Any]]:
        sample_dir = self.sample_dir(row)
        meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
        coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
        k = np.asarray(np.load(sample_dir / "k_field.npy"), dtype=np.float64)
        q = np.asarray(np.load(sample_dir / "q_field.npy"), dtype=np.float64).reshape(-1)
        flags = np.asarray(np.load(sample_dir / "bc_features.npy"), dtype=np.float64)[:, :4]
        cv = np.asarray(np.load(sample_dir / "control_volume.npy"), dtype=np.float64).reshape(-1)
        layer = np.asarray(np.load(sample_dir / "layer_id.npy"), dtype=np.int32).reshape(-1)
        if coords.shape != (1024, 3) or k.shape != (1024, 3) or flags.shape != (1024, 4):
            raise RuntimeError(f"{row['sample_id']}: support schema drift")
        if self.family == "p1i":
            physics = meta["physics"]
            top_h = float(meta["top_h_W_m2K"])
            bottom_h = float(meta["bottom_h_W_m2K"])
            top_t = bottom_t = float(physics["ambient_K"])
        else:
            bc = meta["boundary_conditions"]
            top_h = float(bc["top"]["h_W_m2K"])
            bottom_h = float(bc["bottom"]["h_W_m2K"])
            top_t = float(bc["top"]["T_inf_K"])
            bottom_t = float(bc["bottom"]["T_inf_K"])
        features = np.column_stack((
            k, q, flags, np.full(1024, top_h), np.full(1024, bottom_h), np.full(1024, top_t - bottom_t),
        ))
        enriched = dict(meta)
        enriched["v6_adapter"] = {
            "dataset_id": str(self.manifest["dataset_id"]),
            "manifest_split_role": "valid_iid",
            "group_id": str(row.get("group_id") or row["sample_id"]),
            "reference_temperature_K": bottom_t,
            "top_T_inf_K": top_t,
            "bottom_T_inf_K": bottom_t,
            "bottom_boundary_semantics": "robin_not_dirichlet",
            "operator_point_measure": "control_volume_frozen_irregular_1024",
        }
        example = V6DualRobinExample(
            sample_id=str(row["sample_id"]),
            condition=V1SteadyConditionInput(
                coords=coords,
                condition_features=features,
                condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
                k_encoding_mode="diag3",
            ),
            target=V1SteadyTarget(target_u=np.full((1024, 1), bottom_t, dtype=np.float64)),
            meta=enriched,
            operator_point_weights=cv,
        )
        return example, {"coords": coords, "cv": cv, "layer": layer, "q": q, "reference_K": bottom_t}

    def physics(self, row: Mapping[str, Any]) -> dict[str, Any]:
        if self.family == "randomblock":
            return deepcopy(self.randomblock_config["physics"])
        return json.loads((self.sample_dir(row) / "sample_meta.json").read_text())["physics"]

    def full_shared(self) -> dict[str, np.ndarray]:
        with h5py.File(self.full_fields_path, "r") as archive:
            if self.family == "p1i":
                return {
                    "coords": np.asarray(archive["shared/coords_m"][:], dtype=np.float64),
                    "cv": np.asarray(archive["shared/control_volume_m3"][:], dtype=np.float64),
                    "layer": np.asarray(archive["shared/layer_id"][:], dtype=np.int32),
                }
            return {
                "coords": np.asarray(archive["coords"][:], dtype=np.float64),
                "cv": np.asarray(archive["control_volume"][:], dtype=np.float64),
                "layer": np.asarray(archive["layer_id"][:], dtype=np.int32),
            }

    def archive_lookup(self) -> dict[str, int]:
        with h5py.File(self.full_fields_path, "r") as archive:
            key = "samples/sample_id" if self.family == "p1i" else "sample_id"
            ids = [value.decode() if isinstance(value, bytes) else str(value) for value in archive[key][:]]
        return {sample_id: index for index, sample_id in enumerate(ids)}

    def truth(self, row: Mapping[str, Any], *, include_full_kq: bool) -> dict[str, np.ndarray]:
        sample_dir = self.sample_dir(row)
        reference = 300.0
        support_temperature = np.asarray(np.load(sample_dir / "temperature.npy"), dtype=np.float64).reshape(-1)
        lookup = self.archive_lookup()
        archive_row = lookup[str(row["sample_id"])]
        with h5py.File(self.full_fields_path, "r") as archive:
            if self.family == "p1i":
                full_delta = np.asarray(archive["samples/deltaT_K"][archive_row], dtype=np.float64)
                result = {"support_delta": support_temperature - reference, "full_delta": full_delta}
            else:
                full_temperature = np.asarray(archive["temperature_K"][archive_row], dtype=np.float64)
                result = {"support_delta": support_temperature - reference, "full_delta": full_temperature - reference}
                if include_full_kq:
                    result["full_k"] = np.asarray(archive["k_xyz_W_mK"][archive_row], dtype=np.float64)
                    result["full_q"] = np.asarray(archive["q_W_m3"][archive_row], dtype=np.float64)
        return result


class FixedEdgeTargetBuilder:
    """Pad only existing dummy edges to a preregistered fixed JIT shape."""

    def __init__(self, builder: Any, targets: Mapping[str, int | None]):
        self._builder = builder
        self.targets = dict(targets)

    @property
    def config(self):
        return self._builder.config

    def __getattr__(self, name: str):
        return getattr(self._builder, name)

    def build_metadata(self, coords, key=None):
        metadata = self._builder.build_metadata(coords, key=key)
        replacements = {}
        for field in EDGE_FIELDS:
            value = getattr(metadata, field)
            target = self.targets.get(field)
            if value is None:
                if target is not None:
                    raise RuntimeError(f"{field}: target set for absent edge family")
                replacements[field] = None
                continue
            if target is None or int(value.shape[1]) > int(target):
                raise RuntimeError(f"{field}: edge count {value.shape[1]} exceeds target {target}")
            pad_count = int(target) - int(value.shape[1])
            replacements[field] = value if not pad_count else jnp.concatenate(
                (value, jnp.repeat(value[:, -1:, :], pad_count, axis=1)), axis=1
            )
        return type(metadata)(**{
            field: replacements.get(field, getattr(metadata, field))
            for field in metadata._fields
        })


class ModelRuntime:
    def __init__(
        self,
        run_dir: Path,
        checkpoint_sha: str,
        checkpoint_epoch: int,
        edge_targets: Path | None,
        *,
        verify_checkpoint_sha: bool = True,
    ) -> None:
        self.run_dir = run_dir
        checkpoint_path = run_dir / "params_best_valid_point_global.pkl"
        if verify_checkpoint_sha and sha256(checkpoint_path) != checkpoint_sha:
            raise RuntimeError("checkpoint SHA mismatch")
        self.run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        checkpoint = runner._load_params_checkpoint(checkpoint_path)
        if int(checkpoint["epoch"]) != checkpoint_epoch:
            raise RuntimeError("checkpoint epoch mismatch")
        stats = common._materialize_checkpoint_stats(checkpoint["train_only_normalization"])
        checkpoint = dict(checkpoint)
        checkpoint["train_only_normalization"] = stats
        install_checkpoint_feature_hooks(stats)
        self.checkpoint = checkpoint
        self.stats = stats
        self.model_config = runner._resolve_decoder_bypass_model_config(dict(checkpoint["model_config"]), stats)
        self.model = GraphNeuralOperator(**self.model_config)
        self.params = runner._device_params(checkpoint["params"])
        # Preserve the production runner's run-level shared-support contract.
        # Recreating this wrapper per sample silently defeats graph reuse and
        # can trigger one XLA specialization per otherwise identical P1i case.
        base_builder = runner.Heat3DGraphBuilder(**dict(self.run_config["graph_config"]))
        if edge_targets is not None:
            target_payload = json.loads(edge_targets.read_text(encoding="utf-8"))
            mode = target_payload.get("mode", "fixed_dummy_edge_padding_v1")
            if mode == "fixed_dummy_edge_padding_v1":
                base_builder = FixedEdgeTargetBuilder(base_builder, target_payload["edge_targets"])
            elif mode != "raw_shape_family_v1":
                raise RuntimeError(f"unsupported edge JIT contract mode: {mode}")
            self.edge_targets_sha256 = sha256(edge_targets)
        else:
            self.edge_targets_sha256 = None
        self.builder = runner.RunSharedSupportGraphBuilder(base_builder)
        self.compiled_apply = jax.jit(
            lambda params, model_group: runner._model_apply(
                self.model, params, model_group
            )
        )

    def graph(self, example: V6DualRobinExample) -> dict[str, Any]:
        groups = runner._make_v6_padded_groups_with_progress(
            [example], self.stats, self.builder, "v6_inference_qualification",
            False, "off", int(self.run_config["graph_seed"]),
            batch_size=1, drop_last=False,
        )
        context_payload = self.run_config.get("global_context") or {}
        standardizer = context_payload.get("standardizer") or {}
        if standardizer.get("fit_population") != "train_only" or int(standardizer.get("fit_sample_count", -1)) != 768:
            raise RuntimeError("global context standardizer is not frozen train-only/768")
        encoded = {
            example.sample_id: common.standardize_v6_contexts(
                [runner._global_context_row_for_example(example)], standardizer
            )[0]
        }
        runner._attach_global_context_to_groups(
            groups, encoded,
            expected_feature_dim=int(self.model_config["global_context_feature_dim"]),
        )
        by_id = {example.sample_id: example}
        runner._attach_native_physics_to_groups(groups, by_id)
        if (
            self.model_config.get("scale_pooling") == "qk_gated"
            or self.model_config.get("shape_attention_mode") != "none"
            or self.model_config.get("scale_attention_mode") != "none"
        ):
            runner._attach_qk_region_features_to_groups(
                groups, by_id,
                feature_version=self.model_config["qk_region_feature_version"],
            )
        if self.model_config.get("scale_deepsets_mode", "none") != "none":
            runner._attach_scale_deepsets_weights_to_groups(groups, by_id)
        return groups[0]

    def forward(self, group: Mapping[str, Any]) -> np.ndarray:
        # Exclude reporting-only strings/sample IDs from the JIT pytree.  This
        # gives "JIT cached, new case" its literal meaning while preserving the
        # frozen model call and every numerical input used by that call.
        model_group = {
            key: group[key]
            for key in (
                "inputs", "graphs", "global_context", "native_physics",
                "qk_region_features", "scale_context",
                "scale_region_source_weights", "scale_region_volume_weights",
            )
            if key in group
        }
        output = self.compiled_apply(self.params, model_group)
        raw = output["raw_temperature"]
        jax.block_until_ready(raw)
        return np.asarray(raw, dtype=np.float64)[0, 0, :, 0]


def support_key(example: V6DualRobinExample) -> str:
    return hashlib.sha256(np.asarray(example.condition.coords, dtype=np.float64).tobytes()).hexdigest()


def build_map(example: V6DualRobinExample, shared: Mapping[str, np.ndarray]):
    distance, indices = cKDTree(shared["coords"]).query(np.asarray(example.condition.coords), k=1)
    if float(np.max(distance)) > 1e-14 or len(np.unique(indices)) != 1024:
        raise RuntimeError("support is not an exact full-mesh subset")
    boundaries = np.asarray(example.meta.get("physics", {}).get("layer_boundaries_m") or [], dtype=np.float64)
    if boundaries.size == 0:
        layers = example.meta.get("layers_bottom_to_top") or example.meta.get("physics", {}).get("layers_bottom_to_top")
        boundaries = [0.0]
        for layer in layers:
            boundaries.append(boundaries[-1] + float(layer["thickness_m"]))
        boundaries = np.asarray(boundaries, dtype=np.float64)
    try:
        mapping, audit = build_reconstruction_map(
            coords=shared["coords"], layer_id=shared["layer"],
            boundaries=boundaries, support_indices=np.asarray(indices, dtype=np.int32),
        )
        mode = "strict_layer_interface_v1"
    except RuntimeError as error:
        if "support domain is empty" not in str(error):
            raise
        mapping = layer_aware_fallback_map(shared["coords"], shared["layer"], np.asarray(indices, dtype=np.int32))
        audit = {"label_independent": True, "fallback": "same_layer"}
        mode = "explicit_layer_aware_fallback_v1"
    return mapping, {"mode": mode, "audit": audit}


def serialize(array: np.ndarray) -> int:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return len(buffer.getvalue())


def metric_accumulate(
    rows: Sequence[dict[str, Any]],
    *,
    full: bool,
) -> dict[str, Any]:
    sse = energy = weighted_sse = weighted_energy = volume = 0.0
    sample_rel = []
    peak_errors = []
    region_sse = {name: 0.0 for name in ("source", "background", "top", "bottom")}
    region_volume = {name: 0.0 for name in region_sse}
    layer_errors = []
    interface_errors = []
    for row in rows:
        prediction = row["prediction"]
        truth = row["truth"]
        weights = row["weights"]
        coords = row["coords"]
        layers = row["layer"]
        q = row["q"]
        error = prediction - truth
        sse += float(np.sum(error * error)); energy += float(np.sum(truth * truth))
        weighted_sse += float(np.sum(weights * error * error))
        weighted_energy += float(np.sum(weights * truth * truth)); volume += float(np.sum(weights))
        sample_rel.append(math.sqrt(float(np.sum(weights * error * error)) / float(np.sum(weights * truth * truth))) * 100.0)
        peak_errors.append(float(np.max(prediction) - np.max(truth)))
        masks = {
            "source": q > 0.0,
            "background": q <= 0.0,
            "top": np.isclose(coords[:, 2], np.max(coords[:, 2]), atol=1e-15),
            "bottom": np.isclose(coords[:, 2], np.min(coords[:, 2]), atol=1e-15),
        }
        for name, mask in masks.items():
            region_sse[name] += float(np.sum(weights[mask] * error[mask] ** 2))
            region_volume[name] += float(np.sum(weights[mask]))
        means = []
        for layer in sorted(np.unique(layers)):
            mask = layers == layer
            means.append(float(np.sum(weights[mask] * error[mask]) / np.sum(weights[mask])))
        layer_errors.extend(means)
        if len(means) > 1:
            interface_errors.extend(np.diff(means).tolist())
    return {
        "domain": "full_240825" if full else "support_1024",
        "sample_count": len(rows),
        "point_global_true_rms_relative_rmse_pct": math.sqrt(sse / energy) * 100.0,
        "sample_first_cv_relative_rmse_pct": float(np.mean(sample_rel)),
        "raw_cv_weighted_rmse_K": math.sqrt(weighted_sse / volume),
        "peak_rmse_K": math.sqrt(float(np.mean(np.square(peak_errors)))),
        "source_rmse_K": math.sqrt(region_sse["source"] / region_volume["source"]),
        "background_rmse_K": math.sqrt(region_sse["background"] / region_volume["background"]),
        "top_rmse_K": math.sqrt(region_sse["top"] / region_volume["top"]),
        "bottom_rmse_K": math.sqrt(region_sse["bottom"] / region_volume["bottom"]),
        "layer_mean_rmse_K": math.sqrt(float(np.mean(np.square(layer_errors)))),
        "interface_drop_rmse_K": math.sqrt(float(np.mean(np.square(interface_errors)))),
    }


def model_measurements(
    data: FamilyData,
    runtime: ModelRuntime,
    rows: Sequence[Mapping[str, Any]],
    *,
    route: str,
    state: str,
    collect_metrics: bool,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if route not in {"model_support", "production_reconstruction"}:
        raise ValueError(route)
    shared = data.full_shared() if route == "production_reconstruction" else None
    graph_cache: dict[str, dict[str, Any]] = {}
    map_cache: dict[str, Any] = {}
    cache_preparation_seconds = 0.0
    if state == "jit_cached_new_topology":
        if data.family == "randomblock":
            raise RuntimeError(
                "random-block unseen topology changes raw edge shapes; fixed padding failed "
                "the frozen numerical-equivalence gate, so a JIT-cache-hit claim is forbidden"
            )
        warm_keys = set()
        for warm_row in data.warmup_rows(rows):
            example, _ = data.load_example(warm_row)
            warm_keys.add(support_key(example))
            group = runtime.graph(example)
            runtime.forward(group)
        measured_keys = {support_key(data.load_example(row)[0]) for row in rows}
        if warm_keys & measured_keys:
            raise RuntimeError("new-topology timing reused a warmed support hash")
    elif state == "known_topology_new_physics":
        if data.family != "randomblock":
            raise RuntimeError(
                "P1i has no preregistered valid pair sharing a support hash; "
                "known-topology/new-physics is not applicable"
            )
        for warm_row in data.warmup_rows(rows):
            example, _ = data.load_example(warm_row)
            runtime.forward(runtime.graph(example))
    elif state == "fully_cached":
        started = time.perf_counter()
        for row in rows:
            example, _ = data.load_example(row)
            sample_id = str(row["sample_id"])
            key = support_key(example)
            graph_cache[sample_id] = runtime.graph(example)
            runtime.forward(graph_cache[sample_id])
            if route == "production_reconstruction" and key not in map_cache:
                map_cache[key] = build_map(example, shared)[0]
        cache_preparation_seconds = time.perf_counter() - started

    measurements = []
    metric_rows_support = []
    metric_rows_full = []
    oracle_rows_full = []
    map_modes: dict[str, int] = defaultdict(int)
    for row in rows:
        e2e_started = time.perf_counter()
        started = time.perf_counter(); example, public = data.load_example(row); data_s = time.perf_counter() - started
        sample_id = str(row["sample_id"])
        key = support_key(example)
        started = time.perf_counter()
        group = graph_cache.get(sample_id)
        if group is None:
            group = runtime.graph(example)
        graph_s = time.perf_counter() - started
        started = time.perf_counter(); prediction_temperature = runtime.forward(group); forward_s = time.perf_counter() - started
        map_build_s = map_apply_s = 0.0
        output_array = prediction_temperature
        mapping = None
        if route == "production_reconstruction":
            started = time.perf_counter()
            mapping = map_cache.get(key)
            if mapping is None:
                mapping, map_audit = build_map(example, shared)
                map_modes[map_audit["mode"]] += 1
            map_build_s = time.perf_counter() - started
            started = time.perf_counter()
            output_array = mapping.reconstruct(prediction_temperature - public["reference_K"])
            map_apply_s = time.perf_counter() - started
        started = time.perf_counter(); output_bytes = serialize(output_array); output_s = time.perf_counter() - started
        wall = time.perf_counter() - e2e_started
        measurements.append({
            "data_seconds": data_s, "graph_seconds": graph_s,
            "jit_or_forward_seconds": forward_s, "map_build_seconds": map_build_s,
            "map_apply_seconds": map_apply_s, "output_seconds": output_s,
            "continuous_wall_seconds": wall, "output_bytes": output_bytes,
            "prediction_serialization_completed_monotonic_s": time.perf_counter(),
        })

        # Accuracy and oracle work are deliberately after the timed production span.
        if collect_metrics:
            truth = data.truth(row, include_full_kq=False)
            metric_rows_support.append({
                "prediction": prediction_temperature - public["reference_K"], "truth": truth["support_delta"],
                "weights": public["cv"], "coords": public["coords"], "layer": public["layer"], "q": public["q"],
            })
        if route == "production_reconstruction" and collect_metrics:
            full_q = data.full_kq(row)[1]
            metric_rows_full.append({
                "prediction": output_array, "truth": truth["full_delta"], "weights": shared["cv"],
                "coords": shared["coords"], "layer": shared["layer"], "q": full_q,
            })
            oracle = mapping.reconstruct(truth["support_delta"])
            oracle_rows_full.append({
                "prediction": oracle, "truth": truth["full_delta"], "weights": shared["cv"],
                "coords": shared["coords"], "layer": shared["layer"], "q": full_q,
            })
    metrics = {"support": metric_accumulate(metric_rows_support, full=False)} if metric_rows_support else {}
    if metric_rows_full:
        metrics["full_field"] = metric_accumulate(metric_rows_full, full=True)
        metrics["oracle_reconstruction"] = metric_accumulate(oracle_rows_full, full=True)
    return measurements, {
        "metrics": metrics,
        "cache_preparation_seconds_outside_timing": cache_preparation_seconds,
        "reconstruction_mode_counts": dict(map_modes),
        "metrics_and_oracle_outside_production_timing": True,
    }


def _family_full_kq(self: FamilyData, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if self.family == "randomblock":
        lookup = self.archive_lookup()
        archive_row = lookup[str(row["sample_id"])]
        with h5py.File(self.full_fields_path, "r") as archive:
            return (
                np.asarray(archive["k_xyz_W_mK"][archive_row], dtype=np.float64),
                np.asarray(archive["q_W_m3"][archive_row], dtype=np.float64),
            )
    meta = json.loads((self.sample_dir(row) / "sample_meta.json").read_text(encoding="utf-8"))
    mesh = prior.core.build_mesh(meta["physics"])
    k, q, audit = prior._continuous_fields(meta, mesh)
    if audit["relative_power_error"] > 1e-12:
        raise RuntimeError("P1i full-field source power drift")
    return k, q


FamilyData.full_kq = _family_full_kq  # type: ignore[attr-defined]


def randomblock_assemble(mesh, k_diag, q, top_h: float, bottom_h: float):
    """Exact assembly path from the frozen random-block generator, split for timing."""
    core = mesh["_randomblock_core"]
    i, j, conductance = core._neighbor_faces(mesh, k_diag)
    n = int(mesh["node_count"])
    diagonal = np.bincount(
        np.concatenate((i, j)), weights=np.concatenate((conductance, conductance)), minlength=n
    )
    rhs = np.asarray(q, dtype=np.float64) * np.asarray(mesh["weights"], dtype=np.float64)
    grid = np.asarray(mesh["grid"], dtype=np.int64)
    dx, dy, _ = mesh["widths"]
    boundary_area = (dx[:, None] * dy[None, :]).reshape(-1)
    top_nodes, bottom_nodes = grid[:, :, -1].reshape(-1), grid[:, :, 0].reshape(-1)
    top_robin, bottom_robin = top_h * boundary_area, bottom_h * boundary_area
    diagonal[top_nodes] += top_robin; diagonal[bottom_nodes] += bottom_robin
    rhs[top_nodes] += top_robin * 300.0; rhs[bottom_nodes] += bottom_robin * 300.0
    rows = np.concatenate((i, j, np.arange(n, dtype=np.int64)))
    cols = np.concatenate((j, i, np.arange(n, dtype=np.int64)))
    values = np.concatenate((-conductance, -conductance, diagonal))
    matrix = csr_matrix((values, (rows, cols)), shape=(n, n))
    preconditioner = LinearOperator(
        (n, n), matvec=lambda value: np.asarray(value, dtype=np.float64) / diagonal, dtype=np.float64
    )
    return matrix, rhs, preconditioner


def randomblock_solve(system):
    matrix, rhs, preconditioner = system
    iterations = 0
    def callback(_):
        nonlocal iterations
        iterations += 1
    temperature, info = cg(
        matrix, rhs, x0=np.full(rhs.size, 300.0), rtol=1e-10, atol=0.0,
        maxiter=20000, M=preconditioner, callback=callback,
    )
    if info != 0:
        raise RuntimeError(f"random-block CG failed: {info}")
    return np.asarray(temperature, dtype=np.float64), iterations


def fvm_measurements(
    data: FamilyData,
    rows: Sequence[Mapping[str, Any]],
    *,
    state: str,
    collect_metrics: bool,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    first_physics = data.physics(rows[0])
    if data.family == "randomblock":
        mesh = data.randomblock_core.build_mesh(first_physics)
        mesh["_randomblock_core"] = data.randomblock_core
    else:
        mesh = prior.core.build_mesh(first_physics)
    shared = data.full_shared()
    if not np.array_equal(mesh["coords"], shared["coords"]):
        raise RuntimeError("dataset-consistent FVM mesh coordinate drift")
    system_cache: dict[str, tuple[Any, Any, Any]] = {}
    example_cache: dict[str, V6DualRobinExample] = {}
    q_cache: dict[str, np.ndarray] = {}
    cache_preparation_seconds = 0.0
    if state == "fully_cached":
        started = time.perf_counter()
        for row in rows:
            k, q = data.full_kq(row)
            example, _ = data.load_example(row)
            sample_id = str(row["sample_id"])
            example_cache[sample_id] = example
            q_cache[sample_id] = q
            top_h = float(example.condition.condition_features[0, 8])
            bottom_h = float(example.condition.condition_features[0, 9])
            system_cache[sample_id] = (
                randomblock_assemble(mesh, k, q, top_h, bottom_h)
                if data.family == "randomblock"
                else prior._assemble(mesh, k, q, top_h, bottom_h)
            )
        cache_preparation_seconds = time.perf_counter() - started

    measurements = []
    metric_rows = []
    cg_iterations = []
    for row in rows:
        e2e_started = time.perf_counter()
        sample_id = str(row["sample_id"])
        started = time.perf_counter()
        if state == "fully_cached":
            example = example_cache[sample_id]
            q = q_cache[sample_id]
            k = None
        else:
            example, _ = data.load_example(row)
            k, q = data.full_kq(row)
        data_s = time.perf_counter() - started
        started = time.perf_counter()
        system = system_cache.get(sample_id)
        if system is None:
            top_h = float(example.condition.condition_features[0, 8])
            bottom_h = float(example.condition.condition_features[0, 9])
            system = (
                randomblock_assemble(mesh, k, q, top_h, bottom_h)
                if data.family == "randomblock"
                else prior._assemble(mesh, k, q, top_h, bottom_h)
            )
        assembly_s = time.perf_counter() - started
        started = time.perf_counter()
        if data.family == "randomblock":
            temperature, iterations = randomblock_solve(system)
            cg_iterations.append(iterations)
        else:
            temperature = prior._solve(*system)
        solve_s = time.perf_counter() - started
        started = time.perf_counter(); output_bytes = serialize(temperature - 300.0); output_s = time.perf_counter() - started
        wall = time.perf_counter() - e2e_started
        measurements.append({
            "data_seconds": data_s, "assembly_seconds": assembly_s,
            "linear_solve_seconds": solve_s, "output_seconds": output_s,
            "continuous_wall_seconds": wall, "output_bytes": output_bytes,
            "prediction_serialization_completed_monotonic_s": time.perf_counter(),
        })
        if collect_metrics:
            truth = data.truth(row, include_full_kq=False)
            metric_rows.append({
                "prediction": temperature - 300.0, "truth": truth["full_delta"],
                "weights": shared["cv"], "coords": shared["coords"], "layer": shared["layer"], "q": q,
            })
    return measurements, {
        "metrics": {"full_field": metric_accumulate(metric_rows, full=True)} if metric_rows else {},
        "cache_preparation_seconds_outside_timing": cache_preparation_seconds,
        "solver": {"method": "dataset_generator_consistent_FVM_CG", "rtol": 1e-10, "atol": 0.0, "maxiter": 20000},
        "frozen_solver_core_sha256": (
            data.randomblock_core_sha256 if data.family == "randomblock" else sha256(ROOT / "scripts/heat3d_v6_p1i_continuous_core.py")
        ),
        "measured_cg_iterations": (
            distribution(cg_iterations, require_formal_count=len(cg_iterations) >= 20)
            if cg_iterations else None
        ),
        "metrics_outside_production_timing": True,
    }


def summarize_measurements(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in row if key.endswith("_seconds")})
    return {
        key: distribution(
            [float(row[key]) for row in rows],
            require_formal_count=len(rows) >= 20,
        )
        for key in keys
    }


def prepare_edge_targets(args: argparse.Namespace) -> int:
    data = FamilyData(
        family=args.family, dataset_root=args.dataset_root, manifest_path=args.manifest,
        full_fields_path=args.full_fields, randomblock_config=args.randomblock_config,
    )
    runtime = ModelRuntime(args.run_dir, args.checkpoint_sha256, args.checkpoint_epoch, None)
    selected = data.selected_rows(args.sample_count)
    warmup = data.warmup_rows(selected)
    builder = runner.Heat3DGraphBuilder(**dict(runtime.run_config["graph_config"]))
    targets: dict[str, int | None] = {field: None for field in EDGE_FIELDS}
    raw_counts = []
    started = time.perf_counter()
    for row in [*selected, *warmup]:
        example, _ = data.load_example(row)
        metadata = builder.build_metadata(
            runner._graph_coords_for_example(example, runtime.stats),
            key=runner._metadata_key(int(runtime.run_config["graph_seed"])),
        )
        counts = {}
        for field in EDGE_FIELDS:
            value = getattr(metadata, field)
            count = None if value is None else int(value.shape[1])
            counts[field] = count
            if count is not None:
                targets[field] = max(int(targets[field] or 0), count)
        raw_counts.append({"sample_id": row["sample_id"], "edge_counts": counts})
    payload = {
        "schema_version": "heat3d_v6_fixed_edge_jit_targets_v1",
        "mode": args.edge_contract_mode,
        "family": args.family, "dataset_id": data.manifest["dataset_id"],
        "selection_sample_ids": [row["sample_id"] for row in selected],
        "warmup_sample_ids": [row["sample_id"] for row in warmup],
        "edge_targets": (targets if args.edge_contract_mode == "fixed_dummy_edge_padding_v1" else None),
        "raw_counts": raw_counts,
        "preparation_seconds_outside_timing": time.perf_counter() - started,
        "padding_semantics": "repeat_existing_dummy_edge_only",
        "target_or_test_labels_used": False, "test_accessed": False, "sealed_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "family": args.family, "edge_targets": targets}, sort_keys=True))
    return 0


def worker(args: argparse.Namespace) -> int:
    process_started = time.perf_counter()
    data = FamilyData(
        family=args.family, dataset_root=args.dataset_root, manifest_path=args.manifest,
        full_fields_path=args.full_fields, randomblock_config=args.randomblock_config,
    )
    selected = data.selected_rows(args.sample_count)
    if args.sample_id:
        selected = [data.row_by_id[args.sample_id]]
    collect_metrics = args.sample_id is None
    if args.route == "fvm":
        measurements, extra = fvm_measurements(data, selected, state=args.state, collect_metrics=collect_metrics)
    else:
        runtime = ModelRuntime(
            args.run_dir,
            args.checkpoint_sha256,
            args.checkpoint_epoch,
            args.edge_targets,
            verify_checkpoint_sha=not args.checkpoint_sha_preverified,
        )
        measurements, extra = model_measurements(
            data, runtime, selected, route=args.route, state=args.state,
            collect_metrics=collect_metrics,
        )
    payload = {
        "schema_version": "heat3d_v6_inference_qualification_worker_v1",
        "status": "passed",
        "family": args.family, "route": args.route, "state": args.state,
        "sample_ids": [str(row["sample_id"]) for row in selected],
        "sample_count": len(selected),
        "stage_timing": summarize_measurements(measurements),
        "measurements": measurements,
        "process_internal_wall_seconds": time.perf_counter() - process_started,
        "process_peak_ram_bytes": rss_bytes(), "device_memory": device_memory(),
        "environment": {
            "host": platform.node(), "platform": platform.platform(), "python": sys.version,
            "jax": jax.__version__, "numpy": np.__version__, "device": str(jax.devices()[0]),
            "cpu_count": os.cpu_count(), "batch_size": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "dataset": {"id": data.manifest["dataset_id"], "manifest_sha256": sha256(args.manifest), "full_fields_sha256": sha256(args.full_fields)},
        "checkpoint": None if args.route == "fvm" else {"sha256": args.checkpoint_sha256, "epoch": args.checkpoint_epoch},
        "fixed_edge_jit_targets_sha256": (
            sha256(args.edge_targets) if args.edge_targets is not None and args.route != "fvm" else None
        ),
        "accessed_roles": (["valid_iid", "train_frozen_normalization_metadata"] if args.route != "fvm" else ["valid_iid"]),
        "test_accessed": False, "sealed_accessed": False,
        "training_executed": False, "checkpoint_modified": False,
        **extra,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "family": args.family, "route": args.route, "state": args.state, "samples": len(selected)}, sort_keys=True))
    return 0


def worker_command(args: argparse.Namespace, *, route: str, state: str, output: Path, sample_id: str | None = None) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker", "--family", args.family,
        "--route", route, "--state", state, "--sample-count", str(args.sample_count),
        "--dataset-root", str(args.dataset_root), "--manifest", str(args.manifest),
        "--full-fields", str(args.full_fields), "--run-dir", str(args.run_dir),
        "--checkpoint-sha256", args.checkpoint_sha256, "--checkpoint-epoch", str(args.checkpoint_epoch),
        "--output", str(output),
    ]
    if args.edge_targets is not None:
        command.extend(("--edge-targets", str(args.edge_targets)))
    if args.randomblock_config is not None:
        command.extend(("--randomblock-config", str(args.randomblock_config)))
    if args.checkpoint_sha_preverified:
        command.append("--checkpoint-sha-preverified")
    if sample_id is not None:
        command.extend(("--sample-id", sample_id))
    return command


def run_process(command: list[str], *, fvm: bool) -> tuple[dict[str, Any], float, str]:
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "MEM_FRACTION": "0.85", "HEAT3D_REPO_ROOT": str(ROOT),
    })
    if fvm:
        env["JAX_PLATFORMS"] = "cpu"
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    external_wall = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"worker failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}\n{completed.stderr}")
    output_path = Path(command[command.index("--output") + 1])
    payload = json.loads(output_path.read_text())
    if payload["state"] == "process_cold":
        cutoff = float(payload["measurements"][-1]["prediction_serialization_completed_monotonic_s"])
        external_wall = cutoff - started
        if external_wall <= 0.0:
            raise RuntimeError("invalid cross-process monotonic cold cutoff")
    return payload, external_wall, completed.stdout[-2000:]


def orchestrate(args: argparse.Namespace) -> int:
    checkpoint_path = args.run_dir / "params_best_valid_point_global.pkl"
    if sha256(checkpoint_path) != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA preflight failed")
    args.checkpoint_sha_preverified = True
    data = FamilyData(
        family=args.family, dataset_root=args.dataset_root, manifest_path=args.manifest,
        full_fields_path=args.full_fields, randomblock_config=args.randomblock_config,
    )
    selected = data.selected_rows(args.sample_count)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    commands = []
    for route in ROUTES:
        route_payload: dict[str, Any] = {}
        cold_payloads, cold_walls = [], []
        for index, row in enumerate(selected):
            output = work / f"{args.family}_{route}_cold_{index:02d}.json"
            command = worker_command(args, route=route, state="process_cold", output=output, sample_id=str(row["sample_id"]))
            commands.append(" ".join(command))
            payload, wall, _ = run_process(command, fvm=route == "fvm")
            payload["external_fresh_process_wall_seconds"] = wall
            cold_payloads.append(payload); cold_walls.append(wall)
            print(f"[qualification] {args.family} {route} cold {index+1}/{len(selected)} wall={wall:.3f}s", flush=True)
        route_payload["process_cold"] = {
            "fresh_process_count": len(cold_payloads),
            "external_process_wall_seconds": distribution(cold_walls),
            "stage_timing": {
                key: distribution([float(payload["measurements"][0][key]) for payload in cold_payloads])
                for key in sorted(cold_payloads[0]["measurements"][0]) if key.endswith("_seconds")
            },
            "peak_ram_bytes": max(int(payload["process_peak_ram_bytes"]) for payload in cold_payloads),
            "peak_device_bytes": max(int(payload["device_memory"]["peak_bytes_in_use"]) for payload in cold_payloads),
            "sample_ids": [payload["sample_ids"][0] for payload in cold_payloads],
        }
        for state in ("jit_cached_new_topology", "known_topology_new_physics", "fully_cached"):
            if (route == "fvm" and state == "jit_cached_new_topology") or (
                route != "fvm" and args.family == "randomblock" and state == "jit_cached_new_topology"
            ) or (
                route != "fvm" and args.family == "p1i" and state == "known_topology_new_physics"
            ):
                route_payload[state] = {
                    "status": "not_applicable_under_frozen_numerical_contract",
                    "reason": (
                        "FVM has no JIT cache state"
                        if route == "fvm"
                        else "unseen random-block topology changes raw edge shapes and fixed padding failed equivalence"
                        if args.family == "randomblock"
                        else "P1i valid cases do not provide preregistered same-support/new-physics pairs"
                    ),
                }
                continue
            output = work / f"{args.family}_{route}_{state}.json"
            command = worker_command(args, route=route, state=state, output=output)
            commands.append(" ".join(command))
            payload, wall, _ = run_process(command, fvm=route == "fvm")
            payload["external_process_wall_seconds"] = wall
            route_payload[state] = payload
            print(f"[qualification] {args.family} {route} {state} wall={wall:.3f}s", flush=True)
        results[route] = route_payload
    payload = {
        "schema_version": "heat3d_v6_inference_qualification_family_v1",
        "status": "passed", "family": args.family,
        "sample_count": len(selected), "sample_ids": [str(row["sample_id"]) for row in selected],
        "routes": results, "commands": commands,
        "contract": {
            "independent_process_per_route_state": True, "cold_fresh_process_per_sample": True,
            "continuous_wall_clock_not_stage_sum": True, "minimum_measurements": 20,
            "batch_size": 1, "fixed_threads": 1, "production_excludes_oracle": True,
            "cold_cutoff": "prediction_serialization_completed_monotonic_timestamp",
            "hash_metrics_oracle_json_checker_outside_production_timing": True,
        },
        "dataset": {"id": data.manifest["dataset_id"], "manifest_sha256": sha256(args.manifest), "full_fields_sha256": sha256(args.full_fields)},
        "checkpoint": {"sha256": args.checkpoint_sha256, "epoch": args.checkpoint_epoch},
        "fixed_edge_jit_targets": {"path": str(args.edge_targets), "sha256": sha256(args.edge_targets)},
        "accessed_roles": ["valid_iid", "train_frozen_normalization_metadata"],
        "test_accessed": False, "sealed_accessed": False,
        "training_executed": False, "checkpoint_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "family": args.family, "routes": len(results), "samples": len(selected)}, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prepare-edge-targets", action="store_true")
    parser.add_argument("--edge-contract-mode", choices=("fixed_dummy_edge_padding_v1", "raw_shape_family_v1"), default="fixed_dummy_edge_padding_v1")
    parser.add_argument("--family", choices=("p1i", "randomblock"), required=True)
    parser.add_argument("--route", choices=ROUTES)
    parser.add_argument("--state", choices=STATES)
    parser.add_argument("--sample-id")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--randomblock-config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--checkpoint-sha-preverified", action="store_true")
    parser.add_argument("--edge-targets", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/v6_inference_qualification"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker and (args.route is None or args.state is None):
        parser.error("--worker requires --route and --state")
    if not args.prepare_edge_targets and args.edge_targets is None:
        parser.error("formal qualification requires --edge-targets")
    return args


def main() -> int:
    args = parse_args()
    if args.prepare_edge_targets:
        return prepare_edge_targets(args)
    return worker(args) if args.worker else orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
