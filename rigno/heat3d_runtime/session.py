"""Explicit V7 V6/P1i inference session."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from rigno.heat3d_runtime.checkpoint import (
    CheckpointBundle,
    device_params,
    load_checkpoint,
    load_run_config,
)
from rigno.heat3d_runtime.features import FeatureTransform
from rigno.heat3d_runtime.grouping import GroupBuilder
from rigno.heat3d_runtime.preflight import validate_semantic_contract
from rigno.models.rigno import RIGNO


MODEL_GROUP_KEYS = (
    "inputs",
    "graphs",
    "global_context",
    "native_physics",
    "qk_region_features",
    "scale_context",
    "scale_region_source_weights",
    "scale_region_volume_weights",
)


@dataclass
class RuntimeSession:
    """Loaded checkpoint + explicit preprocessing + model-apply state."""

    checkpoint: CheckpointBundle
    run_config: dict[str, Any]
    model_config: dict[str, Any]
    graph_config: dict[str, Any]
    feature_transform: FeatureTransform
    group_builder: GroupBuilder
    model: RIGNO
    params: Any
    execution_role: str
    semantic_contract: dict[str, Any]

    @classmethod
    def from_paths(
        cls,
        checkpoint_path: str | Path,
        run_config_path: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_epoch: int | None = None,
        execution_role: str = "production_inference",
        route_contract: Mapping[str, Any] | None = None,
    ) -> "RuntimeSession":
        run_config = load_run_config(run_config_path)
        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_sha256=expected_sha256,
            expected_epoch=expected_epoch,
            strict_semantic_contract=True,
        )
        semantic_contract = validate_semantic_contract(
            run_config=run_config,
            model_config=checkpoint.model_config,
            stats=checkpoint.stats,
            execution_role=execution_role,
            route_contract=route_contract,
        )
        graph_config = dict(run_config["graph_config"])
        if not graph_config:
            raise ValueError("resolved run_config is missing graph_config")
        model_config = dict(checkpoint.model_config)
        feature_transform = FeatureTransform(checkpoint.stats)
        graph_seed = int(run_config["graph_seed"])
        group_builder = GroupBuilder(
            feature_transform=feature_transform,
            graph_config=graph_config,
            graph_seed=graph_seed,
        )
        model = RIGNO(**model_config)
        return cls(
            checkpoint=checkpoint,
            run_config=run_config,
            model_config=model_config,
            graph_config=graph_config,
            feature_transform=feature_transform,
            group_builder=group_builder,
            model=model,
            params=device_params(checkpoint.params),
            execution_role=execution_role,
            semantic_contract=semantic_contract,
        )

    @classmethod
    def from_checkpoint_and_config(
        cls,
        checkpoint_path: str | Path,
        run_config: Mapping[str, Any],
        *,
        expected_sha256: str | None = None,
        expected_epoch: int | None = None,
        execution_role: str = "production_inference",
        route_contract: Mapping[str, Any] | None = None,
    ) -> "RuntimeSession":
        """Construct a session from an in-memory resolved run config."""

        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_sha256=expected_sha256,
            expected_epoch=expected_epoch,
            strict_semantic_contract=True,
        )
        config = dict(run_config)
        semantic_contract = validate_semantic_contract(
            run_config=config,
            model_config=checkpoint.model_config,
            stats=checkpoint.stats,
            execution_role=execution_role,
            route_contract=route_contract,
        )
        graph_config = dict(config["graph_config"])
        if not graph_config:
            raise ValueError("resolved run_config is missing graph_config")
        feature_transform = FeatureTransform(checkpoint.stats)
        model_config = dict(checkpoint.model_config)
        return cls(
            checkpoint=checkpoint,
            run_config=config,
            model_config=model_config,
            graph_config=graph_config,
            feature_transform=feature_transform,
            group_builder=GroupBuilder(
                feature_transform=feature_transform,
                graph_config=graph_config,
                graph_seed=int(config["graph_seed"]),
            ),
            model=RIGNO(**model_config),
            params=device_params(checkpoint.params),
            execution_role=execution_role,
            semantic_contract=semantic_contract,
        )

    def build_group(
        self,
        examples: list[Any],
        *,
        name: str = "v7_reference_valid_iid",
        context_examples: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Build one model-input group and attach all configured raw-input features."""

        group = self.group_builder.build(examples, name=name)
        return self._attach_features(group, examples, context_examples=context_examples)

    def build_group_from_metadata(
        self,
        examples: list[Any],
        metadata: Any,
        *,
        name: str = "v7_reference_valid_iid",
        edge_targets: dict[str, int | None] | None = None,
        context_examples: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Build a group from stable graph metadata, including fixed edge padding."""

        group = self.group_builder.build_from_metadata(
            examples,
            metadata,
            name=name,
            edge_targets=edge_targets,
        )
        return self._attach_features(group, examples, context_examples=context_examples)

    def _attach_features(
        self,
        group: dict[str, Any],
        examples: list[Any],
        *,
        context_examples: list[Any] | None,
    ) -> dict[str, Any]:
        if context_examples is None:
            context_examples = examples
        if len(context_examples) != len(examples):
            raise ValueError("context_examples must align one-to-one with examples")
        model_config = self.model_config
        context_enabled = (
            model_config["native_output_mode"] == "native_shape_scale"
            or model_config["global_context_mode"] != "none"
        )
        context_payload = self.run_config["global_context"]
        standardizer = context_payload["standardizer"] if context_enabled else None
        if context_enabled:
            if not isinstance(standardizer, Mapping):
                raise ValueError("native/global-context model requires a frozen standardizer")
            context = self.feature_transform.standardize_global_contexts(
                context_examples, standardizer
            )
            expected = int(model_config["global_context_feature_dim"])
            if context.shape != (len(examples), expected):
                raise ValueError(
                    f"global context shape mismatch: got={context.shape} "
                    f"expected={(len(examples), expected)}"
                )
            group["global_context"] = jnp.asarray(context, dtype=jnp.float32)

        if model_config["native_output_mode"] == "native_shape_scale":
            physics = [self.feature_transform.native_physics(example) for example in examples]
            group["native_physics"] = {
                key: jnp.stack([row[key] for row in physics], axis=0)
                for key in physics[0]
            }

        if (
            model_config["scale_pooling"] == "qk_gated"
            or model_config["shape_attention_mode"] != "none"
            or model_config["scale_attention_mode"] != "none"
        ):
            p2r = np.asarray(group["metadata"].p2r_edge_indices)
            rnode_count = int(np.asarray(group["metadata"].x_rnodes).shape[1] - 1)
            rows = [
                self.feature_transform.qk_region_features(
                    example,
                    p2r[row],
                    rnode_count,
                    feature_version=str(model_config["qk_region_feature_version"]),
                )
                for row, example in enumerate(examples)
            ]
            group["qk_region_features"] = jnp.asarray(np.stack(rows), dtype=jnp.float32)

        if model_config["scale_deepsets_mode"] != "none":
            p2r = np.asarray(group["metadata"].p2r_edge_indices)
            rnode_count = int(np.asarray(group["metadata"].x_rnodes).shape[1] - 1)
            weights = [
                self.feature_transform.scale_region_weights(example, p2r[row], rnode_count)
                for row, example in enumerate(examples)
            ]
            group["scale_region_source_weights"] = jnp.asarray(
                np.stack([row[0] for row in weights]), dtype=jnp.float32
            )
            group["scale_region_volume_weights"] = jnp.asarray(
                np.stack([row[1] for row in weights]), dtype=jnp.float32
            )

        if model_config["scale_context_mode"] != "none":
            scale_standardizer = context_payload["scale_standardizer"]
            if not isinstance(scale_standardizer, Mapping):
                raise ValueError("configured scale context requires a frozen standardizer")
            group["scale_context"] = jnp.asarray(
                self.feature_transform.standardize_scale_contexts(examples, scale_standardizer),
                dtype=jnp.float32,
            )
        return group

    def apply(self, group: Mapping[str, Any], *, key: Any = None) -> dict[str, Any]:
        """Apply the loaded RIGNO model using explicit runtime inputs."""

        selected = {name: group[name] for name in MODEL_GROUP_KEYS if name in group}
        if "native_physics" in selected:
            physics = selected["native_physics"]
            output = self.model.apply(
                {"params": self.params},
                inputs=selected["inputs"],
                graphs=selected["graphs"],
                global_context=selected.get("global_context"),
                control_volumes=physics["control_volumes"],
                log_s_phys=physics["log_s_phys"],
                reference_temperature=physics["reference_temperature"],
                dirichlet_mask=physics["dirichlet_mask"],
                prescribed_temperature=physics["prescribed_temperature"],
                qk_region_features=selected.get("qk_region_features"),
                scale_context=selected.get("scale_context"),
                scale_region_source_weights=selected.get("scale_region_source_weights"),
                scale_region_volume_weights=selected.get("scale_region_volume_weights"),
                key=key,
                method=self.model.predict_native_shape_scale,
            )
        else:
            output = self.model.apply(
                {"params": self.params},
                inputs=selected["inputs"],
                graphs=selected["graphs"],
                global_context=selected.get("global_context"),
                key=key,
            )
        return output

    def predict_native_1024(
        self,
        examples: list[Any],
        *,
        batch_size: int = 32,
    ) -> dict[str, dict[str, Any]]:
        """Run native-1024 inference and return predictions without metrics."""

        if self.model_config["native_output_mode"] != "native_shape_scale":
            raise ValueError("native-1024 reference inference requires native_shape_scale")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        results: dict[str, dict[str, Any]] = {}
        for start in range(0, len(examples), int(batch_size)):
            batch = examples[start : start + int(batch_size)]
            group = self.build_group(batch, name=f"v7_reference_valid_iid_{start:05d}")
            output = self.apply(group)
            jax.block_until_ready(output["raw_temperature"])
            raw = np.asarray(output["raw_temperature"], dtype=np.float64)
            delta = np.asarray(output["deltaT_hat"], dtype=np.float64)
            scale = np.asarray(output["s_hat"], dtype=np.float64).reshape(-1)
            if raw.shape[0] != len(batch) or delta.shape[0] != len(batch):
                raise ValueError("native prediction batch dimension drifted")
            if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(scale)):
                raise ValueError("native prediction contains non-finite values")
            for row, example in enumerate(batch):
                results[str(example.sample_id)] = {
                    "raw_temperature": raw[row, 0, :, 0],
                    "deltaT_hat": delta[row, 0, :, 0],
                    "s_hat": float(scale[row]),
                }
        return results

    def descriptor(self) -> dict[str, Any]:
        """Return provenance needed by an old/new equivalence record."""

        return {
            "runtime": "rigno.heat3d_runtime",
            "runtime_api": "RuntimeSession",
            "execution_role": self.execution_role,
            "checkpoint": self.checkpoint.descriptor(),
            "run_config": {
                "graph_config": self.graph_config,
                "graph_seed": self.run_config["graph_seed"],
            },
            "model_config": self.model_config,
            "semantic_contract": self.semantic_contract,
        }


def main() -> None:  # pragma: no cover - CLI wrapper lives in scripts.
    raise RuntimeError("use scripts.run_heat3d_v7_reference_inference")
