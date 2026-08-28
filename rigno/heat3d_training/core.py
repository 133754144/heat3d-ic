"""Explicit, script-independent training lifecycle for V7 readiness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import optax


@dataclass(frozen=True)
class TrainingBatch:
    """A prepared batch with all model-visible arrays and no hidden loaders."""

    batch_id: str
    sample_ids: tuple[str, ...]
    groups: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class TrainingState:
    """Minimal optimizer state owned by the V7 trainer."""

    params: Any
    optimizer_state: Any
    step: int = 0


@dataclass(frozen=True)
class StepResult:
    """Numerical evidence returned by one training update."""

    state: TrainingState
    loss: Any
    gradients: Any
    updates: Any
    prediction: Any


@dataclass(frozen=True)
class TrainingDependencies:
    """All training dependencies are explicit constructor inputs.

    ``data_source``, ``feature_transform``, ``normalization``, and
    ``graph_builder`` are retained as provenance objects even though prepared
    batches are passed to the numerical core.  This makes the boundary
    inspectable and prevents the trainer from reaching into a script-global
    loader or patching a module at runtime.
    """

    data_source: Any
    feature_transform: Any
    normalization: Any
    graph_builder: Any
    model: Any
    model_apply: Callable[[Any, TrainingBatch, Any], Any]
    loss_fn: Callable[[Any, TrainingBatch], Any]
    optimizer: Any
    batch_iterator: Callable[[Any], Any]
    validation_fn: Callable[[Any, Any], Any]
    checkpoint_writer: Callable[[Path, Mapping[str, Any]], None]
    metrics_fn: Callable[[Any, Any], Mapping[str, Any]]
    gradient_transform: Callable[[Any], Any] | None = None


class ManualGradientDescent:
    """Optimizer with the exact update ``params - learning_rate * grad``."""

    def __init__(self, learning_rate: float) -> None:
        if not float(learning_rate) > 0.0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = float(learning_rate)

    def init(self, params: Any) -> dict[str, Any]:
        return {}

    def update(self, gradients: Any, state: Mapping[str, Any], params: Any = None):
        del params
        updates = tree.tree_map(
            lambda gradient: -self.learning_rate * gradient,
            gradients,
        )
        return updates, dict(state)


def make_optimizer(name: str, *, learning_rate: float, weight_decay: float = 0.0):
    """Create an explicitly selected optimizer without changing its semantics."""

    if name == "manual_gd":
        if weight_decay != 0.0:
            raise ValueError("manual_gd does not accept weight_decay")
        return ManualGradientDescent(learning_rate)
    if name == "adam":
        return optax.adam(learning_rate=float(learning_rate))
    if name == "adamw":
        return optax.adamw(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    raise ValueError(f"unsupported V7 optimizer: {name!r}")


class V7FormalTrainer:
    """Stable reference trainer with optional per-batch executable reuse.

    The default is eager execution, which is the semantic oracle for the
    historical small training path.  ``jit_cache=True`` only caches the
    executable for a fixed prepared batch; it does not alter data, graph,
    batching, objective, or optimizer semantics.
    """

    def __init__(self, dependencies: TrainingDependencies, *, jit_cache: bool = False):
        self.dependencies = dependencies
        self.jit_cache = bool(jit_cache)
        self._compiled_steps: dict[str, Callable[..., Any]] = {}
        self.compile_count = 0

    def initialize(self, params: Any) -> TrainingState:
        return TrainingState(
            params=params,
            optimizer_state=self.dependencies.optimizer.init(params),
            step=0,
        )

    def _step_impl(
        self,
        params: Any,
        optimizer_state: Any,
        batch: TrainingBatch,
        rng: Any,
    ) -> tuple[Any, ...]:
        def loss_with_aux(current_params: Any):
            prediction = self.dependencies.model_apply(current_params, batch, rng)
            loss = self.dependencies.loss_fn(prediction, batch)
            return loss, prediction

        (loss, prediction), gradients = jax.value_and_grad(
            loss_with_aux,
            has_aux=True,
        )(params)
        if self.dependencies.gradient_transform is not None:
            gradients = self.dependencies.gradient_transform(gradients)
        updates, new_optimizer_state = self.dependencies.optimizer.update(
            gradients,
            optimizer_state,
            params,
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_optimizer_state, loss, gradients, updates, prediction

    def _compiled_for(self, batch: TrainingBatch) -> Callable[..., Any]:
        if batch.batch_id not in self._compiled_steps:
            def compiled(params: Any, optimizer_state: Any, rng: Any):
                return self._step_impl(params, optimizer_state, batch, rng)

            self._compiled_steps[batch.batch_id] = jax.jit(compiled)
            self.compile_count += 1
        return self._compiled_steps[batch.batch_id]

    def step(self, state: TrainingState, batch: TrainingBatch, rng: Any = None) -> StepResult:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        if self.jit_cache:
            operation = self._compiled_for(batch)
        else:
            operation = lambda params, optimizer_state, key: self._step_impl(
                params, optimizer_state, batch, key
            )
        (
            params,
            optimizer_state,
            loss,
            gradients,
            updates,
            prediction,
        ) = operation(state.params, state.optimizer_state, rng)
        return StepResult(
            state=TrainingState(params=params, optimizer_state=optimizer_state, step=state.step + 1),
            loss=loss,
            gradients=gradients,
            updates=updates,
            prediction=prediction,
        )

    def validate(self, state: TrainingState, batch: TrainingBatch) -> Any:
        return self.dependencies.validation_fn(state.params, batch)

    def write_checkpoint(self, path: Path, state: TrainingState, metadata: Mapping[str, Any]) -> None:
        payload = dict(metadata)
        payload.update(
            {
                "params": state.params,
                "optimizer_state": state.optimizer_state,
                "step": int(state.step),
            }
        )
        self.dependencies.checkpoint_writer(path, payload)
