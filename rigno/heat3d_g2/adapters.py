"""Optional G2-A model adapters with no V7 production-path dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .inputs import P1IInputBatch, unit_cube_latent_queries


@dataclass
class GINOAdapter:
    """Adapter for the upstream NeuralOperator GINO API."""

    model: Any
    latent_queries: Any
    device: str = "cpu"
    upstream_identity: str = "neuraloperator.neuralop.models.GINO"

    def predict(self, inputs: P1IInputBatch) -> Any:
        if inputs.batch_size != 1:
            raise ValueError(
                "GINO qualification uses batch_size=1 because the upstream API "
                "requires shared geometry across a batch"
            )
        coords, features = inputs.to_torch(device=self.device)
        import torch

        latent = torch.as_tensor(self.latent_queries, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = self.model(
                input_geom=coords,
                latent_queries=latent,
                output_queries=coords,
                x=features,
            )
        if isinstance(output, dict):
            raise ValueError("GINO adapter expects a tensor output, not a query mapping")
        return output


def build_gino_model(
    *,
    feature_dim: int = 11,
    output_dim: int = 1,
    latent_resolution: int = 4,
    device: str = "cpu",
) -> GINOAdapter:
    """Instantiate a small explicit GINO smoke model from upstream NeuralOperator."""

    try:
        from neuralop.models import GINO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "GINO adapter requires the pinned NeuralOperator checkout installed "
            "or exposed through PYTHONPATH"
        ) from exc
    model = GINO(
        in_channels=int(feature_dim),
        out_channels=int(output_dim),
        gno_coord_dim=3,
        in_gno_radius=0.6,
        out_gno_radius=0.6,
        fno_in_channels=int(feature_dim),
        fno_n_modes=(2, 2, 2),
        fno_hidden_channels=8,
        fno_lifting_channel_ratio=1,
        fno_n_layers=1,
        in_gno_channel_mlp_hidden_layers=[8],
        out_gno_channel_mlp_hidden_layers=[8],
        gno_embed_channels=4,
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
    ).to(device)
    model.eval()
    return GINOAdapter(
        model=model,
        latent_queries=unit_cube_latent_queries(latent_resolution),
        device=device,
    )


@dataclass
class TransolverAdapter:
    """Adapter for the upstream irregular-mesh Transolver API."""

    model: Any
    device: str = "cpu"
    upstream_identity: str = "Transolver_Irregular_Mesh.Model"

    def predict(self, inputs: P1IInputBatch) -> Any:
        coords, features = inputs.to_torch(device=self.device)
        with _inference_mode():
            output = self.model(coords, features)
        if isinstance(output, dict):
            raise ValueError("Transolver adapter expects a tensor output")
        return output


def build_transolver_model(
    *,
    upstream_root: str | Path,
    feature_dim: int = 11,
    output_dim: int = 1,
    device: str = "cpu",
) -> TransolverAdapter:
    """Load the pinned official irregular-mesh implementation without sys.path edits."""

    model_class = _load_official_transolver_model(Path(upstream_root))
    model = model_class(
        space_dim=3,
        n_layers=2,
        n_hidden=16,
        dropout=0.0,
        n_head=4,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=int(feature_dim),
        out_dim=int(output_dim),
        slice_num=4,
        ref=4,
        unified_pos=False,
    ).to(device)
    model.eval()
    return TransolverAdapter(model=model, device=device)


class _inference_mode:
    def __enter__(self):
        import torch

        self._context = torch.no_grad()
        return self._context.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self._context.__exit__(exc_type, exc_value, traceback)


def _load_official_transolver_model(upstream_root: Path) -> type:
    """Load official source in an isolated in-memory namespace.

    The upstream file uses ``from model...`` imports because it is launched
    from its benchmark directory.  Rewriting those two imports in memory lets
    the adapter avoid changing ``sys.path`` or copying/modifying upstream code.
    """

    model_dir = upstream_root / "PDE-Solving-StandardBenchmark" / "model"
    embedding_path = model_dir / "Embedding.py"
    attention_path = model_dir / "Physics_Attention.py"
    model_path = model_dir / "Transolver_Irregular_Mesh.py"
    for path in (embedding_path, attention_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"official Transolver source is missing: {path}")

    embedding = _exec_module(embedding_path, "v7_g2_transolver_embedding")
    attention = _exec_module(attention_path, "v7_g2_transolver_attention")
    source = model_path.read_text(encoding="utf-8")
    source = source.replace(
        "from model.Embedding import timestep_embedding",
        "timestep_embedding = _embedding.timestep_embedding",
    )
    source = source.replace(
        "from model.Physics_Attention import Physics_Attention_Irregular_Mesh",
        "Physics_Attention_Irregular_Mesh = _attention.Physics_Attention_Irregular_Mesh",
    )
    module = ModuleType("v7_g2_transolver_official_model")
    module.__file__ = str(model_path)
    module.__dict__.update({"_embedding": embedding, "_attention": attention})
    exec(compile(source, str(model_path), "exec"), module.__dict__)
    model_class = module.__dict__.get("Model")
    if not isinstance(model_class, type):
        raise TypeError("official Transolver source did not expose Model")
    return model_class


def _exec_module(path: Path, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def prediction_sha256(output: Any) -> str:
    """Hash a detached prediction for a compatibility receipt."""

    import hashlib

    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(output))
    return hashlib.sha256(array.tobytes()).hexdigest()

