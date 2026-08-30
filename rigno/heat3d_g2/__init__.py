"""G2-A adapters for external neural-operator baselines.

The package is intentionally separate from :mod:`rigno.heat3d_runtime` and
the formal V7 training path.  It owns only input adaptation and optional
upstream model loading for compatibility smoke/qualification.  It never
selects a checkpoint, reads labels, runs a solver, or changes the frozen V7
evaluation definitions.
"""

from .adapters import (
    GINOAdapter,
    TransolverAdapter,
    build_gino_model,
    build_transolver_model,
)
from .inputs import P1IInputBatch, P1I_FEATURE_NAMES, unit_cube_latent_queries
from .p1i import (
    evaluate_valid_prediction,
    load_frozen_p1i_input_only,
    load_frozen_valid_evaluation_sample,
)

__all__ = [
    "GINOAdapter",
    "TransolverAdapter",
    "build_gino_model",
    "build_transolver_model",
    "P1IInputBatch",
    "P1I_FEATURE_NAMES",
    "unit_cube_latent_queries",
    "evaluate_valid_prediction",
    "load_frozen_p1i_input_only",
    "load_frozen_valid_evaluation_sample",
]
