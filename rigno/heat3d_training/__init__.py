"""Library-level V7 formal training primitives.

The package is deliberately independent of the historical ``scripts`` tree.
It provides explicit dependency injection for data preparation, model/loss,
optimization, validation, checkpointing, and diagnostics.  The V7 command
entrypoint assembles these pieces for a readiness fixture only.
"""

from .core import (
    ManualGradientDescent,
    StepResult,
    TrainingBatch,
    TrainingDependencies,
    TrainingState,
    V7FormalTrainer,
    make_optimizer,
)
from .prepare import (
    build_v1_training_batches,
    build_v1_training_stats,
    load_selected_v1_examples,
)

__all__ = [
    "ManualGradientDescent",
    "StepResult",
    "TrainingBatch",
    "TrainingDependencies",
    "TrainingState",
    "V7FormalTrainer",
    "make_optimizer",
    "build_v1_training_batches",
    "build_v1_training_stats",
    "load_selected_v1_examples",
]
