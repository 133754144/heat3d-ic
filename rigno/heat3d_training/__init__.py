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
from .p1i import (
    P1I_BATCH_CONTRACT,
    P1I_DATASET_ID,
    P1IPreparedData,
    atomic_training_checkpoint,
    build_p1i_batches,
    load_selected_p1i_examples,
    loss_fn_full,
    make_gradient_transform,
    make_p1i_optimizer,
    model_apply_full,
    model_init_full,
    prepare_p1i_data,
    tree_max_abs_difference,
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
    "P1I_BATCH_CONTRACT",
    "P1I_DATASET_ID",
    "P1IPreparedData",
    "atomic_training_checkpoint",
    "build_p1i_batches",
    "load_selected_p1i_examples",
    "loss_fn_full",
    "make_gradient_transform",
    "make_p1i_optimizer",
    "model_apply_full",
    "model_init_full",
    "prepare_p1i_data",
    "tree_max_abs_difference",
]
