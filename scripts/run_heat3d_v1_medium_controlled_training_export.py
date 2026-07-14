"""Controlled Heat3D v1 medium training export smoke.

This runner reuses the existing v1 train/valid smoke path and writes recovered
temperature predictions to an ignored output directory for downstream
diagnostic comparison. It is not a formal training experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import pickle
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_heat3d_v1_small_train_valid_smoke import (  # noqa: E402
    MODEL_CONFIG,
    _global_norm,
    _metadata_shape_signature,
    _metrics,
    _sample_root,
    _selected_steps,
    _subset_split_ids,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_normalization import (  # noqa: E402
    legacy_train_only_stats as _train_only_stats,
    normalize_condition,
    normalize_coords as _normalize_coords,
    normalize_target_delta,
    normalize_target_delta,
    normalized_delta_to_raw as _normalized_delta_to_raw,
    recover_raw_condition,
    recover_temperature_from_normalized_delta,
)
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.heat3d_v1_training_semantics import (  # noqa: E402
    COORD_POLICY_SAMPLE_LOCAL_ISOTROPIC,
    build_legacy_zero_delta_bridge as _bridge_for,
    decoder_bypass_required_full_condition_features,
)
from rigno.heat3d_v5_global_context import (  # noqa: E402
    GLOBAL_CONTEXT_FEATURES,
    fit_train_only_standardizer,
    global_context_from_raw_condition,
    standardize_contexts,
    validate_global_context_schema,
)
from rigno.heat3d_v5_metrics import control_volume_weights  # noqa: E402
from rigno.heat3d_v5_shape_scale import (  # noqa: E402
    mask_branch_gradients,
    native_gradient_group_norms,
    native_shape_scale_diagnostics,
    native_shape_scale_losses,
)
from rigno.heat3d_v4_split_map import (  # noqa: E402
    load_sample_split_map,
    split_ids_from_sample_splits,
)
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402
from rigno.models.operator import Inputs  # noqa: E402


RUNNER_MODEL_CONFIG = {
    **MODEL_CONFIG,
    "node_latent_size": 96,
    "edge_latent_size": 96,
    "processor_steps": 6,
    "mlp_hidden_layers": 2,
}

# The V4 wrapper enables this for compact logs when no stress split is defined.
HIDE_MISSING_STRESS_COMPACT_LOG = False
DECODER_BYPASS_MODE_NONE = "none"
DECODER_BYPASS_MODE_POST_DECODER_RESIDUAL = "post_decoder_residual"
DECODER_BYPASS_MODES = (
    DECODER_BYPASS_MODE_NONE,
    DECODER_BYPASS_MODE_POST_DECODER_RESIDUAL,
)
DECODER_BYPASS_FEATURES_NONE = "none"
DECODER_BYPASS_FEATURES_FULL_CONDITION = "full_condition"
DECODER_BYPASS_FEATURES_EXPLICIT_LOCAL_CONDITION = "explicit_local_condition"
DECODER_BYPASS_FEATURES = (
    DECODER_BYPASS_FEATURES_NONE,
    DECODER_BYPASS_FEATURES_FULL_CONDITION,
    DECODER_BYPASS_FEATURES_EXPLICIT_LOCAL_CONDITION,
)
DECODER_BYPASS_FEATURE_SOURCE_NORMALIZED_C = "normalized_c"
DECODER_BYPASS_FEATURE_SOURCES = (DECODER_BYPASS_FEATURE_SOURCE_NORMALIZED_C,)
DECODER_BYPASS_INIT_ZERO_RESIDUAL = "zero_residual"
DECODER_BYPASS_INITS = (DECODER_BYPASS_INIT_ZERO_RESIDUAL,)
DECODER_BYPASS_OUTPUT_SPACES = ("normalized_deltaT", "native_psi")
DECODER_BYPASS_REQUIRED_FULL_CONDITION_FEATURES = (
    "k_x",
    "k_y",
    "k_z",
    "q",
    "is_top",
    "is_bottom",
    "is_side",
    "is_interior",
    "top_h",
    "top_T_inf_minus_T_ref",
    "bottom_T_fixed_minus_T_ref",
)
DECODER_BYPASS_LOCAL_FEATURE_ALLOWLIST = (
    "k_x",
    "k_y",
    "k_z",
    "q",
    "is_top",
    "is_bottom",
    "is_side",
    "is_interior",
)
GLOBAL_CONTEXT_MODE_NONE = "none"
GLOBAL_CONTEXT_MODE_FILM = "film"
GLOBAL_CONTEXT_MODES = (GLOBAL_CONTEXT_MODE_NONE, GLOBAL_CONTEXT_MODE_FILM)
FILM_TARGET_RNODES_PROCESSED = "rnodes_processed"
FILM_TARGETS = (FILM_TARGET_RNODES_PROCESSED,)
FILM_INIT_IDENTITY = "identity"
FILM_INITS = (FILM_INIT_IDENTITY,)
NATIVE_OUTPUT_MODES = ("legacy_normalized_deltaT", "native_shape_scale")
NATIVE_BRANCH_MODES = ("scale_only", "shape_only", "joint")
SCALE_HEAD_MODES = ("physics_only", "physics_plus_pooled_latent")
SCALE_POOLING_MODES = ("mean",)
DEFAULT_SUBSET = (
    REPO_DIR
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2"
)
DEFAULT_SPLIT_MAP = (
    REPO_DIR
    / "configs"
    / "heat3d_v2"
    / "medium1024_gapA_stratified_split_seed0.json"
)
DEFAULT_OUTPUT_DIR = REPO_DIR / "output" / "heat3d_v1_medium_runs" / "export_smoke_seed0"
DEFAULT_FINAL_PROBE_SUBSET = (
    REPO_DIR
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v3_final_target_probe_v0"
)
DEFAULT_FINAL_PROBE_PROVENANCE = (
    REPO_DIR
    / "output"
    / "heat3d_v3_final_target_probe_s5_smoke"
    / "metadata"
    / "probe_data_provenance.json"
)
TRAIN_METRICS_SCHEDULE_CHOICES = ("every_epoch", "half_and_final", "final_only", "none")
RADIUS_POLICY_CHOICES = ("legacy_kdtree_mean4", "discrete_physical_coverage")
COVERAGE_REPAIR_POLICY_CHOICES = ("none", "nearest_rnode")
NODE_COORDINATE_ENCODING_CHOICES = ("raw", "raw_plus_fourier")
INIT_MODE_CHOICES = ("real_first_batch", "upstream_dummy")
PARTIAL_LOAD_POLICY_CHOICES = ("matching", "skip_decoder", "encoder_processor_only")
FINAL_PROBE_CHECKPOINT_KIND_CHOICES = ("best", "final", "both")
SAMPLE_WEIGHT_POLICY_CHOICES = ("none", "hard_sample_list")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled training export smoke for Heat3D v1 medium labels. "
            "Writes ignored predictions for diagnostic comparison only."
        )
    )
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument(
        "--split-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON sample_id-to-split map. For the current Heat3D v2 "
            "medium1024 Gap-A subset, omitted values default to the stratified "
            "split map; train uses split=train, primary validation uses "
            "valid_iid, and valid_stress is reported as diagnostics only."
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=(
            "constant",
            "warmup_cosine",
            "rapid_decay",
            "two_stage",
            "second_stage",
            "upstream_onecycle",
        ),
        default="warmup_cosine",
    )
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--second-stage-epoch", type=int, default=0)
    parser.add_argument("--second-stage-lr", type=float, default=1e-4)
    parser.add_argument("--lr-init", type=float, default=1e-5)
    parser.add_argument("--lr-peak", type=float, default=2e-4)
    parser.add_argument("--lr-base", type=float, default=1e-5)
    parser.add_argument("--lr-lowr", type=float, default=1e-6)
    parser.add_argument("--pct-start", type=float, default=0.02)
    parser.add_argument("--pct-final", type=float, default=0.10)
    parser.add_argument("--optimizer", choices=("manual_gd", "adam", "adamw"), default="adamw")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--node-latent-size", type=int, default=RUNNER_MODEL_CONFIG["node_latent_size"])
    parser.add_argument("--edge-latent-size", type=int, default=RUNNER_MODEL_CONFIG["edge_latent_size"])
    parser.add_argument("--processor-steps", type=int, default=RUNNER_MODEL_CONFIG["processor_steps"])
    parser.add_argument("--mlp-hidden-layers", type=int, default=RUNNER_MODEL_CONFIG["mlp_hidden_layers"])
    parser.add_argument("--p-edge-masking", type=float, default=float(RUNNER_MODEL_CONFIG.get("p_edge_masking", 0.0)))
    parser.add_argument("--decoder-bypass-mode", choices=DECODER_BYPASS_MODES, default=DECODER_BYPASS_MODE_NONE)
    parser.add_argument("--decoder-bypass-features", choices=DECODER_BYPASS_FEATURES, default=DECODER_BYPASS_FEATURES_NONE)
    parser.add_argument(
        "--decoder-bypass-feature-source",
        choices=DECODER_BYPASS_FEATURE_SOURCES,
        default=DECODER_BYPASS_FEATURE_SOURCE_NORMALIZED_C,
    )
    parser.add_argument(
        "--decoder-bypass-local-feature-names",
        type=str,
        default="",
        help=(
            "Comma-separated, node-varying condition feature names for "
            "--decoder-bypass-features explicit_local_condition. The V5 "
            "allowlist excludes sample-global BC/extent broadcasts."
        ),
    )
    parser.add_argument("--decoder-bypass-hidden-size", type=int, default=64)
    parser.add_argument("--decoder-bypass-layers", type=int, default=2)
    parser.add_argument("--decoder-bypass-init", choices=DECODER_BYPASS_INITS, default=DECODER_BYPASS_INIT_ZERO_RESIDUAL)
    parser.add_argument("--decoder-bypass-residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--decoder-bypass-output-space",
        choices=DECODER_BYPASS_OUTPUT_SPACES,
        default="normalized_deltaT",
    )
    parser.add_argument(
        "--global-context-mode",
        choices=GLOBAL_CONTEXT_MODES,
        default=GLOBAL_CONTEXT_MODE_NONE,
        help="Sample-global inference context mode. none preserves the V4 call path exactly.",
    )
    parser.add_argument(
        "--global-context-feature-names",
        type=str,
        default="",
        help=(
            "Comma-separated V5 Global FiLM schema. film requires the exact "
            "inference-only V5 global physics feature order."
        ),
    )
    parser.add_argument("--film-target", choices=FILM_TARGETS, default=FILM_TARGET_RNODES_PROCESSED)
    parser.add_argument("--film-init", choices=FILM_INITS, default=FILM_INIT_IDENTITY)
    parser.add_argument("--film-hidden-size", type=int, default=64)
    parser.add_argument("--native-output-mode", choices=NATIVE_OUTPUT_MODES, default="legacy_normalized_deltaT")
    parser.add_argument("--native-branch-mode", choices=NATIVE_BRANCH_MODES, default="joint")
    parser.add_argument("--scale-head-mode", choices=SCALE_HEAD_MODES, default="physics_only")
    parser.add_argument("--scale-pooling", choices=SCALE_POOLING_MODES, default="mean")
    parser.add_argument("--scale-head-hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=88)
    parser.add_argument("--validation-batch-size", type=int, default=88)
    parser.add_argument("--prediction-batch-size", type=int, default=88)
    parser.add_argument(
        "--prediction-split",
        choices=("all", "train", "valid_iid", "valid_stress", "test_iid"),
        default="all",
        help="Limit final/best prediction export to one split; training behavior is unchanged.",
    )
    parser.add_argument("--shuffle-train-batches", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument(
        "--init-mode",
        choices=INIT_MODE_CHOICES,
        default="real_first_batch",
        help=(
            "Parameter initialization input policy. real_first_batch preserves "
            "legacy behavior. upstream_dummy initializes on zero-valued dummy "
            "inputs with the first batch graph shape, while training still uses "
            "real batches."
        ),
    )
    parser.add_argument(
        "--batch-plan",
        choices=("current_graph_shape", "sample_shuffle"),
        default="sample_shuffle",
        help="Train batch construction plan. Default is the v4 B88 sample-shuffle path.",
    )
    parser.add_argument(
        "--batch-build-seed",
        type=int,
        default=None,
        help="Optional seed for batch construction plans such as sample_shuffle. Defaults to 0.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Optional model initialization seed. Defaults to --seed for legacy compatibility.",
    )
    parser.add_argument(
        "--batch-order-seed",
        type=int,
        default=None,
        help="Optional train batch shuffle seed. Defaults to 0 when omitted.",
    )
    parser.add_argument(
        "--graph-seed",
        type=int,
        default=None,
        help="Optional graph/rmesh metadata seed. Defaults to 0 when omitted.",
    )
    parser.add_argument(
        "--boundary-mask-fallback",
        dest="boundary_mask_fallback",
        action="store_true",
        default=True,
        help="Reconstruct boundary masks from coordinate min/max when boundary_regions is missing.",
    )
    parser.add_argument(
        "--no-boundary-mask-fallback",
        dest="boundary_mask_fallback",
        action="store_false",
        help="Preserve the legacy all-interior mask behavior when boundary_regions is missing.",
    )
    parser.add_argument(
        "--radius-policy",
        choices=RADIUS_POLICY_CHOICES,
        default="discrete_physical_coverage",
        help="Heat3D graph radius policy. Default is v4 discrete physical-node coverage.",
    )
    parser.add_argument(
        "--coverage-repair-policy",
        choices=COVERAGE_REPAIR_POLICY_CHOICES,
        default="none",
        help="Optional Heat3D graph coverage repair policy. Default disables repair for pure discrete coverage.",
    )
    parser.add_argument("--repair-p2r", dest="repair_p2r", action="store_true", default=True)
    parser.add_argument("--no-repair-p2r", dest="repair_p2r", action="store_false")
    parser.add_argument("--repair-r2p", dest="repair_r2p", action="store_true", default=True)
    parser.add_argument("--no-repair-r2p", dest="repair_r2p", action="store_false")
    parser.add_argument("--min-physical-coverage", type=int, default=1)
    parser.add_argument(
        "--node-coordinate-encoding",
        choices=NODE_COORDINATE_ENCODING_CHOICES,
        default="raw",
        help=(
            "Structural node coordinate encoding. raw_plus_fourier appends "
            "Fourier features to normalized unit-box coordinates without "
            "changing Heat3D periodic=False graph topology."
        ),
    )
    parser.add_argument("--node-coordinate-freqs", type=int, default=4)
    parser.add_argument(
        "--sample-weight-policy",
        choices=SAMPLE_WEIGHT_POLICY_CHOICES,
        default="none",
        help="Optional train-only sample weighting policy. Default preserves unweighted training.",
    )
    parser.add_argument(
        "--sample-weight-json",
        type=Path,
        default=None,
        help="JSON hard-sample list or sample_id-to-weight map for --sample-weight-policy hard_sample_list.",
    )
    parser.add_argument("--sample-weight-default", type=float, default=1.0)
    parser.add_argument("--sample-weight-normalize", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate runner options and print the resolved model/batch settings without data or output writes.",
    )
    parser.add_argument("--save-predictions", dest="save_predictions", action="store_true", default=True)
    parser.add_argument(
        "--no-save-predictions",
        dest="save_predictions",
        action="store_false",
        help="Disable final prediction export. Enabled by default for controlled long runs.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("valid_loss", "valid_raw_deltaT_mse", "valid_base_mse"),
        default="valid_base_mse",
        help="Validation metric used to track the best epoch for optional best prediction export.",
    )
    parser.add_argument("--save-best-predictions", dest="save_best_predictions", action="store_true", default=True)
    parser.add_argument(
        "--no-save-best-predictions",
        dest="save_best_predictions",
        action="store_false",
        help="Disable best prediction export. Enabled by default for controlled long runs.",
    )
    parser.add_argument("--best-predictions-name", type=str, default="best_predictions.npz")
    parser.add_argument(
        "--no-save-final-checkpoint",
        dest="save_final_checkpoint",
        action="store_false",
        default=True,
        help="Disable writing params_final.pkl. Enabled by default for future read-only inference audits.",
    )
    parser.add_argument(
        "--no-save-best-checkpoint",
        dest="save_best_checkpoint",
        action="store_false",
        default=True,
        help="Disable writing params_best.pkl. Enabled by default for future read-only inference audits.",
    )
    parser.add_argument("--final-checkpoint-name", type=str, default="params_final.pkl")
    parser.add_argument("--best-checkpoint-name", type=str, default="params_best.pkl")
    parser.add_argument(
        "--save-point-global-best-checkpoint",
        action="store_true",
        help=(
            "Also track and save the checkpoint with the lowest valid true-RMS "
            "point-global relative RMSE. Disabled by default."
        ),
    )
    parser.add_argument(
        "--point-global-best-checkpoint-name",
        type=str,
        default="params_best_valid_point_global.pkl",
    )
    parser.add_argument(
        "--final-probe-eval-after-training",
        dest="final_probe_eval_after_training",
        action="store_true",
        default=True,
        help=(
            "After training and checkpoint export, run read-only final-target "
            "probe inference using saved params checkpoints. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-final-probe-eval-after-training",
        dest="final_probe_eval_after_training",
        action="store_false",
        help="Disable post-training final-target probe checkpoint inference.",
    )
    parser.add_argument(
        "--final-probe-output-dir",
        type=Path,
        default=None,
        help="Ignored output directory for optional post-training final probe inference.",
    )
    parser.add_argument(
        "--final-probe-checkpoint-kind",
        choices=FINAL_PROBE_CHECKPOINT_KIND_CHOICES,
        default="both",
        help="Checkpoint(s) to evaluate when --final-probe-eval-after-training is enabled.",
    )
    parser.add_argument("--final-probe-subset", type=Path, default=DEFAULT_FINAL_PROBE_SUBSET)
    parser.add_argument("--final-probe-provenance", type=Path, default=DEFAULT_FINAL_PROBE_PROVENANCE)
    parser.add_argument(
        "--final-probe-batch-size",
        type=int,
        default=0,
        help="Batch size passed to final probe checkpoint smoke; 0 means one full prediction batch.",
    )
    parser.add_argument(
        "--post-training-diagnostics",
        dest="post_training_diagnostics",
        action="store_true",
        default=True,
        help=(
            "After prediction export, run read-only diagnostics over saved final/best "
            "predictions. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-post-training-diagnostics",
        dest="post_training_diagnostics",
        action="store_false",
        help="Disable post-training read-only diagnostics over saved predictions.",
    )
    parser.add_argument(
        "--post-training-diagnostics-output-dir",
        type=Path,
        default=None,
        help="Ignored output directory for optional post-training diagnostics.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional params-only checkpoint used to initialize model params before training.",
    )
    parser.add_argument(
        "--checkpoint-load-strict",
        choices=("true", "false"),
        default="true",
        help="When true, --init-checkpoint params must exactly match the initialized param tree.",
    )
    parser.add_argument(
        "--partial-load-policy",
        choices=PARTIAL_LOAD_POLICY_CHOICES,
        default="matching",
        help=(
            "Policy used only with --checkpoint-load-strict false. matching loads all "
            "shape-compatible matching leaves; skip_decoder leaves decoder params "
            "fresh; encoder_processor_only loads only encoder/processor params."
        ),
    )
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument(
        "--train-metrics-schedule",
        choices=TRAIN_METRICS_SCHEDULE_CHOICES,
        default="half_and_final",
        help="Schedule for full train split metrics during the epoch loop.",
    )
    parser.add_argument(
        "--grad-norm-report-every",
        type=int,
        default=10,
        help=(
            "Frequency for external grad norm diagnostics. "
            "Use 1 for every train batch, N>1 for every N batches, or 0 to disable reporting."
        ),
    )
    parser.add_argument("--log-mode", choices=("compact", "full", "quiet"), default="compact")
    parser.add_argument("--progress-log", dest="progress_log", action="store_true", default=True)
    parser.add_argument("--no-progress-log", dest="progress_log", action="store_false")
    parser.add_argument(
        "--progress-detail",
        choices=("off", "quiet", "basic", "verbose", "full"),
        default="basic",
        help=(
            "Startup progress detail. basic uses compact progress updates; "
            "verbose/full prints per-group details; off/quiet disables group-build progress."
        ),
    )
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--profile-timing-json", type=Path, default=None)
    parser.add_argument(
        "--memory-audit-jsonl",
        type=Path,
        default=None,
        help="Optional ignored JSONL path for CPU/GPU memory trace events.",
    )
    parser.add_argument(
        "--memory-audit-every-batch",
        action="store_true",
        help="Record per-train-batch memory audit events when --memory-audit-jsonl is set.",
    )
    parser.add_argument(
        "--memory-audit-gc",
        action="store_true",
        help="Run gc.collect() at epoch boundaries while memory audit is enabled.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=(
            "mse",
            "background_hotspot",
            "background_l1_bias",
            "background_l1_relative",
            "background_pseudo_negative",
            "hotspot_strong_q",
        ),
        default="mse",
    )
    parser.add_argument("--background-quantile", type=float, default=0.50)
    parser.add_argument("--hotspot-quantile", type=float, default=0.90)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--hotspot-weight", type=float, default=0.1)
    parser.add_argument("--strong-q-quantile", type=float, default=0.90)
    parser.add_argument("--strong-q-weight", type=float, default=0.05)
    parser.add_argument("--native-shape-cv-weight", type=float, default=1.0)
    parser.add_argument("--native-log-scale-weight", type=float, default=1.0)
    parser.add_argument("--native-relative-field-weight", type=float, default=1.0)
    parser.add_argument("--native-raw-field-weight", type=float, default=1.0)
    parser.add_argument("--background-l1-weight", type=float, default=1.0)
    parser.add_argument("--background-bias-weight", type=float, default=1.0)
    parser.add_argument("--background-over-weight", type=float, default=1.0)
    parser.add_argument("--background-relative-weight", type=float, default=0.0)
    parser.add_argument("--relative-floor", type=float, default=0.02)
    parser.add_argument("--relative-floor-mode", choices=("fixed", "p50", "p75"), default="fixed")
    parser.add_argument("--pseudo-negative-quantile", type=float, default=0.25)
    parser.add_argument("--pseudo-negative-delta-threshold", type=float, default=None)
    parser.add_argument("--pseudo-negative-weight", type=float, default=0.1)
    parser.add_argument("--pseudo-negative-over-margin", type=float, default=0.0)
    parser.add_argument("--pseudo-negative-min-count", type=int, default=1)
    parser.add_argument(
        "--pseudo-negative-loss-type",
        choices=("mse", "l1", "relative_l1", "relative_mse"),
        default="mse",
    )
    parser.add_argument("--pseudo-negative-relative-floor", type=float, default=0.02)
    parser.add_argument("--loss-weight-schedule", choices=("constant", "two_phase", "linear_anneal"), default="constant")
    parser.add_argument("--loss-transition-epoch", type=int, default=0)
    parser.add_argument("--background-relative-weight-start", type=float, default=None)
    parser.add_argument("--background-relative-weight-end", type=float, default=None)
    parser.add_argument("--hotspot-weight-start", type=float, default=None)
    parser.add_argument("--hotspot-weight-end", type=float, default=None)
    parser.add_argument("--background-l1-weight-start", type=float, default=None)
    parser.add_argument("--background-l1-weight-end", type=float, default=None)
    parser.add_argument("--background-bias-weight-start", type=float, default=None)
    parser.add_argument("--background-bias-weight-end", type=float, default=None)
    parser.add_argument("--background-over-weight-start", type=float, default=None)
    parser.add_argument("--background-over-weight-end", type=float, default=None)
    args = parser.parse_args()
    if args.split_map is None and _is_medium1024_gapA_subset(args.subset):
        args.split_map = DEFAULT_SPLIT_MAP
    return args


def _is_medium1024_gapA_subset(subset: Path) -> bool:
    return "medium1024_gapA_full1024_v2" in str(subset)


def _emit(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _progress_enabled(args: argparse.Namespace) -> bool:
    return bool(args.progress_log) and args.log_mode != "quiet"


def _format_elapsed(start_time: float | None) -> str:
    if start_time is None:
        return ""
    return f" elapsed={time.perf_counter() - start_time:.2f}s"


def _progress(enabled: bool, stage: str, message: str, start_time: float | None = None) -> None:
    if enabled:
        _emit(f"[{stage}] {message}{_format_elapsed(start_time)}")


def _progress_detail_mode(progress_detail: str) -> str:
    if progress_detail in {"off", "quiet"}:
        return "off"
    if progress_detail in {"verbose", "full"}:
        return "full"
    return "basic"


def _progress_detail_enabled(args: argparse.Namespace) -> bool:
    return _progress_enabled(args) and _progress_detail_mode(args.progress_detail) != "off"


def _verbose_progress_enabled(args: argparse.Namespace) -> bool:
    return _progress_enabled(args) and _progress_detail_mode(args.progress_detail) == "full"


def _progress_checkpoints(total: int) -> set[int]:
    if total <= 0:
        return set()
    if total >= 768:
        step = 256
    elif total >= 256:
        step = 128
    elif total >= 64:
        step = 64
    else:
        step = total
    checkpoints = set(range(step, total + 1, step))
    checkpoints.add(total)
    return checkpoints


class _ProgressBar:
    """Small stdout progress helper for startup phases without extra deps."""

    def __init__(
        self,
        enabled: bool,
        label: str,
        total: int,
        *,
        min_interval_s: float = 2.0,
        width: int = 24,
        stream=None,
    ) -> None:
        self.enabled = bool(enabled)
        self.label = label
        self.total = max(0, int(total))
        self.min_interval_s = float(min_interval_s)
        self.width = max(4, int(width))
        self.stream = sys.stdout if stream is None else stream
        self.start_time = time.perf_counter()
        self.last_emit_time = self.start_time
        self.last_current = 0
        self.closed = False
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.percent_step = max(1, math.ceil(max(1, self.total) * 0.05))
        self.next_percent_current = self.percent_step

    def update(self, current: int, *, detail: str | None = None, force: bool = False) -> None:
        if not self.enabled or self.closed:
            return
        current = max(0, int(current))
        if self.total:
            current = min(current, self.total)
        now = time.perf_counter()
        should_emit = force or current >= self.total
        should_emit = should_emit or (now - self.last_emit_time) >= self.min_interval_s
        should_emit = should_emit or current >= self.next_percent_current
        if not should_emit:
            self.last_current = current
            return
        self._write(self._format_line(current, detail=detail), overwrite=self.is_tty and not force)
        self.last_emit_time = now
        self.last_current = current
        while self.next_percent_current <= current:
            self.next_percent_current += self.percent_step

    def close(self, *, current: int | None = None) -> None:
        if not self.enabled or self.closed:
            return
        final_current = self.last_current if current is None else int(current)
        self.update(final_current, force=True)
        if self.is_tty:
            self.stream.write("\n")
        elapsed = time.perf_counter() - self.start_time
        avg = elapsed / final_current if final_current > 0 else 0.0
        self.stream.write(
            f"{self.label} completed groups={final_current} "
            f"elapsed={elapsed:.1f}s avg={avg:.2f}s\n"
        )
        self.stream.flush()
        self.closed = True

    def _format_line(self, current: int, *, detail: str | None) -> str:
        elapsed = time.perf_counter() - self.start_time
        avg = elapsed / current if current > 0 else 0.0
        if self.total > 0 and current > 0:
            eta = max(0.0, avg * (self.total - current))
        else:
            eta = 0.0
        if self.total > 0:
            ratio = min(1.0, max(0.0, current / self.total))
        else:
            ratio = 1.0
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f" {detail}" if detail else ""
        return (
            f"{self.label} [{bar}] {current}/{self.total} "
            f"elapsed={elapsed:.1f}s avg={avg:.2f}s eta={eta:.1f}s{suffix}"
        )

    def _write(self, text: str, *, overwrite: bool) -> None:
        if overwrite:
            self.stream.write("\r" + text)
        else:
            self.stream.write(text + "\n")
        self.stream.flush()


def _record_timing(timings: dict[str, float], key: str, start_time: float) -> float:
    elapsed = time.perf_counter() - start_time
    timings[key] = elapsed
    return elapsed


def _timing_summary(timings: dict[str, float]) -> str:
    keys = (
        "dataset_load",
        "normalization",
        "group_build",
        "model_init",
        "initial_loss",
        "epoch_loop",
        "final_metrics",
        "prediction_export",
        "prediction_save",
        "best_prediction_export",
        "best_prediction_save",
        "summary_write",
    )
    return " ".join(f"{key}={timings[key]:.2f}s" for key in keys if key in timings)


def _profile_timing_enabled(args: argparse.Namespace) -> bool:
    return bool(args.profile_timing)


def train_metrics_epochs(schedule: str, epochs: int) -> list[int]:
    if epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if schedule == "every_epoch":
        return list(range(1, epochs + 1))
    if schedule == "half_and_final":
        midpoint = int(math.ceil(epochs / 2))
        return sorted({midpoint, epochs})
    if schedule == "final_only":
        return [epochs]
    if schedule == "none":
        return []
    raise ValueError(f"Unknown train metrics schedule: {schedule!r}")


def should_report_grad_norm(grad_norm_report_every: int, batch_index: int) -> bool:
    if grad_norm_report_every < 0:
        raise ValueError("--grad-norm-report-every must be >= 0")
    if batch_index < 1:
        raise ValueError("batch_index must be 1-indexed")
    if grad_norm_report_every == 0:
        return False
    return batch_index % grad_norm_report_every == 0


def grad_norm_reporting_mode(grad_norm_report_every: int) -> str:
    if grad_norm_report_every < 0:
        raise ValueError("--grad-norm-report-every must be >= 0")
    if grad_norm_report_every == 0:
        return "disabled"
    if grad_norm_report_every == 1:
        return "every_batch"
    return f"every_{grad_norm_report_every}_batches"


def should_build_final_predictions(save_predictions: bool) -> bool:
    return bool(save_predictions)


def should_reuse_final_metrics(final_epoch_train_metrics_computed: bool) -> bool:
    return bool(final_epoch_train_metrics_computed)


def _maybe_float(payload: dict[str, Any] | None, key: str) -> float:
    if payload is None:
        return float("nan")
    return float(payload[key])


def _format_progress_value(value: Any, *, precision: int = 6) -> str:
    if value is None:
        return "skipped"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "skipped"
    return f"{numeric:.{precision}e}"


def _format_progress_sigfig_decimal(value: Any, *, significant: int = 3) -> str:
    if value is None:
        return "skipped"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "skipped"
    if numeric == 0.0:
        return "0." + ("0" * max(significant - 1, 0))
    decimals = max(significant - 1 - math.floor(math.log10(abs(numeric))), 0)
    return f"{numeric:.{decimals}f}"


def _format_progress_decimal(value: Any, *, precision: int = 2) -> str:
    if value is None:
        return "skipped"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "skipped"
    return f"{numeric:.{precision}f}"


def _format_progress_percent(value: Any, *, precision: int = 2) -> str:
    if value is None:
        return "skipped"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "skipped"
    return f"{numeric:.{precision}f}%"


def _format_progress_loss(value: Any) -> str:
    if value is None:
        return "skipped"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "skipped"
    return f"{numeric:.3g}"


def _format_progress_int(value: Any) -> str:
    if value is None:
        return "skipped"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "skipped"
    return str(int(numeric))


def _deltaT_error_pct(raw_delta_mse: Any, mean_square_true_deltaT: Any) -> float | None:
    if raw_delta_mse is None or mean_square_true_deltaT is None:
        return None
    try:
        mse = float(raw_delta_mse)
        target_mean_square = float(mean_square_true_deltaT)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(mse)
        or not math.isfinite(target_mean_square)
        or target_mean_square <= 0.0
    ):
        return None
    return 100.0 * math.sqrt(max(mse, 0.0) / target_mean_square)


def _rmse_from_mse(mse: Any) -> float | None:
    if mse is None:
        return None
    try:
        value = float(mse)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return math.sqrt(max(value, 0.0))


def _metric_error_pct(metrics: dict[str, Any] | None) -> float | None:
    if metrics is None:
        return None
    for key in (
        "rel_rmse_v4_pct",
        "raw_deltaT_relative_rmse_pct_v4",
        "raw_deltaT_relative_rmse_pct",
    ):
        if metrics.get(key) is not None:
            return float(metrics[key])
    return _deltaT_error_pct(
        metrics.get("raw_delta_mse"), metrics.get("mean_square_true_deltaT")
    )


def _progress_numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _first_progress_numeric(*values: Any) -> float | None:
    for value in values:
        numeric = _progress_numeric_or_none(value)
        if numeric is not None:
            return numeric
    return None


def _short_hash(parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _group_sample_id_hash(groups: list[dict]) -> str:
    parts = []
    for group in groups:
        parts.append("[group]")
        parts.extend(group.get("sample_ids", ()))
    return _short_hash(parts)


def _current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _bump_profile_count(counts: dict[str, int] | None, key: str, amount: int = 1) -> None:
    if counts is not None:
        counts[key] = int(counts.get(key, 0)) + int(amount)


def _block_until_ready_tree(value) -> None:
    for leaf in tree.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _shape_list(value) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _sample_count(group: dict[str, Any]) -> int:
    target_shape = getattr(group["target_normalized"], "shape", ())
    return int(target_shape[0]) if target_shape else 0


def _graph_edge_shape_count(graphs) -> int | None:
    total = 0
    found = False
    for graph_name in ("p2r", "r2r", "r2p"):
        graph = getattr(graphs, graph_name, None)
        if graph is None:
            continue
        for edge in getattr(graph, "edges", {}).values():
            leaves = tree.tree_leaves(edge.features)
            for leaf in leaves:
                shape = getattr(leaf, "shape", None)
                if shape is not None and len(shape) >= 2:
                    total += int(shape[1])
                    found = True
                    break
    return total if found else None


def _batch_shape_signature(group: dict[str, Any]) -> dict[str, Any]:
    inputs = group["inputs"]
    graph_leaf_shapes = sorted(
        {
            str(tuple(int(dim) for dim in leaf.shape))
            for leaf in tree.tree_leaves(group["graphs"])
            if hasattr(leaf, "shape")
        }
    )
    input_x_shape = _shape_list(inputs.x_inp)
    sample_count = _sample_count(group)
    nodes_per_sample = input_x_shape[-2] if input_x_shape and len(input_x_shape) >= 2 else None
    return {
        "group_count": 1,
        "sample_count": sample_count,
        "total_nodes": sample_count * nodes_per_sample if nodes_per_sample is not None else None,
        "total_edges": _graph_edge_shape_count(group["graphs"]),
        "input_u_shape": _shape_list(inputs.u),
        "input_c_shape": _shape_list(inputs.c),
        "input_x_inp_shape": input_x_shape,
        "input_x_out_shape": _shape_list(inputs.x_out),
        "target_shape": _shape_list(group["target_normalized"]),
        "graph_leaf_shapes": graph_leaf_shapes,
        "graph_leaf_shape_count": len(graph_leaf_shapes),
    }


def _groups_memory_signature(groups: list[dict[str, Any]]) -> dict[str, Any]:
    shape_counts: dict[str, int] = {}
    total_sample_count = 0
    total_edge_count = 0
    for group in groups:
        signature = _batch_shape_signature(group)
        key = _shape_signature_key(signature)
        shape_counts[key] = shape_counts.get(key, 0) + 1
        total_sample_count += int(signature.get("sample_count") or 0)
        total_edge_count += int(signature.get("total_edges") or 0)
    return {
        "group_count": int(len(groups)),
        "sample_count": int(total_sample_count),
        "total_edges": int(total_edge_count),
        "shape_signature_count": int(len(shape_counts)),
        "shape_signature_counts": shape_counts,
    }


def _shape_signature_key(signature: dict[str, Any]) -> str:
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _summarize_batch_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    times = [float(record["total_batch_time"]) for record in records]
    later_times = times[1:]
    later_median = float(np.median(later_times)) if later_times else None
    possible_recompile_count = 0
    for index, record in enumerate(records):
        possible = bool(index > 0 and later_median and float(record["total_batch_time"]) > 3.0 * later_median)
        record["possible_recompile"] = possible
        record["possible_recompile_reason"] = (
            "later_batch_time_gt_3x_later_median" if possible else None
        )
        possible_recompile_count += int(possible)
    return {
        "mean_train_batch_time": float(np.mean(times)) if times else 0.0,
        "median_train_batch_time": float(np.median(times)) if times else 0.0,
        "max_train_batch_time": float(max(times)) if times else 0.0,
        "first_train_batch_time": float(times[0]) if times else 0.0,
        "later_train_batch_median_time": later_median,
        "possible_recompile_batch_count": int(possible_recompile_count),
    }


def _combine_loss_components(weighted_entries: list[tuple[int, dict[str, Any]]]) -> dict[str, float]:
    total_count = sum(count for count, _ in weighted_entries)
    if total_count <= 0:
        return {}
    keys = set()
    for _, components in weighted_entries:
        keys.update(components.keys())
    combined: dict[str, float] = {}
    for key in sorted(keys):
        if key == "pseudo_negative_count":
            combined[key] = float(sum(float(components.get(key, 0.0)) for _, components in weighted_entries))
        else:
            combined[key] = float(
                sum(float(components.get(key, 0.0)) * count for count, components in weighted_entries)
                / total_count
            )
    return combined


def _combine_metric_payloads(weighted_entries: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    total_count = sum(count for count, _ in weighted_entries)
    if total_count <= 0:
        return {}
    numeric_keys = sorted(
        {
            key
            for _, metrics in weighted_entries
            for key, value in metrics.items()
            if key not in {"finite_ok", "shape_ok"}
            and not isinstance(value, (bool, np.bool_))
            and np.asarray(value).ndim == 0
        }
    )
    combined = {
        key: float(
            sum(float(metrics.get(key, 0.0)) * count for count, metrics in weighted_entries)
            / total_count
        )
        for key in numeric_keys
    }
    combined["raw_rmse_K"] = _rmse_from_mse(combined["raw_delta_mse"])
    combined["recovered_T_rmse_K"] = _rmse_from_mse(combined["recovered_temperature_mse"])
    relative_pct = _deltaT_error_pct(
        combined["raw_delta_mse"], combined["mean_square_true_deltaT"]
    )
    combined["rel_rmse_v4_pct"] = relative_pct
    combined["raw_deltaT_relative_rmse_pct_v4"] = relative_pct
    combined["raw_deltaT_relative_rmse_pct"] = relative_pct
    combined["finite_ok"] = all(bool(metrics.get("finite_ok")) for _, metrics in weighted_entries)
    combined["shape_ok"] = all(bool(metrics.get("shape_ok")) for _, metrics in weighted_entries)
    return combined


def _evaluate_groups_profiled(
    model,
    params,
    groups: list[dict],
    stats: dict,
    loss_config: dict[str, Any],
    metrics_fn,
    *,
    epoch: int,
    split: str,
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    component_entries = []
    metric_entries = []
    batch_records = []
    for index, group in enumerate(groups, start=1):
        batch_start = time.perf_counter()
        components = _loss_components(model, params, [group], stats, loss_config)
        metrics = metrics_fn(model, params, [group], stats)
        _block_until_ready_tree((components, metrics))
        elapsed = time.perf_counter() - batch_start
        count = _sample_count(group)
        component_entries.append((count, components))
        metric_entries.append((count, metrics))
        batch_records.append(
            {
                "epoch_index": int(epoch),
                "batch_index": int(index),
                "split": split,
                "batch_size": int(count),
                "group_count": 1,
                "total_batch_time": float(elapsed),
                "batch_shape_signature": _batch_shape_signature(group),
            }
        )
    return (
        _combine_loss_components(component_entries),
        _combine_metric_payloads(metric_entries),
        batch_records,
    )


def _profile_timing_payload(
    *,
    timings: dict[str, float],
    profile_counts: dict[str, int],
    epoch_records: list[dict[str, Any]],
    train_batch_records: list[dict[str, Any]] | None = None,
    validation_batch_records: list[dict[str, Any]] | None = None,
    train_group_count: int,
    valid_group_count: int,
    all_group_count: int,
    prediction_group_count: int | None = None,
    all_groups_built: bool = True,
    train_batch_counts: list[int],
    subset: Path,
    output_dir: Path,
    train_metrics_schedule: str,
    train_metrics_epoch_values: list[int],
    grad_norm_report_every: int = 1,
    grad_norm_reported_batch_count: int = 0,
    grad_norm_skipped_batch_count: int = 0,
    final_metrics_reused: bool = False,
    final_metrics_reuse_source: str | None = None,
    final_prediction_export_skipped: bool = False,
    final_prediction_export_skip_reason: str | None = None,
    total_run_time_so_far: float | None = None,
) -> dict[str, Any]:
    train_batch_records = train_batch_records or []
    validation_batch_records = validation_batch_records or []
    prediction_group_count = int(all_group_count if prediction_group_count is None else prediction_group_count)
    per_epoch = []
    for record in epoch_records:
        batch_summary = record.get("train_batch_timing_summary", {})
        per_epoch.append(
            {
                "epoch": int(record["epoch"]),
                "epoch_index": int(record["epoch"]),
                "total_time_s": float(record.get("epoch_total_time_s", 0.0)),
                "epoch_total_time": float(record.get("epoch_total_time_s", 0.0)),
                "train_time_s": float(record.get("epoch_train_time_s", 0.0)),
                "train_total_time": float(record.get("epoch_train_time_s", 0.0)),
                "train_metrics_time_s": float(record.get("epoch_train_metrics_time_s", 0.0)),
                "train_metrics_time": float(record.get("epoch_train_metrics_time_s", 0.0)),
                "train_metrics_computed": bool(record.get("train_full_metrics_computed", False)),
                "validation_time_s": float(record.get("epoch_validation_time_s", 0.0)),
                "validation_total_time": float(record.get("epoch_validation_time_s", 0.0)),
                "train_batch_count": int(record.get("train_batch_count", 0)),
                "num_train_batches": int(record.get("train_batch_count", 0)),
                "valid_batch_count": int(record.get("valid_batch_count", valid_group_count)),
                "num_valid_batches": int(record.get("valid_batch_count", valid_group_count)),
                "mean_train_batch_time": float(batch_summary.get("mean_train_batch_time", 0.0)),
                "median_train_batch_time": float(batch_summary.get("median_train_batch_time", 0.0)),
                "max_train_batch_time": float(batch_summary.get("max_train_batch_time", 0.0)),
                "first_train_batch_time": float(batch_summary.get("first_train_batch_time", 0.0)),
                "later_train_batch_median_time": batch_summary.get("later_train_batch_median_time"),
                "possible_recompile_batch_count": int(batch_summary.get("possible_recompile_batch_count", 0)),
            }
        )

    run_level = {
        "dataset_load_time": float(timings.get("dataset_load", 0.0)),
        "group_build_time": float(timings.get("group_build", 0.0)),
        "train_groups_count": int(train_group_count),
        "valid_groups_count": int(valid_group_count),
        "all_groups_count": int(all_group_count),
        "all_groups_built": bool(all_groups_built),
        "all_groups_status": "built" if all_groups_built else "skipped",
        "prediction_groups_count": int(prediction_group_count),
        "metadata_calls": int(profile_counts.get("graph_metadata_build_calls", 0)),
        "build_graphs_calls": int(profile_counts.get("graph_build_graphs_calls", 0)),
        "total_run_time_so_far": float(total_run_time_so_far or 0.0),
        "train_metrics_schedule": train_metrics_schedule,
        "train_metrics_epochs": [int(epoch) for epoch in train_metrics_epoch_values],
        "grad_norm_report_every": int(grad_norm_report_every),
        "grad_norm_reported_batch_count": int(grad_norm_reported_batch_count),
        "grad_norm_skipped_batch_count": int(grad_norm_skipped_batch_count),
        "grad_norm_reporting_mode": grad_norm_reporting_mode(int(grad_norm_report_every)),
        "final_metrics_reused": bool(final_metrics_reused),
        "final_metrics_reuse_source": final_metrics_reuse_source,
        "final_metrics_time": float(timings.get("final_metrics", 0.0)),
        "final_prediction_export_skipped": bool(final_prediction_export_skipped),
        "final_prediction_export_skip_reason": final_prediction_export_skip_reason,
    }
    return {
        "schema_version": 1,
        "diagnostic_scope": "Heat3D v2 graph/group build timing profile",
        "subset": str(subset),
        "output_dir": str(output_dir),
        "run_level": run_level,
        "train_metrics_schedule": train_metrics_schedule,
        "train_metrics_epochs": [int(epoch) for epoch in train_metrics_epoch_values],
        "grad_norm_report_every": int(grad_norm_report_every),
        "grad_norm_reporting_mode": grad_norm_reporting_mode(int(grad_norm_report_every)),
        "final_metrics_reused": bool(final_metrics_reused),
        "final_metrics_reuse_source": final_metrics_reuse_source,
        "final_prediction_export_skipped": bool(final_prediction_export_skipped),
        "final_prediction_export_skip_reason": final_prediction_export_skip_reason,
        "timings_s": {key: float(value) for key, value in sorted(timings.items())},
        "counts": {
            "train_groups": int(train_group_count),
            "valid_groups": int(valid_group_count),
            "all_groups": int(all_group_count),
            "train_batches_per_epoch": [int(value) for value in train_batch_counts],
            "valid_batches": int(valid_group_count),
            "prediction_batches": int(prediction_group_count),
            **{key: int(value) for key, value in sorted(profile_counts.items())},
        },
        "per_epoch": per_epoch,
        "train_batches": train_batch_records,
        "validation_batches": validation_batch_records,
    }


def _print_profile_timing(payload: dict[str, Any]) -> None:
    counts = payload["counts"]
    timings = payload["timings_s"]
    run_level = payload.get("run_level", {})
    _emit("")
    _emit("profile timing")
    _emit(
        "  graph/group: "
        f"group_build={run_level.get('group_build_time', timings.get('group_build', 0.0)):.2f}s "
        f"metadata_calls={run_level.get('metadata_calls', counts.get('graph_metadata_build_calls', 0))} "
        f"build_graphs_calls={run_level.get('build_graphs_calls', counts.get('graph_build_graphs_calls', 0))}"
    )
    _emit(
        "  groups/batches: "
        f"train_groups={counts['train_groups']} valid_groups={counts['valid_groups']} "
        f"all_groups={counts['all_groups']} "
        f"all_groups_status={run_level.get('all_groups_status', 'built')} "
        f"valid_batches={counts['valid_batches']} "
        f"prediction_batches={counts['prediction_batches']}"
    )
    _emit(
        "  train_metrics: "
        f"schedule={payload.get('train_metrics_schedule')} "
        f"epochs={payload.get('train_metrics_epochs')}"
    )
    _emit(
        "  grad_norm: "
        f"mode={payload.get('grad_norm_reporting_mode')} "
        f"reported={run_level.get('grad_norm_reported_batch_count', 0)} "
        f"skipped={run_level.get('grad_norm_skipped_batch_count', 0)}"
    )
    _emit(
        "  final: "
        f"metrics_reused={payload.get('final_metrics_reused')} "
        f"prediction_export_skipped={payload.get('final_prediction_export_skipped')}"
    )
    for record in payload["per_epoch"]:
        _emit(
            "  epoch "
            f"{record['epoch_index']:03d}: total={record['epoch_total_time']:.2f}s "
            f"train={record['train_total_time']:.2f}s "
            f"train_metrics={record['train_metrics_time']:.2f}s "
            f"train_metrics_computed={record['train_metrics_computed']} "
            f"validation={record['validation_total_time']:.2f}s "
            f"train_batches={record['num_train_batches']} "
            f"first_batch={record['first_train_batch_time']:.2f}s "
            f"later_median={record['later_train_batch_median_time']} "
            f"possible_recompile={record['possible_recompile_batch_count']}"
        )
    if "prediction_export" in timings:
        _emit(f"  prediction_export={timings['prediction_export']:.2f}s")
    if "final_metrics" in timings:
        _emit(f"  final_metrics={timings['final_metrics']:.2f}s")


def _ensure_ignored_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    output_root = (REPO_DIR / "output").resolve()
    if resolved != output_root and output_root not in resolved.parents:
        raise ValueError(f"--output-dir must be under ignored output/: {path}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _ensure_ignored_output_file(path: Path, flag: str) -> Path:
    resolved = path.resolve()
    output_root = (REPO_DIR / "output").resolve()
    if resolved == output_root or output_root not in resolved.parents:
        raise ValueError(f"--{flag} must be a file under ignored output/: {path}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _current_rss_mb() -> float | None:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def _jax_memory_snapshot() -> dict[str, Any]:
    """Return optional JAX device memory statistics without probing hardware."""

    snapshot: dict[str, Any] = {"jax_devices": []}
    for device in jax.devices():
        memory_stats = getattr(device, "memory_stats", None)
        if memory_stats is None:
            continue
        try:
            stats = memory_stats()
        except Exception:
            continue
        if not isinstance(stats, dict):
            continue
        converted = {
            "device": str(device),
            "platform": getattr(device, "platform", None),
        }
        for key, value in stats.items():
            if key.endswith("bytes") or key in {
                "bytes_in_use",
                "peak_bytes_in_use",
                "bytes_limit",
                "bytes_reserved",
                "peak_bytes_reserved",
                "pool_bytes",
                "peak_pool_bytes",
                "largest_alloc_size",
                "largest_free_block_bytes",
            }:
                converted[f"{key}_mb"] = float(value) / (1024.0 * 1024.0)
            else:
                converted[key] = value
        snapshot["jax_devices"].append(converted)

    return snapshot


class MemoryAudit:
    def __init__(self, path: Path, *, every_batch: bool = False, gc_enabled: bool = False):
        self.path = path
        self.every_batch = bool(every_batch)
        self.gc_enabled = bool(gc_enabled)
        self.event_index = 0
        self.peak_rss_mb = 0.0
        self.peak_device_memory_mb: dict[str, float] = {}
        with self.path.open("w", encoding="utf-8") as file:
            file.write("")

    def record(
        self,
        stage: str,
        *,
        epoch: int | None = None,
        batch_index: int | None = None,
        split: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.event_index += 1
        jax_memory = _jax_memory_snapshot()
        rss_mb = _current_rss_mb()
        if rss_mb is not None:
            self.peak_rss_mb = max(self.peak_rss_mb, float(rss_mb))
        for device in jax_memory.get("jax_devices", []):
            device_name = str(device.get("device", "unknown"))
            candidates = [
                float(value)
                for key, value in device.items()
                if key in {"bytes_in_use_mb", "peak_bytes_in_use_mb", "peak_pool_bytes_mb"}
                and value is not None
            ]
            if candidates:
                self.peak_device_memory_mb[device_name] = max(
                    self.peak_device_memory_mb.get(device_name, 0.0),
                    *candidates,
                )
        payload = {
            "event_index": self.event_index,
            "time_unix": time.time(),
            "stage": stage,
            "epoch": epoch,
            "batch_index": batch_index,
            "split": split,
            "rss_mb": rss_mb,
            "jax_memory": jax_memory,
            "detail": detail or {},
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")
            file.flush()

    def summary(self) -> dict[str, Any]:
        return {
            "event_count": int(self.event_index),
            "peak_rss_mb": float(self.peak_rss_mb),
            "peak_device_memory_mb": dict(sorted(self.peak_device_memory_mb.items())),
            "peak_device_memory_all_mb": (
                max(self.peak_device_memory_mb.values())
                if self.peak_device_memory_mb
                else None
            ),
        }

    def collect(self, stage: str, *, epoch: int | None = None) -> None:
        if self.gc_enabled:
            gc.collect()
            self.record(stage, epoch=epoch)


def _output_filename(name: str, flag: str) -> str:
    path = Path(name)
    if path.name != name or path.is_absolute():
        raise ValueError(f"--{flag} must be a filename under --output-dir, found {name}")
    if not name:
        raise ValueError(f"--{flag} must not be empty")
    return name


def _require_train_valid_splits(split_ids: dict[str, list[str]]) -> None:
    train_ids = split_ids.get("train", [])
    valid_ids = split_ids.get("valid", [])
    if not train_ids or not valid_ids:
        raise ValueError(
            "Expected non-empty train and valid splits for controlled training export, "
            f"found train={len(train_ids)} valid={len(valid_ids)}"
        )


def _load_external_split_map(path: Path) -> dict[str, list[str]]:
    return split_ids_from_sample_splits(load_sample_split_map(path))


def _is_heat3d_v4_subset(sample_root: Path) -> bool:
    return sample_root.name.startswith("heat3d_v4_") or "heat3d_v4_" in str(sample_root)


def _normalize_v4_train_test_splits(
    split_ids: dict[str, list[str]],
) -> tuple[dict[str, list[str]], bool]:
    normalized = {split: list(ids) for split, ids in split_ids.items()}
    normalized.pop("valid_stress", None)
    bridged = False
    if not normalized.get("valid_iid") and normalized.get("test"):
        normalized["valid_iid"] = normalized.pop("test")
        bridged = True
    return normalized, bridged


def _resolve_training_splits(
    sample_root: Path,
    split_map_path: Path | None,
) -> tuple[dict[str, list[str]], str, str, str | None]:
    if split_map_path is None:
        split_ids = _subset_split_ids(sample_root)
        if _is_heat3d_v4_subset(sample_root):
            split_ids, bridged = _normalize_v4_train_test_splits(split_ids)
            train_ids = split_ids.get("train", [])
            valid_iid_ids = split_ids.get("valid_iid", [])
            if not train_ids or not valid_iid_ids:
                raise ValueError(
                    "Expected non-empty train and valid_iid/test splits for "
                    "Heat3D V4 dataset sample_meta, found "
                    f"train={len(train_ids)} valid_iid={len(valid_iid_ids)} "
                    f"test={len(split_ids.get('test', []))}"
                )
            source = "sample_meta_v4_train_test_bridge" if bridged else "sample_meta_v4"
            return split_ids, source, "valid_iid", None
        _require_train_valid_splits(split_ids)
        return split_ids, "sample_meta", "valid", None

    split_ids = _load_external_split_map(split_map_path)
    if _is_heat3d_v4_subset(sample_root):
        split_ids, bridged = _normalize_v4_train_test_splits(split_ids)
        train_ids = split_ids.get("train", [])
        valid_iid_ids = split_ids.get("valid_iid", [])
        if not train_ids or not valid_iid_ids:
            raise ValueError(
                "Expected non-empty train and valid_iid/test splits for "
                "--split-map on Heat3D V4 dataset, found "
                f"train={len(train_ids)} valid_iid={len(valid_iid_ids)} "
                f"test={len(split_ids.get('test', []))}"
            )
        source = "split_map_v4_train_test_bridge" if bridged else "split_map_v4"
        return split_ids, source, "valid_iid", None

    train_ids = split_ids.get("train", [])
    valid_iid_ids = split_ids.get("valid_iid", [])
    if not train_ids or not valid_iid_ids:
        raise ValueError(
            "Expected non-empty train and valid_iid splits for --split-map, "
            f"found train={len(train_ids)} valid_iid={len(valid_iid_ids)}"
        )
    return split_ids, "split_map", "valid_iid", "valid_stress" if split_ids.get("valid_stress") else None


def _should_report_epoch(epoch: int, epochs: int, report_every: int) -> bool:
    return epoch == 1 or epoch == epochs or epoch % report_every == 0


def _loss_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "loss_mode": args.loss_mode,
        "background_quantile": float(args.background_quantile),
        "hotspot_quantile": float(args.hotspot_quantile),
        "background_weight": float(args.background_weight),
        "hotspot_weight": float(args.hotspot_weight),
        "strong_q_quantile": float(args.strong_q_quantile),
        "strong_q_weight": float(args.strong_q_weight),
        "native_shape_cv_weight": float(args.native_shape_cv_weight),
        "native_log_scale_weight": float(args.native_log_scale_weight),
        "native_relative_field_weight": float(args.native_relative_field_weight),
        "native_raw_field_weight": float(args.native_raw_field_weight),
        "background_l1_weight": float(args.background_l1_weight),
        "background_bias_weight": float(args.background_bias_weight),
        "background_over_weight": float(args.background_over_weight),
        "background_relative_weight": float(args.background_relative_weight),
        "relative_floor": float(args.relative_floor),
        "relative_floor_mode": args.relative_floor_mode,
        "pseudo_negative_quantile": float(args.pseudo_negative_quantile),
        "pseudo_negative_delta_threshold": (
            None if args.pseudo_negative_delta_threshold is None else float(args.pseudo_negative_delta_threshold)
        ),
        "pseudo_negative_weight": float(args.pseudo_negative_weight),
        "pseudo_negative_over_margin": float(args.pseudo_negative_over_margin),
        "pseudo_negative_min_count": int(args.pseudo_negative_min_count),
        "pseudo_negative_loss_type": args.pseudo_negative_loss_type,
        "pseudo_negative_relative_floor": float(args.pseudo_negative_relative_floor),
        "loss_weight_schedule": args.loss_weight_schedule,
        "loss_transition_epoch": int(args.loss_transition_epoch),
        "background_relative_weight_start": args.background_relative_weight_start,
        "background_relative_weight_end": args.background_relative_weight_end,
        "hotspot_weight_start": args.hotspot_weight_start,
        "hotspot_weight_end": args.hotspot_weight_end,
        "background_l1_weight_start": args.background_l1_weight_start,
        "background_l1_weight_end": args.background_l1_weight_end,
        "background_bias_weight_start": args.background_bias_weight_start,
        "background_bias_weight_end": args.background_bias_weight_end,
        "background_over_weight_start": args.background_over_weight_start,
        "background_over_weight_end": args.background_over_weight_end,
        "loss_space": (
            "base, hotspot, and strong-q terms use normalized_deltaT; "
            "background MSE/L1/bias/overprediction/relative terms use raw_deltaT_K"
        ),
        "base_loss_space": "normalized_deltaT",
        "background_mask_space": "raw_deltaT_K quantile",
        "background_penalty_space": "raw_deltaT_K_squared; penalizes pred_raw_deltaT toward 0",
        "background_l1_space": "raw_deltaT_K_abs; penalizes abs(pred_raw_deltaT) in background",
        "background_signed_bias_loss_space": "raw_deltaT_K_abs_bias; penalizes abs(mean(pred_raw_deltaT - true_raw_deltaT))",
        "background_overprediction_loss_space": "raw_deltaT_K_positive_error; penalizes mean(relu(pred_raw_deltaT - true_raw_deltaT))",
        "background_relative_abs_space": "safe raw_deltaT_K relative absolute error in background",
        "background_relative_abs_denominator": (
            "max(abs(true_raw_deltaT), floor), where floor is fixed relative_floor or "
            "max(relative_floor, batch/group abs true raw DeltaT p50/p75)"
        ),
        "pseudo_negative_mask_space": "raw_deltaT_K high-confidence near-zero quantile and optional raw threshold",
        "pseudo_negative_over_loss_space": "raw_deltaT_K overprediction-only hinge; mse/l1/relative_l1/relative_mse selectable",
        "hotspot_mask_space": "raw_deltaT_K quantile",
        "hotspot_retention_loss_space": "normalized_deltaT",
        "hotspot_strong_q_hotspot_mask_space": "sample-wise raw_deltaT_K top quantile",
        "strong_q_mask_space": "sample-wise q>0 top quantile from unnormalized q feature",
        "strong_q_loss_space": "normalized_deltaT",
        "target_normalization": "normalized_deltaT = (raw_deltaT - train_target_delta_mean) / train_target_delta_std",
    }


def _lr_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "lr": float(args.lr),
        "lr_schedule": args.lr_schedule,
        "warmup_epochs": int(args.warmup_epochs),
        "min_lr": float(args.min_lr),
        "second_stage_epoch": int(args.second_stage_epoch),
        "second_stage_lr": float(args.second_stage_lr),
        "lr_init": float(args.lr_init),
        "lr_peak": float(args.lr_peak),
        "lr_base": float(args.lr_base),
        "lr_lowr": float(args.lr_lowr),
        "pct_start": float(args.pct_start),
        "pct_final": float(args.pct_final),
    }


def _optimizer_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "optimizer": args.optimizer,
        "gradient_clip_norm": (
            None if args.gradient_clip_norm is None else float(args.gradient_clip_norm)
        ),
        "weight_decay": float(args.weight_decay),
    }


def _seed_config_from_args(args: argparse.Namespace) -> dict[str, int]:
    legacy_seed = int(args.seed)
    return {
        "seed": legacy_seed,
        "legacy_seed": legacy_seed,
        "model_seed": legacy_seed if args.model_seed is None else int(args.model_seed),
        "batch_order_seed": 0 if args.batch_order_seed is None else int(args.batch_order_seed),
        "graph_seed": 0 if args.graph_seed is None else int(args.graph_seed),
    }


def _model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    local_feature_names = _parse_csv_feature_names(
        args.decoder_bypass_local_feature_names,
        "--decoder-bypass-local-feature-names",
    )
    global_feature_names = _parse_csv_feature_names(
        args.global_context_feature_names,
        "--global-context-feature-names",
    )
    model_config = dict(RUNNER_MODEL_CONFIG)
    model_config.update(
        {
            "node_latent_size": int(args.node_latent_size),
            "edge_latent_size": int(args.edge_latent_size),
            "processor_steps": int(args.processor_steps),
            "mlp_hidden_layers": int(args.mlp_hidden_layers),
            "p_edge_masking": float(args.p_edge_masking),
            "decoder_bypass_mode": args.decoder_bypass_mode,
            "decoder_bypass_features": args.decoder_bypass_features,
            "decoder_bypass_feature_source": args.decoder_bypass_feature_source,
            "decoder_bypass_feature_indices": (),
            "decoder_bypass_feature_names": (),
            "decoder_bypass_num_features": 0,
            "decoder_bypass_local_feature_names": local_feature_names,
            "decoder_bypass_output_space": args.decoder_bypass_output_space,
            "decoder_bypass_hidden_size": int(args.decoder_bypass_hidden_size),
            "decoder_bypass_layers": int(args.decoder_bypass_layers),
            "decoder_bypass_init": args.decoder_bypass_init,
            "decoder_bypass_residual_scale": float(args.decoder_bypass_residual_scale),
            "global_context_mode": args.global_context_mode,
            "global_context_feature_dim": len(global_feature_names),
            "global_context_feature_names": global_feature_names,
            "film_target": args.film_target,
            "film_init": args.film_init,
            "film_hidden_size": int(args.film_hidden_size),
            "native_output_mode": args.native_output_mode,
            "native_branch_mode": args.native_branch_mode,
            "scale_head_mode": args.scale_head_mode,
            "scale_pooling": args.scale_pooling,
            "scale_head_hidden_size": int(args.scale_head_hidden_size),
            "scale_head_init": "identity",
            "shape_scale_epsilon": 1.0e-12,
        }
    )
    return model_config


def _parse_csv_feature_names(value: str, flag_name: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in str(value or "").split(",") if name.strip())
    if len(names) != len(set(names)):
        raise ValueError(f"{flag_name} must not contain duplicate feature names")
    return names


def _batch_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size": _optional_batch_size(args.batch_size, "batch-size"),
        "validation_batch_size": _optional_batch_size(
            args.validation_batch_size, "validation-batch-size"
        ),
        "prediction_batch_size": _optional_batch_size(
            args.prediction_batch_size, "prediction-batch-size"
        ),
        "shuffle_train_batches": bool(args.shuffle_train_batches),
        "drop_last": bool(args.drop_last),
        "batch_plan": args.batch_plan,
        "batch_build_seed": 0 if args.batch_build_seed is None else int(args.batch_build_seed),
    }


def _graph_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "node_coordinate_encoding": args.node_coordinate_encoding,
        "node_coordinate_freqs": int(args.node_coordinate_freqs),
        "radius_policy": args.radius_policy,
        "coverage_repair_policy": args.coverage_repair_policy,
        "repair_p2r": bool(args.repair_p2r),
        "repair_r2p": bool(args.repair_r2p),
        "min_physical_coverage": int(args.min_physical_coverage),
    }


def _validate_loss_config(config: dict[str, Any]) -> None:
    background_quantile = float(config["background_quantile"])
    hotspot_quantile = float(config["hotspot_quantile"])
    if not 0.0 <= background_quantile <= 1.0:
        raise ValueError("--background-quantile must be in [0, 1]")
    if not 0.0 <= hotspot_quantile <= 1.0:
        raise ValueError("--hotspot-quantile must be in [0, 1]")
    if background_quantile > hotspot_quantile:
        raise ValueError("--background-quantile must be <= --hotspot-quantile")
    if float(config["background_weight"]) < 0.0:
        raise ValueError("--background-weight must be >= 0")
    if float(config["hotspot_weight"]) < 0.0:
        raise ValueError("--hotspot-weight must be >= 0")
    strong_q_quantile = float(config["strong_q_quantile"])
    if not 0.0 <= strong_q_quantile <= 1.0:
        raise ValueError("--strong-q-quantile must be in [0, 1]")
    if float(config["strong_q_weight"]) < 0.0:
        raise ValueError("--strong-q-weight must be >= 0")
    for key in (
        "native_shape_cv_weight", "native_log_scale_weight",
        "native_relative_field_weight", "native_raw_field_weight",
    ):
        if float(config[key]) < 0.0:
            raise ValueError(f"--{key.replace('_', '-')} must be >= 0")
    if float(config["background_l1_weight"]) < 0.0:
        raise ValueError("--background-l1-weight must be >= 0")
    if float(config["background_bias_weight"]) < 0.0:
        raise ValueError("--background-bias-weight must be >= 0")
    if float(config["background_over_weight"]) < 0.0:
        raise ValueError("--background-over-weight must be >= 0")
    if float(config["background_relative_weight"]) < 0.0:
        raise ValueError("--background-relative-weight must be >= 0")
    if float(config["relative_floor"]) <= 0.0:
        raise ValueError("--relative-floor must be > 0")
    pseudo_negative_quantile = float(config["pseudo_negative_quantile"])
    if not 0.0 <= pseudo_negative_quantile <= 1.0:
        raise ValueError("--pseudo-negative-quantile must be in [0, 1]")
    if float(config["pseudo_negative_weight"]) < 0.0:
        raise ValueError("--pseudo-negative-weight must be >= 0")
    if float(config["pseudo_negative_over_margin"]) < 0.0:
        raise ValueError("--pseudo-negative-over-margin must be >= 0")
    if int(config["pseudo_negative_min_count"]) < 0:
        raise ValueError("--pseudo-negative-min-count must be >= 0")
    if float(config["pseudo_negative_relative_floor"]) <= 0.0:
        raise ValueError("--pseudo-negative-relative-floor must be > 0")
    if int(config["loss_transition_epoch"]) < 0:
        raise ValueError("--loss-transition-epoch must be >= 0")
    for key in (
        "background_relative_weight_start",
        "background_relative_weight_end",
        "hotspot_weight_start",
        "hotspot_weight_end",
        "background_l1_weight_start",
        "background_l1_weight_end",
        "background_bias_weight_start",
        "background_bias_weight_end",
        "background_over_weight_start",
        "background_over_weight_end",
    ):
        value = config.get(key)
        if value is not None and float(value) < 0.0:
            raise ValueError(f"--{key.replace('_', '-')} must be >= 0")


def _validate_lr_config(config: dict[str, Any]) -> None:
    if float(config["lr"]) < 0.0:
        raise ValueError("--lr must be >= 0")
    if int(config["warmup_epochs"]) < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if float(config["min_lr"]) < 0.0:
        raise ValueError("--min-lr must be >= 0")
    if int(config["second_stage_epoch"]) < 0:
        raise ValueError("--second-stage-epoch must be >= 0")
    if float(config["second_stage_lr"]) < 0.0:
        raise ValueError("--second-stage-lr must be >= 0")
    for key in ("lr_init", "lr_peak", "lr_base", "lr_lowr"):
        if float(config[key]) < 0.0:
            raise ValueError(f"--{key.replace('_', '-')} must be >= 0")
    for key in ("pct_start", "pct_final"):
        value = float(config[key])
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{key.replace('_', '-')} must be in [0, 1]")
    if config["lr_schedule"] == "upstream_onecycle":
        if float(config["pct_start"]) <= 0.0:
            raise ValueError("--pct-start must be > 0 for upstream_onecycle")
        if float(config["pct_final"]) < 0.0 or float(config["pct_final"]) >= 1.0:
            raise ValueError("--pct-final must be in [0, 1) for upstream_onecycle")
        if float(config["pct_start"]) + float(config["pct_final"]) >= 1.0:
            raise ValueError("--pct-start + --pct-final must be < 1 for upstream_onecycle")


def _validate_optimizer_config(config: dict[str, Any]) -> None:
    if config["optimizer"] not in {"manual_gd", "adam", "adamw"}:
        raise ValueError("--optimizer must be manual_gd, adam, or adamw")
    gradient_clip_norm = config.get("gradient_clip_norm")
    if gradient_clip_norm is not None and float(gradient_clip_norm) <= 0.0:
        raise ValueError("--gradient-clip-norm must be > 0 when provided")
    if float(config["weight_decay"]) < 0.0:
        raise ValueError("--weight-decay must be >= 0")


def _validate_seed_config(config: dict[str, int]) -> None:
    for key in ("seed", "legacy_seed", "model_seed", "batch_order_seed", "graph_seed"):
        if int(config[key]) < 0:
            raise ValueError(f"--{key.replace('_', '-')} must be >= 0")


def _validate_model_config(config: dict[str, Any]) -> None:
    for key in ("node_latent_size", "edge_latent_size", "processor_steps", "mlp_hidden_layers"):
        if int(config[key]) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be >= 1")
    p_edge_masking = float(config.get("p_edge_masking", 0.0))
    if p_edge_masking < 0.0 or p_edge_masking >= 1.0:
        raise ValueError("--p-edge-masking must be in [0, 1)")
    _validate_decoder_bypass_config(config)
    _validate_global_context_config(config)
    native_mode = config.get("native_output_mode", "legacy_normalized_deltaT")
    if native_mode not in NATIVE_OUTPUT_MODES:
        raise ValueError(f"--native-output-mode must be one of {NATIVE_OUTPUT_MODES}")
    if config.get("native_branch_mode") not in NATIVE_BRANCH_MODES:
        raise ValueError(f"--native-branch-mode must be one of {NATIVE_BRANCH_MODES}")
    if config.get("scale_head_mode") not in SCALE_HEAD_MODES:
        raise ValueError(f"--scale-head-mode must be one of {SCALE_HEAD_MODES}")
    if config.get("scale_pooling") not in SCALE_POOLING_MODES:
        raise ValueError(f"--scale-pooling must be one of {SCALE_POOLING_MODES}")
    if int(config.get("scale_head_hidden_size", 0)) < 1:
        raise ValueError("--scale-head-hidden-size must be >= 1")


def _validate_decoder_bypass_config(config: dict[str, Any]) -> None:
    mode = config.get("decoder_bypass_mode", DECODER_BYPASS_MODE_NONE)
    features = config.get("decoder_bypass_features", DECODER_BYPASS_FEATURES_NONE)
    source = config.get(
        "decoder_bypass_feature_source",
        DECODER_BYPASS_FEATURE_SOURCE_NORMALIZED_C,
    )
    init = config.get("decoder_bypass_init", DECODER_BYPASS_INIT_ZERO_RESIDUAL)
    output_space = config.get("decoder_bypass_output_space", "normalized_deltaT")
    if mode not in DECODER_BYPASS_MODES:
        raise ValueError(f"--decoder-bypass-mode must be one of {DECODER_BYPASS_MODES}")
    if features not in DECODER_BYPASS_FEATURES:
        raise ValueError(
            f"--decoder-bypass-features must be one of {DECODER_BYPASS_FEATURES}"
        )
    if source not in DECODER_BYPASS_FEATURE_SOURCES:
        raise ValueError(
            "--decoder-bypass-feature-source must be one of "
            f"{DECODER_BYPASS_FEATURE_SOURCES}"
        )
    if init not in DECODER_BYPASS_INITS:
        raise ValueError(f"--decoder-bypass-init must be one of {DECODER_BYPASS_INITS}")
    if output_space not in DECODER_BYPASS_OUTPUT_SPACES:
        raise ValueError(
            "--decoder-bypass-output-space must be one of "
            f"{DECODER_BYPASS_OUTPUT_SPACES}"
        )
    if int(config.get("decoder_bypass_hidden_size", 0)) < 1:
        raise ValueError("--decoder-bypass-hidden-size must be >= 1")
    if int(config.get("decoder_bypass_layers", 0)) < 1:
        raise ValueError("--decoder-bypass-layers must be >= 1")
    if float(config.get("decoder_bypass_residual_scale", 0.0)) < 0.0:
        raise ValueError("--decoder-bypass-residual-scale must be >= 0")
    if mode == DECODER_BYPASS_MODE_NONE:
        if features != DECODER_BYPASS_FEATURES_NONE:
            raise ValueError(
                "--decoder-bypass-mode none requires --decoder-bypass-features none"
            )
        return
    if mode != DECODER_BYPASS_MODE_POST_DECODER_RESIDUAL:
        raise ValueError(f"unsupported decoder_bypass_mode: {mode}")
    if features not in {
        DECODER_BYPASS_FEATURES_FULL_CONDITION,
        DECODER_BYPASS_FEATURES_EXPLICIT_LOCAL_CONDITION,
    }:
        raise ValueError(
            "--decoder-bypass-mode post_decoder_residual requires "
            "--decoder-bypass-features full_condition or explicit_local_condition"
        )
    indices = tuple(config.get("decoder_bypass_feature_indices") or ())
    names = tuple(config.get("decoder_bypass_feature_names") or ())
    if indices and len(indices) != int(config.get("decoder_bypass_num_features", 0)):
        raise ValueError("decoder_bypass_num_features must match feature_indices")
    if names and len(names) != int(config.get("decoder_bypass_num_features", 0)):
        raise ValueError("decoder_bypass_num_features must match feature_names")


def _validate_global_context_config(config: dict[str, Any]) -> None:
    mode = config.get("global_context_mode", GLOBAL_CONTEXT_MODE_NONE)
    names = tuple(config.get("global_context_feature_names") or ())
    feature_dim = int(config.get("global_context_feature_dim", 0))
    target = config.get("film_target", FILM_TARGET_RNODES_PROCESSED)
    init = config.get("film_init", FILM_INIT_IDENTITY)
    hidden = int(config.get("film_hidden_size", 64))
    if mode not in GLOBAL_CONTEXT_MODES:
        raise ValueError(f"--global-context-mode must be one of {GLOBAL_CONTEXT_MODES}")
    if target not in FILM_TARGETS:
        raise ValueError(f"--film-target must be one of {FILM_TARGETS}")
    if init not in FILM_INITS:
        raise ValueError(f"--film-init must be one of {FILM_INITS}")
    if hidden < 1:
        raise ValueError("--film-hidden-size must be >= 1")
    if feature_dim != len(names):
        raise ValueError("global_context_feature_dim must match global_context_feature_names")
    native_enabled = config.get("native_output_mode") == "native_shape_scale"
    if mode == GLOBAL_CONTEXT_MODE_NONE:
        if (names or feature_dim) and not native_enabled:
            raise ValueError("--global-context-mode none requires no global context feature names")
        if native_enabled:
            validate_global_context_schema(names)
        return
    validate_global_context_schema(names)


def _resolve_decoder_bypass_model_config(
    model_config: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(model_config)
    if resolved["decoder_bypass_mode"] == DECODER_BYPASS_MODE_NONE:
        resolved["decoder_bypass_feature_indices"] = ()
        resolved["decoder_bypass_feature_names"] = ()
        resolved["decoder_bypass_num_features"] = 0
        return resolved

    feature_names = tuple(stats.get("feature_names") or ())
    feature_mode = resolved["decoder_bypass_features"]
    if feature_mode == DECODER_BYPASS_FEATURES_FULL_CONDITION:
        required_feature_names = decoder_bypass_required_full_condition_features(
            input_feature_schema=str(stats.get("input_feature_schema", "legacy_bc_flags")),
            extent_feature_policy=str(stats.get("extent_feature_policy", "none")),
        )
        missing = [name for name in required_feature_names if name not in feature_names]
        if missing:
            raise ValueError(f"decoder bypass missing condition features: {missing}; available={feature_names}")
        # Preserve the checkpoint's original normalized-c column order exactly.
        selected_feature_names = feature_names
    elif feature_mode == DECODER_BYPASS_FEATURES_EXPLICIT_LOCAL_CONDITION:
        selected_feature_names = tuple(resolved.get("decoder_bypass_local_feature_names") or ())
        if not selected_feature_names:
            raise ValueError(
                "decoder_bypass_features=explicit_local_condition requires "
                "--decoder-bypass-local-feature-names"
            )
        disallowed = [
            name for name in selected_feature_names
            if name not in DECODER_BYPASS_LOCAL_FEATURE_ALLOWLIST
        ]
        if disallowed:
            raise ValueError(
                "explicit_local_condition only accepts audited node-local features; "
                f"disallowed={disallowed} allowlist={DECODER_BYPASS_LOCAL_FEATURE_ALLOWLIST}"
            )
        if len(selected_feature_names) != len(set(selected_feature_names)):
            raise ValueError("explicit local bypass feature names must be unique")
    else:
        raise ValueError(f"unsupported decoder_bypass_features: {feature_mode}")
    if feature_mode != DECODER_BYPASS_FEATURES_FULL_CONDITION:
        missing = [name for name in selected_feature_names if name not in feature_names]
        if missing:
            raise ValueError(f"decoder bypass missing condition features: {missing}; available={feature_names}")
    indices = tuple(feature_names.index(name) for name in selected_feature_names)
    resolved["decoder_bypass_feature_indices"] = indices
    resolved["decoder_bypass_feature_names"] = selected_feature_names
    resolved["decoder_bypass_num_features"] = len(indices)
    _validate_decoder_bypass_config(resolved)
    return resolved


def _decoder_bypass_payload(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "decoder_bypass_mode": model_config.get("decoder_bypass_mode"),
        "decoder_bypass_features": model_config.get("decoder_bypass_features"),
        "decoder_bypass_feature_source": model_config.get("decoder_bypass_feature_source"),
        "decoder_bypass_hidden_size": int(model_config.get("decoder_bypass_hidden_size", 0)),
        "decoder_bypass_layers": int(model_config.get("decoder_bypass_layers", 0)),
        "decoder_bypass_init": model_config.get("decoder_bypass_init"),
        "decoder_bypass_residual_scale": float(
            model_config.get("decoder_bypass_residual_scale", 0.0)
        ),
        "decoder_bypass_feature_names": list(
            model_config.get("decoder_bypass_feature_names") or ()
        ),
        "decoder_bypass_local_feature_names": list(
            model_config.get("decoder_bypass_local_feature_names") or ()
        ),
        "decoder_bypass_feature_indices": [
            int(index) for index in model_config.get("decoder_bypass_feature_indices") or ()
        ],
        "decoder_bypass_num_features": int(
            model_config.get("decoder_bypass_num_features", 0)
        ),
        "decoder_bypass_output_space": model_config.get("decoder_bypass_output_space"),
    }


def _check_decoder_bypass_input_alignment(
    model_config: dict[str, Any],
    groups: list[dict],
) -> None:
    if model_config.get("decoder_bypass_mode") == DECODER_BYPASS_MODE_NONE:
        return
    for group in groups:
        inputs = group["inputs"]
        x_inp = np.asarray(inputs.x_inp)
        x_out = np.asarray(inputs.x_out)
        if x_inp.shape != x_out.shape:
            raise ValueError(
                "decoder bypass requires one-to-one x_inp/x_out node alignment; "
                f"group={group.get('group_name')} x_inp={x_inp.shape} x_out={x_out.shape}"
            )
        max_abs = float(np.max(np.abs(x_inp - x_out))) if x_inp.size else 0.0
        if max_abs > 1.0e-8:
            raise ValueError(
                "decoder bypass requires identical x_inp/x_out node ordering; "
                f"group={group.get('group_name')} max_abs_coord_diff={max_abs}"
            )


def _validate_batch_config(config: dict[str, Any]) -> None:
    for key in ("batch_size", "validation_batch_size", "prediction_batch_size"):
        value = config.get(key)
        if value is not None and int(value) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be >= 1 or 0 for legacy full-batch")
    if config["batch_plan"] not in {"current_graph_shape", "sample_shuffle"}:
        raise ValueError("--batch-plan must be current_graph_shape or sample_shuffle")
    if config["batch_plan"] == "sample_shuffle" and config["batch_size"] is None:
        raise ValueError("--batch-plan sample_shuffle requires --batch-size >= 1")
    if int(config["batch_build_seed"]) < 0:
        raise ValueError("--batch-build-seed must be >= 0")


def _validate_graph_config(config: dict[str, Any]) -> None:
    if config["node_coordinate_encoding"] not in NODE_COORDINATE_ENCODING_CHOICES:
        raise ValueError(
            "--node-coordinate-encoding must be one of "
            f"{NODE_COORDINATE_ENCODING_CHOICES}"
        )
    if int(config["node_coordinate_freqs"]) < 1:
        raise ValueError("--node-coordinate-freqs must be >= 1")
    if config["radius_policy"] not in RADIUS_POLICY_CHOICES:
        raise ValueError(f"--radius-policy must be one of {RADIUS_POLICY_CHOICES}")
    if config["coverage_repair_policy"] not in COVERAGE_REPAIR_POLICY_CHOICES:
        raise ValueError(
            f"--coverage-repair-policy must be one of {COVERAGE_REPAIR_POLICY_CHOICES}"
        )
    if int(config["min_physical_coverage"]) < 1:
        raise ValueError("--min-physical-coverage must be >= 1")


def _batch_config_payload(batch_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_size": batch_config["batch_size"],
        "validation_batch_size": batch_config["validation_batch_size"],
        "prediction_batch_size": batch_config["prediction_batch_size"],
        "shuffle_train_batches": batch_config["shuffle_train_batches"],
        "drop_last": batch_config["drop_last"],
        "batch_plan": batch_config["batch_plan"],
        "batch_build_seed": batch_config["batch_build_seed"],
        "batching_mode": "mini_batch" if batch_config["batch_size"] is not None else "legacy_full_batch",
    }


def _optional_batch_size(value: int | None, flag_name: str) -> int | None:
    if value is None or int(value) == 0:
        return None
    if int(value) < 0:
        raise ValueError(f"--{flag_name} must be >= 1 or 0 for legacy full-batch")
    return int(value)


def _lr_for_epoch(epoch: int, epochs: int, config: dict[str, Any]) -> float:
    base_lr = float(config["lr"])
    schedule = config["lr_schedule"]
    if schedule == "constant":
        return base_lr
    if schedule == "two_stage":
        second_stage_epoch = int(config["second_stage_epoch"])
        if second_stage_epoch <= 0 or epoch <= second_stage_epoch:
            return base_lr
        return float(config["second_stage_lr"])
    if schedule == "second_stage":
        second_stage_epoch = int(config["second_stage_epoch"])
        if second_stage_epoch <= 0 or epoch <= second_stage_epoch:
            return base_lr
        return float(config["second_stage_lr"])
    if schedule == "warmup_cosine":
        warmup_epochs = int(config["warmup_epochs"])
        min_lr = float(config["min_lr"])
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            progress = epoch / warmup_epochs
            return min_lr + progress * (base_lr - min_lr)
        if warmup_epochs > 0:
            decay_epochs = max(epochs - warmup_epochs, 1)
            progress = min(max((epoch - warmup_epochs) / decay_epochs, 0.0), 1.0)
        else:
            decay_epochs = max(epochs - 1, 1)
            progress = min(max((epoch - 1) / decay_epochs, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine * (base_lr - min_lr)
    if schedule == "rapid_decay":
        min_lr = float(config["min_lr"])
        mid_lr = max(min_lr, base_lr * 0.1)
        if epoch <= 1:
            return base_lr
        if epoch <= 10:
            progress = min(max((epoch - 1) / 9.0, 0.0), 1.0)
            return base_lr + progress * (mid_lr - base_lr)
        progress = min(max((epoch - 10) / max(epochs - 10, 1), 0.0), 1.0)
        return mid_lr + progress * (min_lr - mid_lr)
    if schedule == "upstream_onecycle":
        progress = 0.0 if epochs <= 1 else min(max((epoch - 1) / (epochs - 1), 0.0), 1.0)
        pct_start = float(config["pct_start"])
        pct_final = float(config["pct_final"])
        pct_decay_end = max(1.0 - pct_final, pct_start)
        lr_init = float(config["lr_init"])
        lr_peak = float(config["lr_peak"])
        lr_base = float(config["lr_base"])
        lr_lowr = float(config["lr_lowr"])
        if progress <= pct_start:
            local = progress / max(pct_start, 1e-12)
            return lr_init + local * (lr_peak - lr_init)
        if progress <= pct_decay_end:
            local = (progress - pct_start) / max(pct_decay_end - pct_start, 1e-12)
            return lr_peak + local * (lr_base - lr_peak)
        local = (progress - pct_decay_end) / max(1.0 - pct_decay_end, 1e-12)
        return lr_base + local * (lr_lowr - lr_base)
    raise ValueError(f"Unsupported lr schedule: {schedule}")


def _loss_weight_keys() -> tuple[str, ...]:
    return (
        "background_l1_weight",
        "background_bias_weight",
        "background_over_weight",
        "background_relative_weight",
        "hotspot_weight",
        "strong_q_weight",
    )


def _scheduled_weight(config: dict[str, Any], key: str, epoch: int) -> float:
    static = float(config[key])
    schedule = config["loss_weight_schedule"]
    if schedule == "constant":
        return static

    transition_epoch = int(config["loss_transition_epoch"])
    start_value = config.get(f"{key}_start")
    end_value = config.get(f"{key}_end")
    start = static if start_value is None else float(start_value)
    end = static if end_value is None else float(end_value)

    if schedule == "two_phase":
        if epoch <= transition_epoch:
            return start
        return end

    if schedule == "linear_anneal":
        if transition_epoch <= 0:
            return static
        if epoch >= transition_epoch:
            return end
        if transition_epoch == 1:
            return end
        progress = (epoch - 1) / (transition_epoch - 1)
        return start + progress * (end - start)

    raise ValueError(f"Unsupported loss weight schedule: {schedule}")


def _loss_config_for_epoch(config: dict[str, Any], epoch: int) -> dict[str, Any]:
    current = dict(config)
    for key in _loss_weight_keys():
        value = _scheduled_weight(config, key, epoch)
        current[key] = value
        current[f"current_{key}"] = value
    return current


def _current_weight_payload(config: dict[str, Any]) -> dict[str, float]:
    return {f"current_{key}": float(config[key]) for key in _loss_weight_keys()}


def _sequence_summary(values) -> dict[str, float | int | None]:
    floats = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not floats:
        return {"count": 0, "first": None, "last": None, "min": None, "max": None}
    return {
        "count": len(floats),
        "first": floats[0],
        "last": floats[-1],
        "min": min(floats),
        "max": max(floats),
    }


def _epoch_monitor_summary(values: list[float]) -> dict[str, float | None]:
    floats = [float(value) for value in values if np.isfinite(float(value))]
    if not floats:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(floats)),
        "min": float(np.min(floats)),
        "max": float(np.max(floats)),
    }


def _selected_steps_or_empty(values: np.ndarray, report_every: int) -> list[tuple[int, float]]:
    if len(values) == 0:
        return []
    return _selected_steps(values, report_every)


def _history_field_summary(history: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    return _sequence_summary([item.get(field) for item in history])


def _loss_weight_schedule_payload(loss_config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "loss_weight_schedule",
        "loss_transition_epoch",
        "background_relative_weight_start",
        "background_relative_weight_end",
        "hotspot_weight_start",
        "hotspot_weight_end",
        "background_l1_weight_start",
        "background_l1_weight_end",
        "background_bias_weight_start",
        "background_bias_weight_end",
        "background_over_weight_start",
        "background_over_weight_end",
    ]
    return {key: loss_config.get(key) for key in keys}


def _masked_mean(values, mask):
    mask = mask.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _mask_fraction(mask, dtype):
    return jnp.mean(mask.astype(dtype))


def _samplewise_upper_quantile_mask(values, quantile: float):
    flat = values.reshape((values.shape[0], -1))
    thresholds = jnp.quantile(flat, quantile, axis=1, keepdims=True)
    return (flat >= thresholds).reshape(values.shape)


def _group_raw_condition_feature(group: dict, stats: dict, feature_name: str):
    feature_names = tuple(group.get("feature_names") or stats.get("feature_names") or ())
    if feature_name not in feature_names:
        return None
    inputs = group["inputs"]
    if inputs.c is None:
        return None
    feature_index = feature_names.index(feature_name)
    raw_c = recover_raw_condition(inputs.c, stats)
    return raw_c[..., feature_index : feature_index + 1]


def _samplewise_positive_upper_quantile_mask(values, quantile: float):
    flat = values.reshape((values.shape[0], -1))
    positive = flat > 0.0
    counts = jnp.sum(positive.astype(jnp.int32), axis=1)
    sorted_values = jnp.sort(jnp.where(positive, flat, -jnp.inf), axis=1)
    n_values = flat.shape[1]
    positive_start = n_values - counts
    positive_offset = jnp.floor((jnp.maximum(counts, 1) - 1) * quantile).astype(jnp.int32)
    threshold_index = jnp.clip(positive_start + positive_offset, 0, n_values - 1)
    thresholds = jnp.take_along_axis(sorted_values, threshold_index[:, None], axis=1)
    mask = jnp.logical_and(positive, flat >= thresholds)
    mask = jnp.logical_and(mask, counts[:, None] > 0)
    return mask.reshape(values.shape)


def _safe_relative_denominator(target_raw, loss_config: dict[str, Any]):
    abs_target = jnp.abs(target_raw)
    mode = loss_config["relative_floor_mode"]
    floor = jnp.asarray(loss_config["relative_floor"], dtype=target_raw.dtype)
    if mode == "fixed":
        safe_floor = floor
    elif mode == "p50":
        safe_floor = jnp.maximum(jnp.quantile(abs_target, 0.50), floor)
    elif mode == "p75":
        safe_floor = jnp.maximum(jnp.quantile(abs_target, 0.75), floor)
    else:
        raise ValueError(f"Unsupported relative floor mode: {mode}")
    return jnp.maximum(abs_target, safe_floor)


def _pseudo_negative_mask(target_raw, loss_config: dict[str, Any]):
    threshold = jnp.quantile(target_raw, loss_config["pseudo_negative_quantile"])
    mask = target_raw <= threshold
    delta_threshold = loss_config.get("pseudo_negative_delta_threshold")
    if delta_threshold is not None:
        mask = jnp.logical_and(mask, target_raw <= float(delta_threshold))
    return mask


def _pseudo_negative_unweighted_loss(pn_over, target_raw, pseudo_negative_mask, loss_config: dict[str, Any]):
    loss_type = loss_config["pseudo_negative_loss_type"]
    if loss_type == "mse":
        values = jnp.square(pn_over)
    elif loss_type == "l1":
        values = pn_over
    elif loss_type in {"relative_l1", "relative_mse"}:
        floor = jnp.asarray(loss_config["pseudo_negative_relative_floor"], dtype=target_raw.dtype)
        denom = jnp.maximum(jnp.abs(target_raw), floor)
        relative_over = pn_over / denom
        values = relative_over if loss_type == "relative_l1" else jnp.square(relative_over)
    else:
        raise ValueError(f"Unsupported pseudo-negative loss type: {loss_type}")
    return _masked_mean(values, pseudo_negative_mask)


def _sample_weights_for_group(group: dict) -> Any | None:
    weights = group.get("sample_weights")
    if weights is None:
        return None
    return jnp.asarray(weights, dtype=jnp.float32)


def _mean_axes_except_batch(values) -> tuple[int, ...]:
    return tuple(range(1, values.ndim))


def _sample_weighted_mean(values, sample_weights):
    if sample_weights is None:
        return jnp.mean(values)
    values = jnp.asarray(values)
    per_sample = jnp.mean(values, axis=_mean_axes_except_batch(values)) if values.ndim > 1 else values
    weights = sample_weights.astype(per_sample.dtype)
    return jnp.sum(per_sample * weights) / jnp.maximum(
        jnp.sum(weights),
        jnp.asarray(1.0e-12, dtype=per_sample.dtype),
    )


def _sample_weighted_masked_mean(values, mask, sample_weights):
    if sample_weights is None:
        return _masked_mean(values, mask)
    values = jnp.asarray(values)
    mask_values = mask.astype(values.dtype)
    axes = _mean_axes_except_batch(values)
    numerator = jnp.sum(values * mask_values, axis=axes) if axes else values * mask_values
    denominator = jnp.sum(mask_values, axis=axes) if axes else mask_values
    per_sample = jnp.where(
        denominator > 0.0,
        numerator / jnp.maximum(denominator, jnp.asarray(1.0e-12, dtype=values.dtype)),
        jnp.asarray(0.0, dtype=values.dtype),
    )
    active = (denominator > 0.0).astype(values.dtype)
    weights = sample_weights.astype(values.dtype) * active
    return jnp.sum(per_sample * weights) / jnp.maximum(
        jnp.sum(weights),
        jnp.asarray(1.0e-12, dtype=values.dtype),
    )


def _native_loss_components(
    model, params, groups: list[dict], stats: dict, loss_config: dict[str, Any]
) -> dict[str, Any]:
    names = (
        "shape_cv_loss", "log_scale_loss", "relative_field_loss",
        "raw_absolute_field_loss", "total_loss",
    )
    weighted = {name: jnp.asarray(0.0) for name in names}
    base_mse = jnp.asarray(0.0)
    count = 0
    native_weights = {
        "shape_cv": loss_config["native_shape_cv_weight"],
        "log_scale": loss_config["native_log_scale_weight"],
        "relative_field": loss_config["native_relative_field_weight"],
        "raw_absolute": loss_config["native_raw_field_weight"],
    }
    for group in groups:
        prediction = _model_apply(model, params, group)
        physics = group["native_physics"]
        components = native_shape_scale_losses(
            prediction,
            target_deltaT=group["target_delta_raw"],
            control_volumes=physics["control_volumes"],
            dirichlet_mask=physics["dirichlet_mask"],
            loss_weights=native_weights,
        )
        branch_mode = getattr(model, "native_branch_mode", "joint")
        if branch_mode == "scale_only":
            components = dict(components)
            components["total_loss"] = (
                native_weights["log_scale"] * components["log_scale_loss"]
                + native_weights["relative_field"] * components["relative_field_loss"]
                + native_weights["raw_absolute"] * components["raw_absolute_field_loss"]
            )
        elif branch_mode == "shape_only":
            components = dict(components)
            components["total_loss"] = (
                native_weights["shape_cv"] * components["shape_cv_loss"]
                + native_weights["relative_field"] * components["relative_field_loss"]
                + native_weights["raw_absolute"] * components["raw_absolute_field_loss"]
            )
        predicted_normalized = normalize_target_delta(prediction["deltaT_hat"], stats)
        group_base_mse = jnp.mean(jnp.square(predicted_normalized - group["target_normalized"]))
        n = int(group["target_normalized"].shape[0])
        for name in names:
            weighted[name] = weighted[name] + components[name] * n
        base_mse = base_mse + group_base_mse * n
        count += n
    divisor = max(count, 1)
    result = {name: value / divisor for name, value in weighted.items()}
    result["base_mse"] = base_mse / divisor
    zero = jnp.asarray(0.0, dtype=result["total_loss"].dtype)
    for name in (
        "background_penalty", "background_l1", "background_signed_bias_loss",
        "background_overprediction_loss", "background_relative_abs",
        "pseudo_negative_over_loss", "pseudo_negative_unweighted_loss",
        "pseudo_negative_weighted_loss", "pseudo_negative_weighted_fraction_of_total_loss",
        "pseudo_negative_bias", "pseudo_negative_over_ratio", "hotspot_retention_loss",
        "hotspot_mse", "strong_q_mse", "hotspot_mask_fraction", "strong_q_mask_fraction",
        "bg_pred_raw_mean", "bg_signed_bias", "bg_abs_mean", "hotspot_raw_mae",
        "pseudo_negative_count",
    ):
        result[name] = zero
    return result


def _loss_components(model, params, groups: list[dict], stats: dict, loss_config: dict[str, Any]) -> dict[str, Any]:
    if groups and "native_physics" in groups[0]:
        return _native_loss_components(model, params, groups, stats, loss_config)
    weighted = {
        "base_mse": 0.0,
        "background_penalty": 0.0,
        "background_l1": 0.0,
        "background_signed_bias_loss": 0.0,
        "background_overprediction_loss": 0.0,
        "background_relative_abs": 0.0,
        "pseudo_negative_over_loss": 0.0,
        "pseudo_negative_unweighted_loss": 0.0,
        "pseudo_negative_weighted_loss": 0.0,
        "pseudo_negative_weighted_fraction_of_total_loss": 0.0,
        "pseudo_negative_bias": 0.0,
        "pseudo_negative_over_ratio": 0.0,
        "hotspot_retention_loss": 0.0,
        "hotspot_mse": 0.0,
        "strong_q_mse": 0.0,
        "hotspot_mask_fraction": 0.0,
        "strong_q_mask_fraction": 0.0,
        "total_loss": 0.0,
        "bg_pred_raw_mean": 0.0,
        "bg_signed_bias": 0.0,
        "bg_abs_mean": 0.0,
        "hotspot_raw_mae": 0.0,
    }
    count = 0
    pseudo_negative_count = jnp.asarray(0.0)
    for group in groups:
        sample_weights = _sample_weights_for_group(group)
        pred = _model_apply(model, params, group)
        target = group["target_normalized"]
        target_raw = group["target_delta_raw"]
        pred_raw_delta = _normalized_delta_to_raw(pred, stats)
        base_mse = _sample_weighted_mean(jnp.square(pred - target), sample_weights)
        background_threshold = jnp.quantile(target_raw, loss_config["background_quantile"])
        hotspot_threshold = jnp.quantile(target_raw, loss_config["hotspot_quantile"])
        background_mask = target_raw <= background_threshold
        hotspot_mask = target_raw >= hotspot_threshold
        background_penalty = jnp.asarray(0.0, dtype=base_mse.dtype)
        background_l1 = jnp.asarray(0.0, dtype=base_mse.dtype)
        background_signed_bias_loss = jnp.asarray(0.0, dtype=base_mse.dtype)
        background_overprediction_loss = jnp.asarray(0.0, dtype=base_mse.dtype)
        background_relative_abs = jnp.asarray(0.0, dtype=base_mse.dtype)
        pseudo_negative_over_loss = jnp.asarray(0.0, dtype=base_mse.dtype)
        pseudo_negative_unweighted_loss = jnp.asarray(0.0, dtype=base_mse.dtype)
        pseudo_negative_weighted_loss = jnp.asarray(0.0, dtype=base_mse.dtype)
        pseudo_negative_weighted_fraction = jnp.asarray(0.0, dtype=base_mse.dtype)
        pseudo_negative_bias = jnp.asarray(0.0, dtype=base_mse.dtype)
        pseudo_negative_over_ratio = jnp.asarray(0.0, dtype=base_mse.dtype)
        hotspot_retention_loss = jnp.asarray(0.0, dtype=base_mse.dtype)
        hotspot_mse = jnp.asarray(0.0, dtype=base_mse.dtype)
        strong_q_mse = jnp.asarray(0.0, dtype=base_mse.dtype)
        hotspot_mask_fraction = jnp.asarray(0.0, dtype=base_mse.dtype)
        strong_q_mask_fraction = jnp.asarray(0.0, dtype=base_mse.dtype)
        raw_error = pred_raw_delta - target_raw
        if loss_config["loss_mode"] == "background_hotspot":
            background_penalty = _sample_weighted_masked_mean(jnp.square(pred_raw_delta), background_mask, sample_weights)
            hotspot_retention_loss = _sample_weighted_masked_mean(jnp.square(pred - target), hotspot_mask, sample_weights)
            total_loss = (
                base_mse
                + loss_config["background_weight"] * background_penalty
                + loss_config["hotspot_weight"] * hotspot_retention_loss
            )
        elif loss_config["loss_mode"] in {"background_l1_bias", "background_l1_relative", "background_pseudo_negative"}:
            background_l1 = _sample_weighted_masked_mean(jnp.abs(pred_raw_delta), background_mask, sample_weights)
            background_signed_bias_loss = jnp.abs(_sample_weighted_masked_mean(raw_error, background_mask, sample_weights))
            background_overprediction_loss = _sample_weighted_masked_mean(
                jnp.maximum(raw_error, 0.0), background_mask, sample_weights
            )
            hotspot_retention_loss = _sample_weighted_masked_mean(jnp.square(pred - target), hotspot_mask, sample_weights)
            if loss_config["loss_mode"] in {"background_l1_relative", "background_pseudo_negative"}:
                denom = _safe_relative_denominator(target_raw, loss_config)
                background_relative_abs = _sample_weighted_masked_mean(
                    jnp.abs(raw_error) / denom, background_mask, sample_weights
                )
            if loss_config["loss_mode"] == "background_pseudo_negative":
                pseudo_negative_mask = _pseudo_negative_mask(target_raw, loss_config)
                pn_count = jnp.sum(pseudo_negative_mask.astype(base_mse.dtype))
                enough_points = pn_count >= float(loss_config["pseudo_negative_min_count"])
                pn_over = jnp.maximum(raw_error - loss_config["pseudo_negative_over_margin"], 0.0)
                pseudo_negative_unweighted_loss = jnp.where(
                    enough_points,
                    _pseudo_negative_unweighted_loss(pn_over, target_raw, pseudo_negative_mask, loss_config),
                    jnp.asarray(0.0, dtype=base_mse.dtype),
                )
                pseudo_negative_over_loss = pseudo_negative_unweighted_loss
                pseudo_negative_weighted_loss = loss_config["pseudo_negative_weight"] * pseudo_negative_unweighted_loss
                pseudo_negative_bias = jnp.where(
                    enough_points,
                    _sample_weighted_masked_mean(raw_error, pseudo_negative_mask, sample_weights),
                    jnp.asarray(0.0, dtype=base_mse.dtype),
                )
                pseudo_negative_over_ratio = jnp.where(
                    enough_points,
                    _sample_weighted_masked_mean(
                        (raw_error > loss_config["pseudo_negative_over_margin"]).astype(base_mse.dtype),
                        pseudo_negative_mask,
                        sample_weights,
                    ),
                    jnp.asarray(0.0, dtype=base_mse.dtype),
                )
                pseudo_negative_count = pseudo_negative_count + pn_count
            total_loss = (
                base_mse
                + loss_config["background_l1_weight"] * background_l1
                + loss_config["background_bias_weight"] * background_signed_bias_loss
                + loss_config["background_over_weight"] * background_overprediction_loss
                + loss_config["background_relative_weight"] * background_relative_abs
                + pseudo_negative_weighted_loss
                + loss_config["hotspot_weight"] * hotspot_retention_loss
            )
            pseudo_negative_weighted_fraction = pseudo_negative_weighted_loss / jnp.maximum(
                jnp.abs(total_loss), jnp.asarray(1.0e-12, dtype=base_mse.dtype)
            )
        elif loss_config["loss_mode"] == "hotspot_strong_q":
            hotspot_top_mask = _samplewise_upper_quantile_mask(
                target_raw,
                float(loss_config["hotspot_quantile"]),
            )
            q_values = _group_raw_condition_feature(group, stats, "q")
            if q_values is None:
                strong_q_mask = jnp.zeros_like(target_raw, dtype=bool)
            else:
                strong_q_mask = _samplewise_positive_upper_quantile_mask(
                    q_values,
                    float(loss_config["strong_q_quantile"]),
                )
            point_mse = jnp.square(pred - target)
            hotspot_mse = _sample_weighted_masked_mean(point_mse, hotspot_top_mask, sample_weights)
            strong_q_mse = _sample_weighted_masked_mean(point_mse, strong_q_mask, sample_weights)
            hotspot_mask_fraction = (
                _sample_weighted_mean(hotspot_top_mask.astype(base_mse.dtype), sample_weights)
                if sample_weights is not None
                else _mask_fraction(hotspot_top_mask, base_mse.dtype)
            )
            strong_q_mask_fraction = (
                _sample_weighted_mean(strong_q_mask.astype(base_mse.dtype), sample_weights)
                if sample_weights is not None
                else _mask_fraction(strong_q_mask, base_mse.dtype)
            )
            total_loss = (
                base_mse
                + loss_config["hotspot_weight"] * hotspot_mse
                + loss_config["strong_q_weight"] * strong_q_mse
            )
        else:
            total_loss = base_mse
        bg_pred_raw_mean = _sample_weighted_masked_mean(pred_raw_delta, background_mask, sample_weights)
        bg_signed_bias = _sample_weighted_masked_mean(raw_error, background_mask, sample_weights)
        bg_abs_mean = _sample_weighted_masked_mean(jnp.abs(raw_error), background_mask, sample_weights)
        hotspot_raw_mae = _sample_weighted_masked_mean(jnp.abs(raw_error), hotspot_mask, sample_weights)
        n = target.shape[0]
        group_weight = (
            jnp.sum(sample_weights.astype(base_mse.dtype))
            if sample_weights is not None
            else jnp.asarray(n, dtype=base_mse.dtype)
        )
        weighted["base_mse"] = weighted["base_mse"] + base_mse * group_weight
        weighted["background_penalty"] = weighted["background_penalty"] + background_penalty * group_weight
        weighted["background_l1"] = weighted["background_l1"] + background_l1 * group_weight
        weighted["background_signed_bias_loss"] = (
            weighted["background_signed_bias_loss"] + background_signed_bias_loss * group_weight
        )
        weighted["background_overprediction_loss"] = (
            weighted["background_overprediction_loss"] + background_overprediction_loss * group_weight
        )
        weighted["background_relative_abs"] = weighted["background_relative_abs"] + background_relative_abs * group_weight
        weighted["pseudo_negative_over_loss"] = weighted["pseudo_negative_over_loss"] + pseudo_negative_over_loss * group_weight
        weighted["pseudo_negative_unweighted_loss"] = (
            weighted["pseudo_negative_unweighted_loss"] + pseudo_negative_unweighted_loss * group_weight
        )
        weighted["pseudo_negative_weighted_loss"] = (
            weighted["pseudo_negative_weighted_loss"] + pseudo_negative_weighted_loss * group_weight
        )
        weighted["pseudo_negative_weighted_fraction_of_total_loss"] = (
            weighted["pseudo_negative_weighted_fraction_of_total_loss"] + pseudo_negative_weighted_fraction * group_weight
        )
        weighted["pseudo_negative_bias"] = weighted["pseudo_negative_bias"] + pseudo_negative_bias * group_weight
        weighted["pseudo_negative_over_ratio"] = weighted["pseudo_negative_over_ratio"] + pseudo_negative_over_ratio * group_weight
        weighted["hotspot_retention_loss"] = weighted["hotspot_retention_loss"] + hotspot_retention_loss * group_weight
        weighted["hotspot_mse"] = weighted["hotspot_mse"] + hotspot_mse * group_weight
        weighted["strong_q_mse"] = weighted["strong_q_mse"] + strong_q_mse * group_weight
        weighted["hotspot_mask_fraction"] = weighted["hotspot_mask_fraction"] + hotspot_mask_fraction * group_weight
        weighted["strong_q_mask_fraction"] = weighted["strong_q_mask_fraction"] + strong_q_mask_fraction * group_weight
        weighted["total_loss"] = weighted["total_loss"] + total_loss * group_weight
        weighted["bg_pred_raw_mean"] = weighted["bg_pred_raw_mean"] + bg_pred_raw_mean * group_weight
        weighted["bg_signed_bias"] = weighted["bg_signed_bias"] + bg_signed_bias * group_weight
        weighted["bg_abs_mean"] = weighted["bg_abs_mean"] + bg_abs_mean * group_weight
        weighted["hotspot_raw_mae"] = weighted["hotspot_raw_mae"] + hotspot_raw_mae * group_weight
        count += int(n) if sample_weights is None else float(np.sum(np.asarray(group["sample_weights"], dtype=np.float64)))
    divisor = max(count, 1)
    result = {key: value / divisor for key, value in weighted.items()}
    result["pseudo_negative_count"] = pseudo_negative_count
    return result


def _loss_components_payload(components: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in components.items()}


def _weighted_metrics(model, params, groups: list[dict], stats: dict) -> dict[str, Any]:
    weighted_normalized_loss = 0.0
    weighted_raw_delta_mse = 0.0
    weighted_recovered_mse = 0.0
    weighted_mean_abs_true_delta = 0.0
    weighted_mean_square_true_delta = 0.0
    native_metric_sums: dict[str, Any] = {}
    count = 0
    finite_ok = True
    shape_ok = True
    for group in groups:
        prediction = _model_apply(model, params, group)
        if isinstance(prediction, Mapping):
            pred_delta = prediction["deltaT_hat"]
            pred_normalized = normalize_target_delta(pred_delta, stats)
            recovered = prediction["raw_temperature"]
            physics = group["native_physics"]
            native_diagnostics = native_shape_scale_diagnostics(
                prediction,
                target_deltaT=group["target_delta_raw"],
                control_volumes=physics["control_volumes"],
                dirichlet_mask=physics["dirichlet_mask"],
                s_phys=jnp.exp(physics["log_s_phys"]),
            )
            native_values = {
                "scale_log_abs_error": native_diagnostics["scale_log_abs_error"],
                "shape_cv_rmse": native_diagnostics["shape_cv_rmse"],
            }
            for field_name, metrics in native_diagnostics["metrics"].items():
                for metric_name, value in metrics.items():
                    native_values[f"{field_name}_{metric_name}"] = value
            for name, value in native_values.items():
                native_metric_sums[name] = native_metric_sums.get(name, 0.0) + value * prediction["deltaT_hat"].shape[0]
        else:
            pred_normalized = prediction
            pred_delta = _normalized_delta_to_raw(pred_normalized, stats)
            recovered = recover_temperature_from_normalized_delta(pred_normalized, group["t_ref"], stats)
        n = pred_normalized.shape[0]
        weighted_normalized_loss = weighted_normalized_loss + jnp.mean(
            jnp.square(pred_normalized - group["target_normalized"])
        ) * n
        weighted_raw_delta_mse = weighted_raw_delta_mse + jnp.mean(
            jnp.square(pred_delta - group["target_delta_raw"])
        ) * n
        weighted_recovered_mse = weighted_recovered_mse + jnp.mean(
            jnp.square(recovered - group["target_temperature"])
        ) * n
        weighted_mean_abs_true_delta = weighted_mean_abs_true_delta + jnp.mean(
            jnp.abs(group["target_delta_raw"])
        ) * n
        weighted_mean_square_true_delta = weighted_mean_square_true_delta + jnp.mean(
            jnp.square(group["target_delta_raw"])
        ) * n
        finite_ok = (
            finite_ok
            and bool(jnp.all(jnp.isfinite(pred_normalized)))
            and bool(jnp.all(jnp.isfinite(pred_delta)))
            and bool(jnp.all(jnp.isfinite(recovered)))
        )
        shape_ok = shape_ok and pred_normalized.shape == group["target_normalized"].shape
        count += int(n)

    divisor = max(count, 1)
    mean_abs_true_delta = float(weighted_mean_abs_true_delta / divisor)
    mean_square_true_delta = float(weighted_mean_square_true_delta / divisor)
    raw_delta_mse = float(weighted_raw_delta_mse / divisor)
    recovered_temperature_mse = float(weighted_recovered_mse / divisor)
    relative_pct = _deltaT_error_pct(raw_delta_mse, mean_square_true_delta)
    result = {
        "normalized_loss": float(weighted_normalized_loss / divisor),
        "raw_delta_mse": raw_delta_mse,
        "raw_rmse_K": _rmse_from_mse(raw_delta_mse),
        "recovered_temperature_mse": recovered_temperature_mse,
        "recovered_T_rmse_K": _rmse_from_mse(recovered_temperature_mse),
        "mean_abs_true_deltaT": mean_abs_true_delta,
        "mean_square_true_deltaT": mean_square_true_delta,
        "rel_rmse_v4_pct": relative_pct,
        "raw_deltaT_relative_rmse_pct_v4": relative_pct,
        "raw_deltaT_relative_rmse_pct": relative_pct,
        "finite_ok": finite_ok,
        "shape_ok": shape_ok,
    }
    result.update({name: float(value / divisor) for name, value in native_metric_sums.items()})
    return result


def _legacy_full_batch_metrics_with_true_rms_denominator(
    model, params, groups: list[dict], stats: dict
) -> dict[str, Any]:
    metrics = dict(_metrics(model, params, groups, stats))
    weighted_mean_square = 0.0
    count = 0
    for group in groups:
        n = int(group["target_delta_raw"].shape[0])
        weighted_mean_square += float(jnp.mean(jnp.square(group["target_delta_raw"]))) * n
        count += n
    mean_square_true_delta = weighted_mean_square / max(count, 1)
    metrics["mean_square_true_deltaT"] = mean_square_true_delta
    relative_pct = _deltaT_error_pct(
        metrics.get("raw_delta_mse"), mean_square_true_delta
    )
    metrics["rel_rmse_v4_pct"] = relative_pct
    metrics["raw_deltaT_relative_rmse_pct_v4"] = relative_pct
    metrics["raw_deltaT_relative_rmse_pct"] = relative_pct
    return metrics


def _optax_learning_rate_schedule(epochs: int, lr_config: dict[str, Any]):
    schedule = lr_config["lr_schedule"]
    base_lr = float(lr_config["lr"])
    updates_per_epoch = max(int(lr_config.get("updates_per_epoch", 1)), 1)

    if schedule == "constant":
        return base_lr

    def learning_rate(count):
        update_count = jnp.asarray(count, dtype=jnp.float32)
        epoch = jnp.floor(update_count / float(updates_per_epoch)) + 1.0
        base = jnp.asarray(base_lr, dtype=jnp.float32)
        if schedule == "two_stage":
            second_stage_epoch = int(lr_config["second_stage_epoch"])
            if second_stage_epoch <= 0:
                return base
            second_lr = jnp.asarray(float(lr_config["second_stage_lr"]), dtype=jnp.float32)
            return jnp.where(epoch <= float(second_stage_epoch), base, second_lr)

        if schedule == "second_stage":
            second_stage_epoch = int(lr_config["second_stage_epoch"])
            if second_stage_epoch <= 0:
                return base
            second_lr = jnp.asarray(float(lr_config["second_stage_lr"]), dtype=jnp.float32)
            return jnp.where(epoch <= float(second_stage_epoch), base, second_lr)

        if schedule == "warmup_cosine":
            min_lr = jnp.asarray(float(lr_config["min_lr"]), dtype=jnp.float32)
            warmup_epochs = int(lr_config["warmup_epochs"])
            if warmup_epochs > 0:
                warmup_progress = jnp.clip(epoch / float(warmup_epochs), 0.0, 1.0)
                warmup_lr = min_lr + warmup_progress * (base - min_lr)
                decay_epochs = max(epochs - warmup_epochs, 1)
                decay_progress = jnp.clip((epoch - float(warmup_epochs)) / float(decay_epochs), 0.0, 1.0)
                cosine_lr = min_lr + 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress)) * (base - min_lr)
                return jnp.where(epoch <= float(warmup_epochs), warmup_lr, cosine_lr)

            decay_epochs = max(epochs - 1, 1)
            decay_progress = jnp.clip((epoch - 1.0) / float(decay_epochs), 0.0, 1.0)
            return min_lr + 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress)) * (base - min_lr)

        if schedule == "rapid_decay":
            min_lr = jnp.asarray(float(lr_config["min_lr"]), dtype=jnp.float32)
            mid_lr = jnp.maximum(min_lr, base * jnp.asarray(0.1, dtype=jnp.float32))
            early_progress = jnp.clip((epoch - 1.0) / 9.0, 0.0, 1.0)
            early_lr = base + early_progress * (mid_lr - base)
            late_progress = jnp.clip((epoch - 10.0) / float(max(epochs - 10, 1)), 0.0, 1.0)
            late_lr = mid_lr + late_progress * (min_lr - mid_lr)
            return jnp.where(epoch <= 10.0, early_lr, late_lr)

        if schedule == "upstream_onecycle":
            total_updates = max(epochs * updates_per_epoch - 1, 1)
            progress = jnp.clip(update_count / float(total_updates), 0.0, 1.0)
            pct_start = float(lr_config["pct_start"])
            pct_final = float(lr_config["pct_final"])
            pct_decay_end = max(1.0 - pct_final, pct_start)
            lr_init = jnp.asarray(float(lr_config["lr_init"]), dtype=jnp.float32)
            lr_peak = jnp.asarray(float(lr_config["lr_peak"]), dtype=jnp.float32)
            lr_base = jnp.asarray(float(lr_config["lr_base"]), dtype=jnp.float32)
            lr_lowr = jnp.asarray(float(lr_config["lr_lowr"]), dtype=jnp.float32)
            warmup_progress = jnp.clip(progress / max(pct_start, 1e-12), 0.0, 1.0)
            warmup_lr = lr_init + warmup_progress * (lr_peak - lr_init)
            decay_progress = jnp.clip(
                (progress - pct_start) / max(pct_decay_end - pct_start, 1e-12),
                0.0,
                1.0,
            )
            decay_lr = lr_peak + decay_progress * (lr_base - lr_peak)
            final_progress = jnp.clip(
                (progress - pct_decay_end) / max(1.0 - pct_decay_end, 1e-12),
                0.0,
                1.0,
            )
            final_lr = lr_base + final_progress * (lr_lowr - lr_base)
            return jnp.where(
                progress <= pct_start,
                warmup_lr,
                jnp.where(progress <= pct_decay_end, decay_lr, final_lr),
            )

        raise ValueError(f"Unsupported lr schedule: {schedule}")

    return learning_rate


def _build_optax_state(
    params,
    *,
    epochs: int,
    lr_config: dict[str, Any],
    optimizer_config: dict[str, Any],
):
    optimizer_name = optimizer_config["optimizer"]
    if optimizer_name == "manual_gd":
        return None

    try:
        import optax
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("--optimizer adam/adamw requires optax") from exc

    transforms = []
    gradient_clip_norm = optimizer_config.get("gradient_clip_norm")
    if gradient_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(float(gradient_clip_norm)))

    learning_rate = _optax_learning_rate_schedule(epochs, lr_config)
    weight_decay = float(optimizer_config["weight_decay"])
    if optimizer_name == "adam":
        if weight_decay > 0.0:
            transforms.append(optax.add_decayed_weights(weight_decay))
        transforms.append(optax.adam(learning_rate=learning_rate))
    elif optimizer_name == "adamw":
        transforms.append(optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    tx = optax.chain(*transforms)
    return {
        "tx": tx,
        "state": tx.init(params),
        "apply_updates": optax.apply_updates,
    }


def _copy_params(params):
    return tree.tree_map(lambda value: value.copy() if hasattr(value, "copy") else value, params)


def _dummy_inputs_like(inputs: Inputs) -> Inputs:
    return Inputs(
        u=jnp.zeros_like(inputs.u),
        c=None if inputs.c is None else jnp.zeros_like(inputs.c),
        x_inp=jnp.zeros_like(inputs.x_inp),
        x_out=jnp.zeros_like(inputs.x_out),
        t=None if inputs.t is None else jnp.zeros_like(inputs.t),
        tau=None if inputs.tau is None else jnp.zeros_like(inputs.tau),
    )


def _init_inputs_for_mode(group: dict[str, Any], init_mode: str) -> Inputs:
    if init_mode == "real_first_batch":
        return group["inputs"]
    if init_mode == "upstream_dummy":
        return _dummy_inputs_like(group["inputs"])
    raise ValueError(f"Unsupported init mode: {init_mode}")


def _best_selection_payload(
    result: dict[str, Any],
    *,
    best_predictions_path: Path | None,
    best_predictions_saved: bool,
) -> dict[str, Any]:
    best_record = result.get("best_record") or {}
    return {
        "selection_metric": result.get("selection_metric"),
        "primary_validation_split": result.get("primary_validation_split"),
        "stress_validation_split": result.get("stress_validation_split"),
        "best_epoch": best_record.get("epoch"),
        "best_valid_loss": best_record.get("valid_loss"),
        "best_valid_iid_loss": best_record.get("valid_iid_loss"),
        "best_valid_stress_loss": best_record.get("valid_stress_loss"),
        "best_valid_raw_deltaT_mse": best_record.get("valid_raw_deltaT_mse"),
        "best_valid_iid_raw_deltaT_mse": best_record.get("valid_iid_raw_deltaT_mse"),
        "best_valid_stress_raw_deltaT_mse": best_record.get("valid_stress_raw_deltaT_mse"),
        "best_valid_raw_deltaT_rmse_K": best_record.get("valid_raw_deltaT_rmse_K"),
        "best_valid_iid_raw_deltaT_rmse_K": best_record.get("valid_iid_raw_deltaT_rmse_K"),
        "best_valid_stress_raw_deltaT_rmse_K": best_record.get("valid_stress_raw_deltaT_rmse_K"),
        "best_valid_recovered_T_mse": best_record.get("valid_recovered_T_mse"),
        "best_valid_iid_recovered_T_mse": best_record.get("valid_iid_recovered_T_mse"),
        "best_valid_stress_recovered_T_mse": best_record.get("valid_stress_recovered_T_mse"),
        "best_valid_recovered_T_rmse_K": best_record.get("valid_recovered_T_rmse_K"),
        "best_valid_iid_recovered_T_rmse_K": best_record.get("valid_iid_recovered_T_rmse_K"),
        "best_valid_stress_recovered_T_rmse_K": best_record.get("valid_stress_recovered_T_rmse_K"),
        "best_valid_relative_rmse_pct_v4": best_record.get("valid_rel_rmse_v4_pct"),
        "best_valid_iid_relative_rmse_pct_v4": best_record.get("valid_iid_rel_rmse_v4_pct"),
        "best_valid_stress_relative_rmse_pct_v4": best_record.get("valid_stress_rel_rmse_v4_pct"),
        "best_relative_metric_denominator_mean_abs_true_deltaT": best_record.get(
            "valid_relative_metric_denominator_mean_abs_true_deltaT"
        ),
        "best_valid_base_mse": best_record.get("valid_base_mse"),
        "best_valid_iid_base_mse": best_record.get("valid_iid_base_mse"),
        "best_valid_stress_base_mse": best_record.get("valid_stress_base_mse"),
        "final_epoch": result.get("final_epoch"),
        "final_valid_loss": result.get("final_valid_loss"),
        "final_valid_iid_loss": result.get("final_valid_iid_loss"),
        "final_valid_stress_loss": result.get("final_valid_stress_loss"),
        "final_valid_raw_deltaT_mse": result.get("valid_metrics", {}).get("raw_delta_mse"),
        "final_valid_iid_raw_deltaT_mse": result.get("valid_metrics", {}).get("raw_delta_mse"),
        "final_valid_stress_raw_deltaT_mse": result.get("final_valid_stress_raw_deltaT_mse"),
        "final_valid_raw_deltaT_rmse_K": result.get("valid_metrics", {}).get("raw_rmse_K"),
        "final_valid_iid_raw_deltaT_rmse_K": result.get("valid_metrics", {}).get("raw_rmse_K"),
        "final_valid_stress_raw_deltaT_rmse_K": (
            result.get("valid_stress_metrics", {}) or {}
        ).get("raw_rmse_K"),
        "final_valid_recovered_T_mse": result.get("valid_metrics", {}).get("recovered_temperature_mse"),
        "final_valid_iid_recovered_T_mse": result.get("valid_metrics", {}).get("recovered_temperature_mse"),
        "final_valid_stress_recovered_T_mse": (
            result.get("valid_stress_metrics", {}) or {}
        ).get("recovered_temperature_mse"),
        "final_valid_recovered_T_rmse_K": result.get("valid_metrics", {}).get("recovered_T_rmse_K"),
        "final_valid_iid_recovered_T_rmse_K": result.get("valid_metrics", {}).get("recovered_T_rmse_K"),
        "final_valid_stress_recovered_T_rmse_K": (
            result.get("valid_stress_metrics", {}) or {}
        ).get("recovered_T_rmse_K"),
        "final_valid_relative_rmse_pct_v4": result.get("valid_metrics", {}).get("rel_rmse_v4_pct"),
        "final_valid_iid_relative_rmse_pct_v4": result.get("valid_metrics", {}).get("rel_rmse_v4_pct"),
        "final_valid_stress_relative_rmse_pct_v4": (
            result.get("valid_stress_metrics", {}) or {}
        ).get("rel_rmse_v4_pct"),
        "final_relative_metric_denominator_mean_abs_true_deltaT": result.get("valid_metrics", {}).get(
            "mean_abs_true_deltaT"
        ),
        "final_valid_base_mse": result.get("final_valid_loss_components", {}).get("base_mse"),
        "final_valid_iid_base_mse": (
            result.get("final_valid_iid_loss_components", {}) or {}
        ).get("base_mse"),
        "final_valid_stress_base_mse": (
            result.get("final_valid_stress_loss_components", {}) or {}
        ).get("base_mse"),
        "best_predictions_saved": bool(best_predictions_saved),
        "best_predictions_path": str(best_predictions_path) if best_predictions_path is not None else None,
    }


def _epoch_history_record(
    epoch: int,
    lr_epoch: float,
    current_loss_config: dict[str, Any],
    train_components: dict[str, Any] | None,
    valid_components: dict[str, Any],
    valid_metrics: dict[str, Any],
    train_metrics: dict[str, Any] | None,
    *,
    primary_validation_split: str = "valid",
    valid_stress_components: dict[str, Any] | None = None,
    valid_stress_metrics: dict[str, Any] | None = None,
    stress_validation_split: str | None = None,
) -> dict[str, Any]:
    record = {
        "epoch": int(epoch),
        "lr": float(lr_epoch),
        "primary_validation_split": primary_validation_split,
        "train_loss": _maybe_float(train_components, "total_loss"),
        "valid_loss": float(valid_components["total_loss"]),
        "valid_iid_loss": float(valid_components["total_loss"]) if primary_validation_split == "valid_iid" else None,
        "train_base_mse": _maybe_float(train_components, "base_mse"),
        "valid_base_mse": float(valid_components["base_mse"]),
        "valid_iid_base_mse": float(valid_components["base_mse"]) if primary_validation_split == "valid_iid" else None,
        "train_background_penalty": _maybe_float(train_components, "background_penalty"),
        "valid_background_penalty": float(valid_components["background_penalty"]),
        "train_background_l1": _maybe_float(train_components, "background_l1"),
        "valid_background_l1": float(valid_components["background_l1"]),
        "train_background_signed_bias_loss": _maybe_float(train_components, "background_signed_bias_loss"),
        "valid_background_signed_bias_loss": float(valid_components["background_signed_bias_loss"]),
        "train_background_overprediction_loss": _maybe_float(train_components, "background_overprediction_loss"),
        "valid_background_overprediction_loss": float(valid_components["background_overprediction_loss"]),
        "train_background_relative_abs": _maybe_float(train_components, "background_relative_abs"),
        "valid_background_relative_abs": float(valid_components["background_relative_abs"]),
        "train_pseudo_negative_count": _maybe_float(train_components, "pseudo_negative_count"),
        "valid_pseudo_negative_count": float(valid_components["pseudo_negative_count"]),
        "train_pseudo_negative_over_loss": _maybe_float(train_components, "pseudo_negative_over_loss"),
        "valid_pseudo_negative_over_loss": float(valid_components["pseudo_negative_over_loss"]),
        "train_pseudo_negative_unweighted_loss": _maybe_float(train_components, "pseudo_negative_unweighted_loss"),
        "valid_pseudo_negative_unweighted_loss": float(valid_components["pseudo_negative_unweighted_loss"]),
        "train_pseudo_negative_weighted_loss": _maybe_float(train_components, "pseudo_negative_weighted_loss"),
        "valid_pseudo_negative_weighted_loss": float(valid_components["pseudo_negative_weighted_loss"]),
        "train_pseudo_negative_weighted_fraction_of_total_loss": _maybe_float(
            train_components,
            "pseudo_negative_weighted_fraction_of_total_loss",
        ),
        "valid_pseudo_negative_weighted_fraction_of_total_loss": float(
            valid_components["pseudo_negative_weighted_fraction_of_total_loss"]
        ),
        "train_pseudo_negative_bias": _maybe_float(train_components, "pseudo_negative_bias"),
        "valid_pseudo_negative_bias": float(valid_components["pseudo_negative_bias"]),
        "train_pseudo_negative_over_ratio": _maybe_float(train_components, "pseudo_negative_over_ratio"),
        "valid_pseudo_negative_over_ratio": float(valid_components["pseudo_negative_over_ratio"]),
        "valid_pn_bias": float(valid_components["pseudo_negative_bias"]),
        "valid_pn_over": float(valid_components["pseudo_negative_over_loss"]),
        "valid_pn_over_ratio": float(valid_components["pseudo_negative_over_ratio"]),
        "train_hotspot_retention_loss": _maybe_float(train_components, "hotspot_retention_loss"),
        "valid_hotspot_retention_loss": float(valid_components["hotspot_retention_loss"]),
        "train_hotspot_mse": _maybe_float(train_components, "hotspot_mse"),
        "valid_hotspot_mse": float(valid_components["hotspot_mse"]),
        "train_strong_q_mse": _maybe_float(train_components, "strong_q_mse"),
        "valid_strong_q_mse": float(valid_components["strong_q_mse"]),
        "train_hotspot_mask_fraction": _maybe_float(train_components, "hotspot_mask_fraction"),
        "valid_hotspot_mask_fraction": float(valid_components["hotspot_mask_fraction"]),
        "train_strong_q_mask_fraction": _maybe_float(train_components, "strong_q_mask_fraction"),
        "valid_strong_q_mask_fraction": float(valid_components["strong_q_mask_fraction"]),
        "train_bg_pred_raw_mean": _maybe_float(train_components, "bg_pred_raw_mean"),
        "valid_bg_pred_raw_mean": float(valid_components["bg_pred_raw_mean"]),
        "train_bg_signed_bias": _maybe_float(train_components, "bg_signed_bias"),
        "valid_bg_signed_bias": float(valid_components["bg_signed_bias"]),
        "train_bg_abs_mean": _maybe_float(train_components, "bg_abs_mean"),
        "valid_bg_abs_mean": float(valid_components["bg_abs_mean"]),
        "train_hotspot_raw_mae": _maybe_float(train_components, "hotspot_raw_mae"),
        "valid_hotspot_raw_mae": float(valid_components["hotspot_raw_mae"]),
        "train_raw_deltaT_mse": _maybe_float(train_metrics, "raw_delta_mse"),
        "valid_raw_deltaT_mse": float(valid_metrics["raw_delta_mse"]),
        "valid_iid_raw_deltaT_mse": float(valid_metrics["raw_delta_mse"]) if primary_validation_split == "valid_iid" else None,
        "train_raw_rmse_K": _maybe_float(train_metrics, "raw_rmse_K"),
        "valid_raw_rmse_K": float(valid_metrics["raw_rmse_K"]),
        "valid_iid_raw_rmse_K": float(valid_metrics["raw_rmse_K"]) if primary_validation_split == "valid_iid" else None,
        "train_raw_deltaT_rmse_K": _maybe_float(train_metrics, "raw_rmse_K"),
        "valid_raw_deltaT_rmse_K": float(valid_metrics["raw_rmse_K"]),
        "valid_iid_raw_deltaT_rmse_K": (
            float(valid_metrics["raw_rmse_K"]) if primary_validation_split == "valid_iid" else None
        ),
        "train_error_pct": _metric_error_pct(train_metrics),
        "valid_error_pct": _metric_error_pct(valid_metrics),
        "valid_iid_error_pct": _metric_error_pct(valid_metrics) if primary_validation_split == "valid_iid" else None,
        "train_rel_rmse_v4_pct": _metric_error_pct(train_metrics),
        "valid_rel_rmse_v4_pct": _metric_error_pct(valid_metrics),
        "valid_iid_rel_rmse_v4_pct": _metric_error_pct(valid_metrics) if primary_validation_split == "valid_iid" else None,
        "train_relative_metric_denominator_mean_abs_true_deltaT": _maybe_float(
            train_metrics, "mean_abs_true_deltaT"
        ),
        "valid_relative_metric_denominator_mean_abs_true_deltaT": float(valid_metrics["mean_abs_true_deltaT"]),
        "valid_iid_relative_metric_denominator_mean_abs_true_deltaT": (
            float(valid_metrics["mean_abs_true_deltaT"]) if primary_validation_split == "valid_iid" else None
        ),
        "train_recovered_T_mse": _maybe_float(train_metrics, "recovered_temperature_mse"),
        "valid_recovered_T_mse": float(valid_metrics["recovered_temperature_mse"]),
        "valid_iid_recovered_T_mse": (
            float(valid_metrics["recovered_temperature_mse"]) if primary_validation_split == "valid_iid" else None
        ),
        "train_recovered_T_rmse_K": _maybe_float(train_metrics, "recovered_T_rmse_K"),
        "valid_recovered_T_rmse_K": float(valid_metrics["recovered_T_rmse_K"]),
        "valid_iid_recovered_T_rmse_K": (
            float(valid_metrics["recovered_T_rmse_K"]) if primary_validation_split == "valid_iid" else None
        ),
        "train_full_metrics_computed": train_components is not None and train_metrics is not None,
    }
    for component in (
        "shape_cv_loss", "log_scale_loss", "relative_field_loss", "raw_absolute_field_loss"
    ):
        record[f"train_{component}"] = _maybe_float(train_components, component)
        record[f"valid_{component}"] = _maybe_float(valid_components, component)
    for metric in (
        "scale_log_abs_error", "shape_cv_rmse", "joint_relative_rmse",
        "joint_amplitude_ratio", "joint_spatial_correlation", "joint_hotspot_rmse",
        "joint_topk_rmse", "oracle_scale_relative_rmse", "oracle_shape_relative_rmse",
        "physics_scale_relative_rmse",
    ):
        record[f"train_native_{metric}"] = _maybe_float(train_metrics, metric)
        record[f"valid_native_{metric}"] = _maybe_float(valid_metrics, metric)
    if valid_stress_components is not None and valid_stress_metrics is not None:
        record.update(
            {
                "stress_validation_split": stress_validation_split or "valid_stress",
                "valid_stress_loss": float(valid_stress_components["total_loss"]),
                "valid_stress_base_mse": float(valid_stress_components["base_mse"]),
                "valid_stress_raw_deltaT_mse": float(valid_stress_metrics["raw_delta_mse"]),
                "valid_stress_raw_rmse_K": float(valid_stress_metrics["raw_rmse_K"]),
                "valid_stress_raw_deltaT_rmse_K": float(valid_stress_metrics["raw_rmse_K"]),
                "valid_stress_error_pct": _metric_error_pct(valid_stress_metrics),
                "valid_stress_rel_rmse_v4_pct": _metric_error_pct(valid_stress_metrics),
                "valid_stress_relative_metric_denominator_mean_abs_true_deltaT": float(
                    valid_stress_metrics["mean_abs_true_deltaT"]
                ),
                "valid_stress_recovered_T_mse": float(valid_stress_metrics["recovered_temperature_mse"]),
                "valid_stress_recovered_T_rmse_K": float(valid_stress_metrics["recovered_T_rmse_K"]),
                "valid_stress_bg_signed_bias": float(valid_stress_components["bg_signed_bias"]),
                "valid_stress_hotspot_raw_mae": float(valid_stress_components["hotspot_raw_mae"]),
                "valid_stress_hotspot_mse": float(valid_stress_components["hotspot_mse"]),
                "valid_stress_strong_q_mse": float(valid_stress_components["strong_q_mse"]),
                "valid_stress_hotspot_mask_fraction": float(valid_stress_components["hotspot_mask_fraction"]),
                "valid_stress_strong_q_mask_fraction": float(valid_stress_components["strong_q_mask_fraction"]),
            }
        )
    record.update(_current_weight_payload(current_loss_config))
    return record


def _print_epoch_progress(record: dict[str, Any], epochs: int, log_mode: str) -> None:
    if log_mode == "quiet":
        return
    if log_mode == "compact":
        train_base_mse = _first_progress_numeric(
            record.get("train_base_mse"),
            record.get("epoch_mean_train_batch_base_mse"),
        )
        valid_iid_base_mse = _first_progress_numeric(
            record.get("valid_iid_base_mse"),
            record.get("valid_base_mse"),
        )
        best_valid_iid_base_mse = _first_progress_numeric(
            record.get("best_valid_iid_base_mse"),
            record.get("best_valid_base_mse"),
        )
        valid_raw_rmse = _first_progress_numeric(
            record.get("valid_iid_raw_rmse_K"),
            record.get("valid_raw_rmse_K"),
        )
        valid_rel_rmse_pct = _first_progress_numeric(
            record.get("valid_iid_rel_rmse_v4_pct"),
            record.get("valid_rel_rmse_v4_pct"),
            record.get("valid_iid_error_pct"),
            record.get("valid_error_pct"),
        )
        stress_progress = ""
        if not HIDE_MISSING_STRESS_COMPACT_LOG or record.get("valid_stress_loss") is not None:
            stress_progress = (
                f"stress={_format_progress_loss(record.get('valid_stress_loss'))} "
                f"stress_raw_rmse_K={_format_progress_sigfig_decimal(record.get('valid_stress_raw_rmse_K'))} "
            )
        _emit(
            f"epoch {record['epoch']}/{epochs} "
            f"lr={record['lr']:.2e} "
            f"train={_format_progress_loss(train_base_mse)} "
            f"valid={_format_progress_loss(valid_iid_base_mse)} "
            f"raw_rmse_K={_format_progress_sigfig_decimal(valid_raw_rmse)} "
            f"rel_rmse_v4_pct={_format_progress_percent(valid_rel_rmse_pct)} "
            f"{stress_progress}"
            f"best=e{_format_progress_int(record.get('best_epoch'))}/"
            f"{_format_progress_loss(best_valid_iid_base_mse)}"
        )
        return
    _emit(
        f"epoch {record['epoch']:03d}/{epochs:03d} "
        f"lr={record['lr']:.8e} "
        f"train_loss={record['train_loss']:.8e} "
        f"train_full_metrics={'computed' if record.get('train_full_metrics_computed') else 'skipped'} "
        f"primary_valid={record.get('primary_validation_split', 'valid')} "
        f"valid_loss={record['valid_loss']:.8e} "
        f"valid_stress_loss={_format_progress_value(record.get('valid_stress_loss'))} "
        f"train_base_mse={record['train_base_mse']:.8e} "
        f"valid_base_mse={record['valid_base_mse']:.8e} "
        f"train_background_penalty={record['train_background_penalty']:.8e} "
        f"valid_background_penalty={record['valid_background_penalty']:.8e} "
        f"train_background_l1={record['train_background_l1']:.8e} "
        f"valid_background_l1={record['valid_background_l1']:.8e} "
        f"train_background_signed_bias_loss={record['train_background_signed_bias_loss']:.8e} "
        f"valid_background_signed_bias_loss={record['valid_background_signed_bias_loss']:.8e} "
        f"train_background_overprediction_loss={record['train_background_overprediction_loss']:.8e} "
        f"valid_background_overprediction_loss={record['valid_background_overprediction_loss']:.8e} "
        f"train_background_relative_abs={record['train_background_relative_abs']:.8e} "
        f"valid_background_relative_abs={record['valid_background_relative_abs']:.8e} "
        f"train_pseudo_negative_count={record['train_pseudo_negative_count']:.8e} "
        f"valid_pseudo_negative_count={record['valid_pseudo_negative_count']:.8e} "
        f"train_pseudo_negative_over_loss={record['train_pseudo_negative_over_loss']:.8e} "
        f"valid_pseudo_negative_over_loss={record['valid_pseudo_negative_over_loss']:.8e} "
        f"train_pseudo_negative_unweighted_loss={record['train_pseudo_negative_unweighted_loss']:.8e} "
        f"valid_pseudo_negative_unweighted_loss={record['valid_pseudo_negative_unweighted_loss']:.8e} "
        f"train_pseudo_negative_weighted_loss={record['train_pseudo_negative_weighted_loss']:.8e} "
        f"valid_pseudo_negative_weighted_loss={record['valid_pseudo_negative_weighted_loss']:.8e} "
        f"train_pseudo_negative_weighted_fraction_of_total_loss={record['train_pseudo_negative_weighted_fraction_of_total_loss']:.8e} "
        f"valid_pseudo_negative_weighted_fraction_of_total_loss={record['valid_pseudo_negative_weighted_fraction_of_total_loss']:.8e} "
        f"train_pseudo_negative_bias={record['train_pseudo_negative_bias']:.8e} "
        f"valid_pseudo_negative_bias={record['valid_pseudo_negative_bias']:.8e} "
        f"train_pseudo_negative_over_ratio={record['train_pseudo_negative_over_ratio']:.8e} "
        f"valid_pseudo_negative_over_ratio={record['valid_pseudo_negative_over_ratio']:.8e} "
        f"train_hotspot_retention_loss={record['train_hotspot_retention_loss']:.8e} "
        f"valid_hotspot_retention_loss={record['valid_hotspot_retention_loss']:.8e} "
        f"train_hotspot_mse={record['train_hotspot_mse']:.8e} "
        f"valid_hotspot_mse={record['valid_hotspot_mse']:.8e} "
        f"train_strong_q_mse={record['train_strong_q_mse']:.8e} "
        f"valid_strong_q_mse={record['valid_strong_q_mse']:.8e} "
        f"valid_hotspot_mask_fraction={record['valid_hotspot_mask_fraction']:.8e} "
        f"valid_strong_q_mask_fraction={record['valid_strong_q_mask_fraction']:.8e} "
        f"train_bg_pred_raw_mean={record['train_bg_pred_raw_mean']:.8e} "
        f"valid_bg_pred_raw_mean={record['valid_bg_pred_raw_mean']:.8e} "
        f"train_bg_signed_bias={record['train_bg_signed_bias']:.8e} "
        f"valid_bg_signed_bias={record['valid_bg_signed_bias']:.8e} "
        f"train_bg_abs_mean={record['train_bg_abs_mean']:.8e} "
        f"valid_bg_abs_mean={record['valid_bg_abs_mean']:.8e} "
        f"train_hotspot_raw_mae={record['train_hotspot_raw_mae']:.8e} "
        f"valid_hotspot_raw_mae={record['valid_hotspot_raw_mae']:.8e} "
        f"train_raw_deltaT_mse={record['train_raw_deltaT_mse']:.8e} "
        f"valid_raw_deltaT_mse={record['valid_raw_deltaT_mse']:.8e} "
        f"train_raw_rmse_K={_format_progress_sigfig_decimal(record['train_raw_rmse_K'])} "
        f"valid_raw_rmse_K={_format_progress_sigfig_decimal(record['valid_raw_rmse_K'])} "
        f"valid_rel_rmse_v4_pct={record['valid_rel_rmse_v4_pct']:.8e} "
        f"train_recovered_T_mse={record['train_recovered_T_mse']:.8e} "
        f"valid_recovered_T_mse={record['valid_recovered_T_mse']:.8e} "
        f"current_background_l1_weight={record['current_background_l1_weight']:.8e} "
        f"current_background_bias_weight={record['current_background_bias_weight']:.8e} "
        f"current_background_over_weight={record['current_background_over_weight']:.8e} "
        f"current_background_relative_weight={record['current_background_relative_weight']:.8e} "
        f"current_hotspot_weight={record['current_hotspot_weight']:.8e} "
        f"current_strong_q_weight={record['current_strong_q_weight']:.8e}"
    )


def _print_epoch_light_progress(record: dict[str, Any], epochs: int, log_mode: str) -> None:
    if log_mode == "quiet":
        return
    valid_total = record.get("valid_iid_loss")
    if valid_total is None:
        valid_total = record.get("valid_loss")
    valid_base = record.get("valid_iid_base_mse")
    if valid_base is None:
        valid_base = record.get("valid_base_mse")
    valid_raw_rmse = _first_progress_numeric(
        record.get("valid_iid_raw_rmse_K"),
        record.get("valid_raw_rmse_K"),
    )
    valid_rel_rmse_pct = _first_progress_numeric(
        record.get("valid_iid_rel_rmse_v4_pct"),
        record.get("valid_rel_rmse_v4_pct"),
        record.get("valid_iid_error_pct"),
        record.get("valid_error_pct"),
    )
    _emit(
        f"epoch {record['epoch']:03d}/{epochs:03d} "
        f"valid_total={_format_progress_loss(valid_total)} "
        f"valid_base={_format_progress_loss(valid_base)} "
        f"raw_rmse_K={_format_progress_sigfig_decimal(valid_raw_rmse)} "
        f"rel_rmse_v4_pct={_format_progress_percent(valid_rel_rmse_pct)}"
    )


def _fit_once(
    train_groups: list[dict],
    valid_groups: list[dict],
    valid_stress_groups: list[dict],
    stats: dict,
    epochs: int,
    lr_config: dict[str, Any],
    model_seed: int,
    batch_order_seed: int,
    report_every: int,
    train_metrics_schedule: str,
    grad_norm_report_every: int,
    loss_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    model_config: dict[str, Any],
    batch_config: dict[str, Any],
    selection_metric: str,
    log_mode: str,
    progress_enabled: bool,
    init_mode: str = "real_first_batch",
    init_checkpoint: Path | None = None,
    checkpoint_load_strict: bool = True,
    partial_load_policy: str = "matching",
    timings: dict[str, float] | None = None,
    profile_enabled: bool = False,
    memory_audit: MemoryAudit | None = None,
    primary_validation_split: str = "valid",
    stress_validation_split: str | None = None,
    track_point_global_best: bool = False,
) -> dict:
    timings = timings if timings is not None else {}
    init_start = time.perf_counter()
    if memory_audit is not None:
        memory_audit.record("model_init_start")
    _progress(progress_enabled, "startup", "initializing model parameters ...")
    model = GraphNeuralOperator(**model_config)
    init_inputs = _init_inputs_for_mode(train_groups[0], init_mode)
    params = _model_init(
        model,
        jax.random.PRNGKey(model_seed),
        train_groups[0],
        init_inputs,
    )["params"]
    params, checkpoint_load_info = _load_init_checkpoint_params(
        params,
        init_checkpoint,
        strict=checkpoint_load_strict,
        partial_load_policy=partial_load_policy,
    )
    _record_timing(timings, "model_init", init_start)
    if memory_audit is not None:
        memory_audit.record("model_init_end")
    _progress(
        progress_enabled,
        "startup",
        (
            f"model parameters initialized init_mode={init_mode} "
            f"checkpoint_loaded={checkpoint_load_info['loaded']} "
            f"loaded_keys={checkpoint_load_info['loaded_key_count']}"
        ),
        init_start,
    )

    batch_enabled = batch_config.get("batch_size") is not None
    native_enabled = model_config.get("native_output_mode") == "native_shape_scale"
    metrics_fn = (
        _weighted_metrics
        if batch_enabled or native_enabled
        else _legacy_full_batch_metrics_with_true_rms_denominator
    )
    updates_per_epoch = int(len(train_groups) if batch_enabled else 1)

    initial_start = time.perf_counter()
    if memory_audit is not None:
        memory_audit.record("initial_loss_start")
    _progress(progress_enabled, "startup", "computing initial train/valid losses ...")
    initial_loss_config = _loss_config_for_epoch(loss_config, 1)
    train_initial_components = _loss_components(model, params, train_groups, stats, initial_loss_config)
    valid_initial_components = _loss_components(model, params, valid_groups, stats, initial_loss_config)
    valid_initial_metrics = metrics_fn(model, params, valid_groups, stats)
    valid_stress_initial_components = (
        _loss_components(model, params, valid_stress_groups, stats, initial_loss_config)
        if valid_stress_groups
        else None
    )
    valid_stress_initial_metrics = (
        metrics_fn(model, params, valid_stress_groups, stats)
        if valid_stress_groups
        else None
    )
    _record_timing(timings, "initial_loss", initial_start)
    if memory_audit is not None:
        memory_audit.record("initial_loss_end")
    _progress(progress_enabled, "startup", "initial train/valid losses computed", initial_start)
    train_losses = [float(train_initial_components["total_loss"])]
    train_loss_epochs = [0]
    valid_losses = [float(valid_initial_components["total_loss"])]
    valid_stress_losses = (
        [float(valid_stress_initial_components["total_loss"])]
        if valid_stress_initial_components is not None
        else []
    )
    grad_norms = []
    grad_norm_reported_batch_count = 0
    grad_norm_skipped_batch_count = 0
    epoch_batch_counts = []
    epoch_train_batch_order_hashes = []
    lr_history = []
    loss_weight_history = []
    grad_finite = True
    epoch_history = []
    train_batch_records: list[dict[str, Any]] = []
    validation_batch_records: list[dict[str, Any]] = []
    best_score: float | None = None
    best_record: dict[str, Any] | None = None
    best_params = None
    best_params_storage: str | None = None
    point_global_best_score: float | None = None
    point_global_best_record: dict[str, Any] | None = None
    point_global_best_params = None
    final_epoch_train_components = None
    final_epoch_train_metrics = None
    final_epoch_valid_components = None
    final_epoch_valid_metrics = None
    final_epoch_valid_stress_components = None
    final_epoch_valid_stress_metrics = None
    optax_lr_config = dict(lr_config)
    optax_lr_config["updates_per_epoch"] = updates_per_epoch
    optax_state = _build_optax_state(
        params,
        epochs=epochs,
        lr_config=optax_lr_config,
        optimizer_config=optimizer_config,
    )
    train_metrics_epoch_values = train_metrics_epochs(train_metrics_schedule, epochs)
    train_metrics_epoch_lookup = set(train_metrics_epoch_values)
    if batch_enabled:
        _progress(
            progress_enabled,
            "train",
            (
                f"epoch loop start epochs={epochs} report_every={report_every} "
                f"mini_batch_groups={len(train_groups)} batch_size={batch_config['batch_size']} "
                f"train_metrics_schedule={train_metrics_schedule} "
                f"train_metrics_epochs={train_metrics_epoch_values}"
            ),
        )
    else:
        _progress(
            progress_enabled,
            "train",
            (
                f"epoch loop start epochs={epochs} report_every={report_every} "
                f"train_metrics_schedule={train_metrics_schedule} "
                f"train_metrics_epochs={train_metrics_epoch_values}"
            ),
        )
    epoch_loop_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        lr_epoch = _lr_for_epoch(epoch, epochs, lr_config)
        current_loss_config = _loss_config_for_epoch(loss_config, epoch)
        loss_weight_history.append({"epoch": int(epoch), **_current_weight_payload(current_loss_config)})
        lr_history.append(lr_epoch)
        should_report = _should_report_epoch(epoch, epochs, report_every)
        epoch_start = time.perf_counter()
        if memory_audit is not None:
            memory_audit.record("epoch_start", epoch=epoch)
        if should_report or epoch <= 3:
            if batch_enabled:
                _progress(
                    progress_enabled,
                    "train",
                    (
                        f"epoch {epoch:03d}/{epochs:03d} start lr={lr_epoch:.3e} "
                        f"batches={len(train_groups)}"
                    ),
                )
            else:
                _progress(progress_enabled, "train", f"epoch {epoch:03d}/{epochs:03d} start lr={lr_epoch:.3e}")

        train_step_start = time.perf_counter()
        epoch_train_batch_records: list[dict[str, Any]] = []
        epoch_train_batch_losses: list[float] = []
        epoch_train_batch_base_mses: list[tuple[int, float]] = []
        epoch_grad_norms: list[float] = []
        epoch_native_group_grad_norms: dict[str, list[float]] = {
            "backbone": [],
            "shape_decoder": [],
            "scale_head": [],
        }
        epoch_update_norms: list[float] = []
        epoch_param_norms: list[float] = []
        epoch_update_to_param_ratios: list[float] = []
        if batch_enabled:
            train_epoch_groups = _epoch_train_groups(
                train_groups,
                epoch=epoch,
                seed=batch_order_seed,
                shuffle=bool(batch_config.get("shuffle_train_batches")),
            )
            epoch_train_batch_order_hashes.append(_group_sample_id_hash(train_epoch_groups))
            if memory_audit is not None:
                memory_audit.record(
                    "train_epoch_groups_ready",
                    epoch=epoch,
                    detail=_groups_memory_signature(train_epoch_groups),
                )
            batch_grad_norms = []
            for batch_index, batch_group in enumerate(train_epoch_groups, start=1):
                if memory_audit is not None and memory_audit.every_batch:
                    memory_audit.record(
                        "train_batch_start",
                        epoch=epoch,
                        batch_index=batch_index,
                        split="train",
                        detail=_batch_shape_signature(batch_group),
                    )
                def loss_fn(current_params, group=batch_group):
                    components = _loss_components(
                        model, current_params, [group], stats, current_loss_config
                    )
                    return components["total_loss"], components["base_mse"]

                batch_start = time.perf_counter()
                loss_grad_start = time.perf_counter()
                (loss_value, batch_base_mse), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(params)
                if native_enabled:
                    grads = mask_branch_gradients(grads, model_config["native_branch_mode"])
                if profile_enabled:
                    _block_until_ready_tree((loss_value, grads))
                loss_grad_time = time.perf_counter() - loss_grad_start
                batch_loss_value = float(loss_value)
                batch_base_mse_value = float(batch_base_mse)
                epoch_train_batch_losses.append(batch_loss_value)
                epoch_train_batch_base_mses.append(
                    (_sample_count(batch_group), batch_base_mse_value)
                )

                grad_norm_reported = should_report_grad_norm(grad_norm_report_every, batch_index)
                compute_batch_norms = bool(grad_norm_reported or profile_enabled)
                grad_norm = None
                native_group_grad_norms: dict[str, float] = {}
                grad_norm_time = 0.0
                if compute_batch_norms:
                    grad_norm_start = time.perf_counter()
                    grad_norm = _global_norm(grads)
                    if native_enabled:
                        native_group_grad_norms = {
                            name: float(value)
                            for name, value in native_gradient_group_norms(grads).items()
                        }
                        for name, value in native_group_grad_norms.items():
                            epoch_native_group_grad_norms[name].append(value)
                    grad_norm_time = time.perf_counter() - grad_norm_start
                    epoch_grad_norms.append(grad_norm)
                    grad_finite = grad_finite and bool(np.isfinite(grad_norm))
                if grad_norm_reported:
                    assert grad_norm is not None
                    batch_grad_norms.append(grad_norm)
                    grad_norm_reported_batch_count += 1
                else:
                    grad_norm_skipped_batch_count += 1

                optimizer_update_start = time.perf_counter()
                if optax_state is None:
                    updates = tree.tree_map(lambda grad: -lr_epoch * grad, grads)
                    params = tree.tree_map(lambda param, update: param + update, params, updates)
                else:
                    updates, opt_state = optax_state["tx"].update(grads, optax_state["state"], params)
                    optax_state["state"] = opt_state
                    if native_enabled:
                        updates = mask_branch_gradients(
                            updates, model_config["native_branch_mode"]
                        )
                    params = optax_state["apply_updates"](params, updates)
                if profile_enabled:
                    _block_until_ready_tree(params)
                optimizer_update_time = time.perf_counter() - optimizer_update_start
                update_norm = None
                param_norm = None
                update_to_param_norm_ratio = None
                if compute_batch_norms:
                    update_norm = _global_norm(updates)
                    param_norm = _global_norm(params)
                    update_to_param_norm_ratio = update_norm / max(param_norm, 1.0e-12)
                    epoch_update_norms.append(update_norm)
                    epoch_param_norms.append(param_norm)
                    epoch_update_to_param_ratios.append(update_to_param_norm_ratio)

                if profile_enabled:
                    total_batch_time = time.perf_counter() - batch_start
                    output_scalar_extraction_time = 0.0
                    other_time = max(
                        total_batch_time
                        - loss_grad_time
                        - grad_norm_time
                        - optimizer_update_time
                        - output_scalar_extraction_time,
                        0.0,
                    )
                    batch_record = {
                        "epoch_index": int(epoch),
                        "batch_index": int(batch_index),
                        "split": "train",
                        "batch_size": _sample_count(batch_group),
                        "group_count": 1,
                        "train_batch_loss": float(batch_loss_value),
                        "train_batch_base_mse": float(batch_base_mse_value),
                        "total_batch_time": float(total_batch_time),
                        "loss_grad_time": float(loss_grad_time),
                        "grad_norm_time": float(grad_norm_time),
                        "grad_norm": float(grad_norm) if grad_norm is not None else None,
                        "native_gradient_group_norms": native_group_grad_norms,
                        "grad_norm_reported": bool(grad_norm_reported),
                        "update_norm": float(update_norm) if update_norm is not None else None,
                        "param_norm": float(param_norm) if param_norm is not None else None,
                        "update_to_param_norm_ratio": (
                            float(update_to_param_norm_ratio) if update_to_param_norm_ratio is not None else None
                        ),
                        "optimizer_update_time": float(optimizer_update_time),
                        "output_scalar_extraction_time": float(output_scalar_extraction_time),
                        "other_time": float(other_time),
                        "batch_shape_signature": _batch_shape_signature(batch_group),
                    }
                    batch_record["batch_shape_signature_key"] = _shape_signature_key(
                        batch_record["batch_shape_signature"]
                    )
                    epoch_train_batch_records.append(batch_record)
                    train_batch_records.append(batch_record)
                if memory_audit is not None and memory_audit.every_batch:
                    memory_audit.record(
                        "train_batch_end",
                        epoch=epoch,
                        batch_index=batch_index,
                        split="train",
                        detail={
                            "loss": float(batch_loss_value),
                            "grad_norm": float(grad_norm) if grad_norm is not None else None,
                            "update_norm": float(update_norm) if update_norm is not None else None,
                            "param_norm": float(param_norm) if param_norm is not None else None,
                        },
                    )
                del grads, updates, loss_value, batch_base_mse
            epoch_batch_counts.append(len(train_epoch_groups))
            if batch_grad_norms:
                grad_norms.append(float(np.mean(batch_grad_norms)))
        else:
            epoch_train_batch_order_hashes.append(_group_sample_id_hash(train_groups))
            def loss_fn(current_params):
                components = _loss_components(
                    model, current_params, train_groups, stats, current_loss_config
                )
                return components["total_loss"], components["base_mse"]

            batch_start = time.perf_counter()
            if memory_audit is not None:
                memory_audit.record(
                    "train_full_batch_start",
                    epoch=epoch,
                    batch_index=1,
                    split="train",
                    detail=_groups_memory_signature(train_groups),
                )
            loss_grad_start = time.perf_counter()
            (loss_value, batch_base_mse), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)
            if native_enabled:
                grads = mask_branch_gradients(grads, model_config["native_branch_mode"])
            if profile_enabled:
                _block_until_ready_tree((loss_value, grads))
            loss_grad_time = time.perf_counter() - loss_grad_start
            batch_loss_value = float(loss_value)
            batch_base_mse_value = float(batch_base_mse)
            epoch_train_batch_losses.append(batch_loss_value)
            epoch_train_batch_base_mses.append(
                (sum(_sample_count(group) for group in train_groups), batch_base_mse_value)
            )

            grad_norm_reported = should_report_grad_norm(grad_norm_report_every, 1)
            compute_batch_norms = bool(grad_norm_reported or profile_enabled)
            grad_norm = None
            native_group_grad_norms: dict[str, float] = {}
            grad_norm_time = 0.0
            if compute_batch_norms:
                grad_norm_start = time.perf_counter()
                grad_norm = _global_norm(grads)
                if native_enabled:
                    native_group_grad_norms = {
                        name: float(value)
                        for name, value in native_gradient_group_norms(grads).items()
                    }
                    for name, value in native_group_grad_norms.items():
                        epoch_native_group_grad_norms[name].append(value)
                grad_norm_time = time.perf_counter() - grad_norm_start
                epoch_grad_norms.append(grad_norm)
                grad_finite = grad_finite and bool(np.isfinite(grad_norm))
            if grad_norm_reported:
                assert grad_norm is not None
                grad_norms.append(grad_norm)
                grad_norm_reported_batch_count += 1
            else:
                grad_norm_skipped_batch_count += 1

            optimizer_update_start = time.perf_counter()
            if optax_state is None:
                updates = tree.tree_map(lambda grad: -lr_epoch * grad, grads)
                params = tree.tree_map(lambda param, update: param + update, params, updates)
            else:
                updates, opt_state = optax_state["tx"].update(grads, optax_state["state"], params)
                optax_state["state"] = opt_state
                if native_enabled:
                    updates = mask_branch_gradients(
                        updates, model_config["native_branch_mode"]
                    )
                params = optax_state["apply_updates"](params, updates)
            if profile_enabled:
                _block_until_ready_tree(params)
            optimizer_update_time = time.perf_counter() - optimizer_update_start
            update_norm = None
            param_norm = None
            update_to_param_norm_ratio = None
            if compute_batch_norms:
                update_norm = _global_norm(updates)
                param_norm = _global_norm(params)
                update_to_param_norm_ratio = update_norm / max(param_norm, 1.0e-12)
                epoch_update_norms.append(update_norm)
                epoch_param_norms.append(param_norm)
                epoch_update_to_param_ratios.append(update_to_param_norm_ratio)
            if profile_enabled:
                total_batch_time = time.perf_counter() - batch_start
                output_scalar_extraction_time = 0.0
                other_time = max(
                    total_batch_time
                    - loss_grad_time
                    - grad_norm_time
                    - optimizer_update_time
                    - output_scalar_extraction_time,
                    0.0,
                )
                signature = {
                    "group_count": len(train_groups),
                    "group_signatures": [_batch_shape_signature(group) for group in train_groups],
                }
                batch_record = {
                    "epoch_index": int(epoch),
                    "batch_index": 1,
                    "split": "train",
                    "batch_size": sum(_sample_count(group) for group in train_groups),
                    "group_count": len(train_groups),
                    "train_batch_loss": float(batch_loss_value),
                    "train_batch_base_mse": float(batch_base_mse_value),
                    "total_batch_time": float(total_batch_time),
                    "loss_grad_time": float(loss_grad_time),
                    "grad_norm_time": float(grad_norm_time),
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "native_gradient_group_norms": native_group_grad_norms,
                    "grad_norm_reported": bool(grad_norm_reported),
                    "update_norm": float(update_norm) if update_norm is not None else None,
                    "param_norm": float(param_norm) if param_norm is not None else None,
                    "update_to_param_norm_ratio": (
                        float(update_to_param_norm_ratio) if update_to_param_norm_ratio is not None else None
                    ),
                    "optimizer_update_time": float(optimizer_update_time),
                    "output_scalar_extraction_time": float(output_scalar_extraction_time),
                    "other_time": float(other_time),
                    "batch_shape_signature": signature,
                }
                batch_record["batch_shape_signature_key"] = _shape_signature_key(signature)
                epoch_train_batch_records.append(batch_record)
                train_batch_records.append(batch_record)
            if memory_audit is not None:
                memory_audit.record(
                    "train_full_batch_end",
                    epoch=epoch,
                    batch_index=1,
                    split="train",
                    detail={
                        "loss": float(batch_loss_value),
                        "grad_norm": float(grad_norm) if grad_norm is not None else None,
                        "update_norm": float(update_norm) if update_norm is not None else None,
                        "param_norm": float(param_norm) if param_norm is not None else None,
                    },
                )
            del grads, updates, loss_value, batch_base_mse
            epoch_batch_counts.append(1)
        train_step_time = time.perf_counter() - train_step_start

        train_metrics_computed = epoch in train_metrics_epoch_lookup
        train_components = None
        train_metrics = None
        train_metrics_time = 0.0
        if train_metrics_computed:
            train_metrics_start = time.perf_counter()
            if memory_audit is not None:
                memory_audit.record(
                    "train_metrics_start",
                    epoch=epoch,
                    split="train",
                    detail=_groups_memory_signature(train_groups),
                )
            train_components = _loss_components(model, params, train_groups, stats, current_loss_config)
            train_metrics = metrics_fn(model, params, train_groups, stats)
            if profile_enabled:
                _block_until_ready_tree((train_components, train_metrics))
            train_metrics_time = time.perf_counter() - train_metrics_start
            if memory_audit is not None:
                memory_audit.record("train_metrics_end", epoch=epoch, split="train")

        validation_start = time.perf_counter()
        if memory_audit is not None:
            memory_audit.record(
                "valid_start",
                epoch=epoch,
                split=primary_validation_split,
                detail=_groups_memory_signature(valid_groups),
            )
        if profile_enabled:
            valid_components, valid_metrics, epoch_validation_records = _evaluate_groups_profiled(
                model,
                params,
                valid_groups,
                stats,
                current_loss_config,
                metrics_fn,
                epoch=epoch,
                split="valid",
            )
            validation_batch_records.extend(epoch_validation_records)
        else:
            valid_components = _loss_components(model, params, valid_groups, stats, current_loss_config)
            valid_metrics = metrics_fn(model, params, valid_groups, stats)
        validation_time = time.perf_counter() - validation_start
        if memory_audit is not None:
            memory_audit.record("valid_end", epoch=epoch, split=primary_validation_split)

        valid_stress_components = None
        valid_stress_metrics = None
        valid_stress_time = 0.0
        if valid_stress_groups:
            valid_stress_start = time.perf_counter()
            if memory_audit is not None:
                memory_audit.record(
                    "valid_stress_start",
                    epoch=epoch,
                    split=stress_validation_split or "valid_stress",
                    detail=_groups_memory_signature(valid_stress_groups),
                )
            if profile_enabled:
                valid_stress_components, valid_stress_metrics, epoch_stress_records = _evaluate_groups_profiled(
                    model,
                    params,
                    valid_stress_groups,
                    stats,
                    current_loss_config,
                    metrics_fn,
                    epoch=epoch,
                    split=stress_validation_split or "valid_stress",
                )
                validation_batch_records.extend(epoch_stress_records)
            else:
                valid_stress_components = _loss_components(
                    model, params, valid_stress_groups, stats, current_loss_config
                )
                valid_stress_metrics = metrics_fn(model, params, valid_stress_groups, stats)
            valid_stress_time = time.perf_counter() - valid_stress_start
            if memory_audit is not None:
                memory_audit.record(
                    "valid_stress_end",
                    epoch=epoch,
                    split=stress_validation_split or "valid_stress",
                )

        if train_components is not None:
            train_losses.append(float(train_components["total_loss"]))
            train_loss_epochs.append(int(epoch))
        valid_losses.append(float(valid_components["total_loss"]))
        if valid_stress_components is not None:
            valid_stress_losses.append(float(valid_stress_components["total_loss"]))
        if epoch == epochs and should_reuse_final_metrics(train_metrics_computed):
            final_epoch_train_components = train_components
            final_epoch_train_metrics = train_metrics
            final_epoch_valid_components = valid_components
            final_epoch_valid_metrics = valid_metrics
            final_epoch_valid_stress_components = valid_stress_components
            final_epoch_valid_stress_metrics = valid_stress_metrics
        record = _epoch_history_record(
            epoch,
            lr_epoch,
            current_loss_config,
            train_components,
            valid_components,
            valid_metrics,
            train_metrics,
            primary_validation_split=primary_validation_split,
            valid_stress_components=valid_stress_components,
            valid_stress_metrics=valid_stress_metrics,
            stress_validation_split=stress_validation_split,
        )
        record["train_batch_count"] = int(epoch_batch_counts[-1])
        record["valid_batch_count"] = int(len(valid_groups))
        record["batch_size"] = batch_config.get("batch_size")
        record["train_full_metrics_schedule"] = train_metrics_schedule
        record["train_full_metrics_epoch"] = bool(train_metrics_computed)
        batch_summary = _summarize_batch_records(epoch_train_batch_records) if profile_enabled else {}
        record["train_batch_timing_summary"] = batch_summary
        batch_loss_summary = _epoch_monitor_summary(epoch_train_batch_losses)
        grad_norm_summary = _epoch_monitor_summary(epoch_grad_norms)
        update_norm_summary = _epoch_monitor_summary(epoch_update_norms)
        param_norm_summary = _epoch_monitor_summary(epoch_param_norms)
        update_param_ratio_summary = _epoch_monitor_summary(epoch_update_to_param_ratios)
        record["epoch_mean_train_batch_loss"] = batch_loss_summary["mean"]
        train_batch_base_mse_count = sum(
            count for count, _ in epoch_train_batch_base_mses
        )
        record["epoch_mean_train_batch_base_mse"] = (
            sum(count * value for count, value in epoch_train_batch_base_mses)
            / train_batch_base_mse_count
            if train_batch_base_mse_count > 0
            else None
        )
        record["epoch_min_train_batch_loss"] = batch_loss_summary["min"]
        record["epoch_max_train_batch_loss"] = batch_loss_summary["max"]
        record["epoch_mean_grad_norm"] = grad_norm_summary["mean"]
        record["epoch_max_grad_norm"] = grad_norm_summary["max"]
        for group_name, values in epoch_native_group_grad_norms.items():
            group_summary = _epoch_monitor_summary(values)
            record[f"epoch_mean_{group_name}_grad_norm"] = group_summary["mean"]
            record[f"epoch_max_{group_name}_grad_norm"] = group_summary["max"]
        record["epoch_mean_update_norm"] = update_norm_summary["mean"]
        record["epoch_max_update_norm"] = update_norm_summary["max"]
        record["epoch_mean_param_norm"] = param_norm_summary["mean"]
        record["epoch_update_to_param_norm_ratio"] = update_param_ratio_summary["mean"]
        record["epoch_max_update_to_param_norm_ratio"] = update_param_ratio_summary["max"]
        score = float(record[selection_metric])
        if best_score is None or score < best_score:
            if memory_audit is not None:
                memory_audit.record(
                    "best_params_copy_start",
                    epoch=epoch,
                    detail={"selection_metric": selection_metric, "score": score},
                )
            best_score = score
            best_record = dict(record)
            best_params = _host_params(params)
            best_params_storage = "cpu"
            if memory_audit is not None:
                memory_audit.record("best_params_copy_end", epoch=epoch)
        if track_point_global_best:
            point_global_score = float(record["valid_rel_rmse_v4_pct"])
            if point_global_best_score is None or point_global_score < point_global_best_score:
                point_global_best_score = point_global_score
                point_global_best_record = dict(record)
                point_global_best_params = _host_params(params)
        record["best_epoch"] = best_record.get("epoch") if best_record is not None else None
        record["best_valid_iid_loss"] = best_record.get("valid_iid_loss") if best_record is not None else None
        record["best_valid_base_mse"] = best_record.get("valid_base_mse") if best_record is not None else None
        record["best_valid_iid_base_mse"] = (
            best_record.get("valid_iid_base_mse") if best_record is not None else None
        )
        record["epoch_train_time_s"] = float(train_step_time)
        record["epoch_train_metrics_time_s"] = float(train_metrics_time)
        record["epoch_validation_time_s"] = float(validation_time)
        record["epoch_valid_stress_time_s"] = float(valid_stress_time)
        record["epoch_total_time_s"] = float(time.perf_counter() - epoch_start)
        epoch_history.append(record)
        if memory_audit is not None:
            memory_audit.collect("epoch_gc_end", epoch=epoch)
            memory_audit.record(
                "epoch_end",
                epoch=epoch,
                detail={
                    "train_time_s": float(train_step_time),
                    "validation_time_s": float(validation_time),
                    "valid_stress_time_s": float(valid_stress_time),
                    "train_metrics_time_s": float(train_metrics_time),
                    "best_epoch": record["best_epoch"],
                },
            )
        if should_report:
            _progress(progress_enabled, "train", f"epoch {epoch:03d}/{epochs:03d} metrics computed", epoch_start)
            _print_epoch_progress(record, epochs, log_mode)
        else:
            _print_epoch_light_progress(record, epochs, log_mode)
    _record_timing(timings, "epoch_loop", epoch_loop_start)

    final_metrics_reused = (
        final_epoch_train_components is not None
        and final_epoch_train_metrics is not None
        and final_epoch_valid_components is not None
        and final_epoch_valid_metrics is not None
    )
    final_metrics_reuse_source = "last_epoch_full_metrics" if final_metrics_reused else None
    if final_metrics_reused:
        timings["final_metrics"] = 0.0
        train_metrics = final_epoch_train_metrics
        valid_metrics = final_epoch_valid_metrics
        final_train_components = final_epoch_train_components
        final_valid_components = final_epoch_valid_components
        final_valid_stress_components = final_epoch_valid_stress_components
        final_valid_stress_metrics = final_epoch_valid_stress_metrics
        _progress(progress_enabled, "train", "final train/valid metrics reused from last epoch full metrics")
    else:
        _progress(progress_enabled, "train", "computing final train/valid metrics ...")
        final_metrics_start = time.perf_counter()
        train_metrics = metrics_fn(model, params, train_groups, stats)
        valid_metrics = metrics_fn(model, params, valid_groups, stats)
        final_valid_stress_metrics = metrics_fn(model, params, valid_stress_groups, stats) if valid_stress_groups else None
        final_loss_config = _loss_config_for_epoch(loss_config, epochs)
        final_train_components = _loss_components(model, params, train_groups, stats, final_loss_config)
        final_valid_components = _loss_components(model, params, valid_groups, stats, final_loss_config)
        final_valid_stress_components = (
            _loss_components(model, params, valid_stress_groups, stats, final_loss_config)
            if valid_stress_groups
            else None
        )
        _record_timing(timings, "final_metrics", final_metrics_start)
        _progress(progress_enabled, "train", "final train/valid metrics computed", final_metrics_start)
    status_ok = (
        grad_finite
        and train_metrics["finite_ok"]
        and valid_metrics["finite_ok"]
        and (final_valid_stress_metrics is None or final_valid_stress_metrics["finite_ok"])
        and train_metrics["shape_ok"]
        and valid_metrics["shape_ok"]
        and (final_valid_stress_metrics is None or final_valid_stress_metrics["shape_ok"])
        and bool(np.all(np.isfinite(train_losses)))
        and bool(np.all(np.isfinite(valid_losses)))
        and (not valid_stress_losses or bool(np.all(np.isfinite(valid_stress_losses))))
    )
    return {
        "model": model,
        "params": params,
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "train_loss_epochs": np.asarray(train_loss_epochs, dtype=np.int64),
        "valid_losses": np.asarray(valid_losses, dtype=np.float64),
        "valid_iid_losses": np.asarray(valid_losses, dtype=np.float64) if primary_validation_split == "valid_iid" else np.asarray([], dtype=np.float64),
        "valid_stress_losses": np.asarray(valid_stress_losses, dtype=np.float64),
        "initial_train_loss": float(train_initial_components["total_loss"]),
        "initial_valid_loss": float(valid_initial_components["total_loss"]),
        "initial_valid_iid_loss": float(valid_initial_components["total_loss"]) if primary_validation_split == "valid_iid" else None,
        "initial_valid_base_mse": float(valid_initial_components["base_mse"]),
        "initial_valid_raw_deltaT_mse": float(valid_initial_metrics["raw_delta_mse"]),
        "initial_valid_iid_raw_deltaT_mse": float(valid_initial_metrics["raw_delta_mse"]) if primary_validation_split == "valid_iid" else None,
        "initial_valid_stress_loss": (
            float(valid_stress_initial_components["total_loss"])
            if valid_stress_initial_components is not None
            else None
        ),
        "initial_valid_stress_raw_deltaT_mse": (
            float(valid_stress_initial_metrics["raw_delta_mse"])
            if valid_stress_initial_metrics is not None
            else None
        ),
        "grad_norms": np.asarray(grad_norms, dtype=np.float64),
        "grad_norm_report_every": int(grad_norm_report_every),
        "grad_norm_reporting_mode": grad_norm_reporting_mode(int(grad_norm_report_every)),
        "grad_norm_reported_batch_count": int(grad_norm_reported_batch_count),
        "grad_norm_skipped_batch_count": int(grad_norm_skipped_batch_count),
        "epoch_batch_counts": np.asarray(epoch_batch_counts, dtype=np.int64),
        "epoch_train_batch_order_hashes": epoch_train_batch_order_hashes,
        "lr_history": np.asarray(lr_history, dtype=np.float64),
        "epoch_lrs": [float(value) for value in lr_history],
        "model_seed": int(model_seed),
        "batch_order_seed": int(batch_order_seed),
        "updates_per_epoch": int(updates_per_epoch),
        "total_update_count": int(sum(epoch_batch_counts)),
        "train_group_count": int(len(train_groups)),
        "train_group_sample_counts": [int(_sample_count(group)) for group in train_groups],
        "train_group_names": [str(group["name"]) for group in train_groups],
        "valid_iid_group_count": int(len(valid_groups)) if primary_validation_split == "valid_iid" else None,
        "valid_stress_group_count": int(len(valid_stress_groups)),
        "train_group_sample_id_hash": _group_sample_id_hash(train_groups),
        "valid_iid_sample_id_hash": (
            _group_sample_id_hash(valid_groups) if primary_validation_split == "valid_iid" else None
        ),
        "valid_stress_sample_id_hash": _group_sample_id_hash(valid_stress_groups),
        "deterministic_audit_enabled": True,
        "code_version_or_git_commit": _current_git_commit(),
        "loss_weight_history": loss_weight_history,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "epoch_history": epoch_history,
        "train_batch_records": train_batch_records,
        "validation_batch_records": validation_batch_records,
        "train_metrics_schedule": train_metrics_schedule,
        "train_metrics_epochs": train_metrics_epoch_values,
        "selection_metric": selection_metric,
        "primary_validation_split": primary_validation_split,
        "stress_validation_split": stress_validation_split,
        "checkpoint_load_info": checkpoint_load_info,
        "best_record": best_record,
        "best_params": best_params,
        "best_params_storage": best_params_storage,
        "best_score": best_score,
        "point_global_best_score": point_global_best_score,
        "point_global_best_record": point_global_best_record,
        "point_global_best_params": point_global_best_params,
        "final_epoch": int(epochs),
        "final_valid_loss": float(valid_losses[-1]),
        "final_valid_iid_loss": float(valid_losses[-1]) if primary_validation_split == "valid_iid" else None,
        "final_valid_stress_loss": (
            float(valid_stress_losses[-1])
            if valid_stress_losses
            else None
        ),
        "final_valid_stress_raw_deltaT_mse": (
            float(final_valid_stress_metrics["raw_delta_mse"])
            if final_valid_stress_metrics is not None
            else None
        ),
        "final_best_ratio": (
            float(valid_losses[-1]) / float(best_record["valid_loss"])
            if best_record is not None and best_record.get("valid_loss") not in (None, 0.0)
            else None
        ),
        "final_train_loss_components": _loss_components_payload(final_train_components),
        "final_valid_loss_components": _loss_components_payload(final_valid_components),
        "final_valid_iid_loss_components": (
            _loss_components_payload(final_valid_components) if primary_validation_split == "valid_iid" else None
        ),
        "final_valid_stress_loss_components": (
            _loss_components_payload(final_valid_stress_components)
            if final_valid_stress_components is not None
            else None
        ),
        "valid_stress_metrics": (
            final_valid_stress_metrics
            if final_valid_stress_metrics is not None
            else None
        ),
        "final_metrics_reused": bool(final_metrics_reused),
        "final_metrics_reuse_source": final_metrics_reuse_source,
        "final_metrics_time_s": float(timings.get("final_metrics", 0.0)),
        "grad_finite": grad_finite,
        "status_ok": status_ok,
    }


def _predict_temperatures(model, params, groups: list[dict], stats: dict) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for group in groups:
        prediction = _model_apply(model, params, group)
        if isinstance(prediction, Mapping):
            recovered = np.asarray(prediction["raw_temperature"])
        else:
            recovered = np.asarray(
                recover_temperature_from_normalized_delta(prediction, group["t_ref"], stats)
            )
        if not np.all(np.isfinite(recovered)):
            raise ValueError(f"Non-finite recovered predictions in group {group['name']}")
        for batch_index, sample_id in enumerate(group["sample_ids"]):
            predictions[sample_id] = recovered[batch_index, 0, :, :].astype(np.float64)
    return predictions


def _prediction_max_abs_difference(
    expected: Mapping[str, np.ndarray], actual: Mapping[str, np.ndarray]
) -> float:
    if set(expected) != set(actual):
        raise ValueError(
            "prediction sample ids differ after reload: "
            f"expected={len(expected)} actual={len(actual)}"
        )
    return max(
        (
            float(np.max(np.abs(np.asarray(expected[key]) - np.asarray(actual[key]))))
            for key in expected
        ),
        default=0.0,
    )


def _tree_max_abs_difference(expected: Any, actual: Any) -> float:
    expected_items = _param_leaf_items(expected)
    actual_items = _param_leaf_items(actual)
    if [path for path, _ in expected_items] != [path for path, _ in actual_items]:
        raise ValueError("checkpoint parameter paths differ after serialization")
    return max(
        (
            float(np.max(np.abs(np.asarray(expected_leaf) - np.asarray(actual_leaf))))
            for (_, expected_leaf), (_, actual_leaf) in zip(expected_items, actual_items)
        ),
        default=0.0,
    )


def _checkpoint_prediction_reload_audit(
    *,
    model: Any,
    groups: list[dict],
    stats: dict,
    entries: list[tuple[str, Path, Path, Mapping[str, np.ndarray], Any]],
    tolerance: float = 5.0e-3,
) -> dict[str, Any]:
    """Reload saved params and NPZ predictions, then reproduce predictions."""

    reports = []
    for label, checkpoint_path, predictions_path, expected, reference_params in entries:
        checkpoint_payload = _load_params_checkpoint(checkpoint_path)
        parameter_max_abs = _tree_max_abs_difference(
            _host_params(reference_params), checkpoint_payload["params"]
        )
        checkpoint_context = (
            checkpoint_payload.get("run_config_metadata", {}).get("global_context") or {}
        )
        standardizer = checkpoint_context.get("standardizer") or {}
        if checkpoint_context.get("enabled") and standardizer.get("fit_population") != "train_only":
            raise RuntimeError(
                f"{label}: checkpoint global-context standardizer is not train-only"
            )
        loaded_params = _device_params(checkpoint_payload["params"])
        try:
            reloaded_predictions = _predict_temperatures(model, loaded_params, groups, stats)
        finally:
            del loaded_params
        with np.load(predictions_path) as saved_payload:
            saved_predictions = {key: np.asarray(saved_payload[key]) for key in saved_payload.files}
        checkpoint_max_abs = _prediction_max_abs_difference(expected, reloaded_predictions)
        npz_max_abs = _prediction_max_abs_difference(expected, saved_predictions)
        passed = bool(
            parameter_max_abs == 0.0
            and checkpoint_max_abs <= tolerance
            and npz_max_abs == 0.0
        )
        reports.append(
            {
                "label": label,
                "checkpoint_path": str(checkpoint_path),
                "predictions_path": str(predictions_path),
                "sample_count": len(expected),
                "parameter_reload_max_abs_error": parameter_max_abs,
                "checkpoint_reload_max_abs_error_K": checkpoint_max_abs,
                "npz_reload_max_abs_error_K": npz_max_abs,
                "tolerance_K": float(tolerance),
                "global_context_fit_population": standardizer.get("fit_population"),
                "global_context_fit_sample_count": standardizer.get("fit_sample_count"),
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError(f"{label}: checkpoint/predictions reload audit failed: {reports[-1]}")
    return {
        "enabled": bool(entries),
        "status": "passed" if entries else "skipped",
        "entries": reports,
    }


def _native_runtime_architecture_audit(model: Any, params: Any, group: dict) -> dict[str, Any]:
    if getattr(model, "native_output_mode", None) != "native_shape_scale":
        return {"enabled": False}
    prediction = _model_apply(model, params, group)
    pooled = np.asarray(prediction["pooled_rnodes"])
    s_hat = np.asarray(prediction["s_hat"])
    payload = {
        "enabled": True,
        "scale_head_mode": str(getattr(model, "scale_head_mode")),
        "node_latent_width": int(getattr(model, "node_latent_size")),
        "pooled_latent_width": int(pooled.shape[-1]),
        "scale_head_input_width": int(getattr(model, "global_context_feature_dim"))
        + int(pooled.shape[-1]),
        "s_hat_positive": bool(np.all(s_hat > 0.0)),
        "finite": bool(
            np.all(np.isfinite(s_hat))
            and np.all(np.isfinite(np.asarray(prediction["deltaT_hat"])))
        ),
    }
    expected_pooled_width = (
        int(getattr(model, "node_latent_size"))
        if getattr(model, "scale_head_mode") == "physics_plus_pooled_latent"
        else 0
    )
    payload["expected_pooled_latent_width"] = expected_pooled_width
    payload["passed"] = bool(
        payload["pooled_latent_width"] == expected_pooled_width
        and payload["s_hat_positive"]
        and payload["finite"]
    )
    if not payload["passed"]:
        raise RuntimeError(f"native runtime architecture audit failed: {payload}")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, allow_nan=False, indent=2, sort_keys=True)


def _stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_path_entry_name(entry: Any) -> str:
    for attr in ("key", "idx", "name"):
        if hasattr(entry, attr):
            return str(getattr(entry, attr))
    return str(entry)


def _tree_path_name(path: tuple[Any, ...]) -> str:
    if not path:
        return "<root>"
    return "/".join(_tree_path_entry_name(entry) for entry in path)


def _param_leaf_items(params: Any) -> list[tuple[str, Any]]:
    return [(_tree_path_name(path), leaf) for path, leaf in tree.tree_flatten_with_path(params)[0]]


def _param_tree_summary(params: Any) -> dict[str, Any]:
    shapes = {}
    dtypes = {}
    param_count = 0
    for path, leaf in _param_leaf_items(params):
        array = np.asarray(leaf)
        shapes[path] = list(array.shape)
        dtypes[path] = str(array.dtype)
        param_count += int(array.size)
    return {
        "leaf_count": len(shapes),
        "param_count": int(param_count),
        "param_shapes": shapes,
        "param_dtypes": dtypes,
    }


def _host_params(params: Any) -> Any:
    host_params = jax.device_get(params)
    return tree.tree_map(
        lambda value: np.asarray(value) if hasattr(value, "shape") else value,
        host_params,
    )


def _device_params(params: Any) -> Any:
    return tree.tree_map(
        lambda value: jnp.asarray(value) if hasattr(value, "shape") else value,
        params,
    )


def _load_params_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint payload must be a dict")
    if "params" not in payload:
        raise ValueError(f"{path}: checkpoint missing params")
    return payload


def _checkpoint_path_allowed(path: str, partial_load_policy: str) -> bool:
    if partial_load_policy == "matching":
        return True
    if partial_load_policy == "skip_decoder":
        return not path.startswith("decoder/")
    if partial_load_policy == "encoder_processor_only":
        return path.startswith("encoder/") or path.startswith("processor/")
    raise ValueError(f"Unsupported partial load policy: {partial_load_policy}")


def _apply_checkpoint_params(
    initial_params: Any,
    checkpoint_params: Any,
    *,
    strict: bool,
    partial_load_policy: str,
) -> tuple[Any, dict[str, Any]]:
    if partial_load_policy not in PARTIAL_LOAD_POLICY_CHOICES:
        raise ValueError(f"--partial-load-policy must be one of {PARTIAL_LOAD_POLICY_CHOICES}")
    initial_items = _param_leaf_items(initial_params)
    checkpoint_items = _param_leaf_items(checkpoint_params)
    initial_map = {path: leaf for path, leaf in initial_items}
    checkpoint_map = {path: leaf for path, leaf in checkpoint_items}
    loaded_keys = []
    missing_keys = []
    unused_keys = []
    skipped_keys = []
    shape_mismatch_keys = []

    if strict:
        missing_keys = sorted(set(initial_map) - set(checkpoint_map))
        unused_keys = sorted(set(checkpoint_map) - set(initial_map))
        for path in sorted(set(initial_map).intersection(checkpoint_map)):
            if tuple(np.shape(initial_map[path])) != tuple(np.shape(checkpoint_map[path])):
                shape_mismatch_keys.append(
                    {
                        "key": path,
                        "expected_shape": list(np.shape(initial_map[path])),
                        "checkpoint_shape": list(np.shape(checkpoint_map[path])),
                    }
                )
        if missing_keys or unused_keys or shape_mismatch_keys:
            raise ValueError(
                "strict checkpoint load failed: "
                f"missing={len(missing_keys)} unused={len(unused_keys)} "
                f"shape_mismatch={len(shape_mismatch_keys)}"
            )
        loaded_keys = sorted(initial_map)
        loaded_params = tree.tree_unflatten(
            tree.tree_structure(initial_params),
            [jnp.asarray(checkpoint_map[path]) for path, _ in initial_items],
        )
    else:
        new_leaves = []
        for path, leaf in initial_items:
            checkpoint_leaf = checkpoint_map.get(path)
            if checkpoint_leaf is None:
                missing_keys.append(path)
                new_leaves.append(leaf)
                continue
            if not _checkpoint_path_allowed(path, partial_load_policy):
                skipped_keys.append(path)
                new_leaves.append(leaf)
                continue
            if tuple(np.shape(leaf)) != tuple(np.shape(checkpoint_leaf)):
                shape_mismatch_keys.append(
                    {
                        "key": path,
                        "expected_shape": list(np.shape(leaf)),
                        "checkpoint_shape": list(np.shape(checkpoint_leaf)),
                    }
                )
                new_leaves.append(leaf)
                continue
            loaded_keys.append(path)
            new_leaves.append(jnp.asarray(checkpoint_leaf))
        loaded_params = tree.tree_unflatten(tree.tree_structure(initial_params), new_leaves)
        unused_keys = sorted(set(checkpoint_map) - set(initial_map))

    info = {
        "checkpoint_load_mode": "params_only",
        "checkpoint_load_strict": bool(strict),
        "partial_load_policy": partial_load_policy,
        "loaded": True,
        "loaded_key_count": len(loaded_keys),
        "skipped_key_count": len(skipped_keys),
        "missing_key_count": len(missing_keys),
        "unused_key_count": len(unused_keys),
        "shape_mismatch_count": len(shape_mismatch_keys),
        "loaded_keys": loaded_keys[:50],
        "skipped_keys": skipped_keys[:50],
        "missing_keys": missing_keys[:50],
        "unused_keys": unused_keys[:50],
        "shape_mismatch_keys": shape_mismatch_keys[:50],
    }
    return loaded_params, info


def _load_init_checkpoint_params(
    initial_params: Any,
    checkpoint_path: Path | None,
    *,
    strict: bool,
    partial_load_policy: str,
) -> tuple[Any, dict[str, Any]]:
    if checkpoint_path is None:
        return initial_params, {
            "checkpoint_load_mode": None,
            "checkpoint_load_strict": bool(strict),
            "partial_load_policy": partial_load_policy,
            "loaded": False,
            "loaded_key_count": 0,
            "skipped_key_count": 0,
            "missing_key_count": 0,
            "unused_key_count": 0,
            "shape_mismatch_count": 0,
            "loaded_keys": [],
            "skipped_keys": [],
            "missing_keys": [],
            "unused_keys": [],
            "shape_mismatch_keys": [],
        }
    payload = _load_params_checkpoint(checkpoint_path)
    loaded_params, info = _apply_checkpoint_params(
        initial_params,
        payload["params"],
        strict=strict,
        partial_load_policy=partial_load_policy,
    )
    info.update(
        {
            "init_checkpoint": str(checkpoint_path),
            "checkpoint_schema_version": payload.get("schema_version"),
            "checkpoint_format_version": payload.get("checkpoint_format_version"),
            "checkpoint_kind": payload.get("checkpoint_kind"),
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_git_commit": payload.get("git_commit"),
            "checkpoint_model_config_hash": payload.get("model_config_hash"),
            "checkpoint_train_stats_hash": payload.get("train_stats_hash"),
            "optimizer_state_loaded": False,
        }
    )
    return loaded_params, info


def _checkpoint_record_from_result(result: dict[str, Any], *, kind: str) -> dict[str, Any]:
    if kind == "best":
        return dict(result.get("best_record") or {})
    if kind == "point_global_best":
        return dict(result.get("point_global_best_record") or {})
    if kind == "final":
        return {
            "epoch": result.get("final_epoch"),
            "valid_loss": result.get("final_valid_loss"),
            "valid_iid_loss": result.get("final_valid_iid_loss"),
            "valid_stress_loss": result.get("final_valid_stress_loss"),
            "valid_raw_deltaT_mse": result.get("valid_metrics", {}).get("raw_delta_mse"),
            "valid_iid_raw_deltaT_mse": result.get("valid_metrics", {}).get("raw_delta_mse"),
            "valid_stress_raw_deltaT_mse": result.get("final_valid_stress_raw_deltaT_mse"),
            "valid_raw_deltaT_rmse_K": result.get("valid_metrics", {}).get("raw_rmse_K"),
            "valid_iid_raw_deltaT_rmse_K": result.get("valid_metrics", {}).get("raw_rmse_K"),
            "valid_stress_raw_deltaT_rmse_K": (result.get("valid_stress_metrics", {}) or {}).get("raw_rmse_K"),
            "valid_recovered_T_mse": result.get("valid_metrics", {}).get("recovered_temperature_mse"),
            "valid_iid_recovered_T_mse": result.get("valid_metrics", {}).get("recovered_temperature_mse"),
            "valid_stress_recovered_T_mse": (
                result.get("valid_stress_metrics", {}) or {}
            ).get("recovered_temperature_mse"),
            "valid_recovered_T_rmse_K": result.get("valid_metrics", {}).get("recovered_T_rmse_K"),
            "valid_iid_recovered_T_rmse_K": result.get("valid_metrics", {}).get("recovered_T_rmse_K"),
            "valid_stress_recovered_T_rmse_K": (
                result.get("valid_stress_metrics", {}) or {}
            ).get("recovered_T_rmse_K"),
            "valid_relative_rmse_pct_v4": result.get("valid_metrics", {}).get("rel_rmse_v4_pct"),
            "valid_iid_relative_rmse_pct_v4": result.get("valid_metrics", {}).get("rel_rmse_v4_pct"),
            "valid_stress_relative_rmse_pct_v4": (
                result.get("valid_stress_metrics", {}) or {}
            ).get("rel_rmse_v4_pct"),
            "relative_metric_denominator_mean_abs_true_deltaT": result.get("valid_metrics", {}).get(
                "mean_abs_true_deltaT"
            ),
            "valid_base_mse": result.get("final_valid_loss_components", {}).get("base_mse"),
            "valid_iid_base_mse": (result.get("final_valid_iid_loss_components", {}) or {}).get("base_mse"),
            "valid_stress_base_mse": (result.get("final_valid_stress_loss_components", {}) or {}).get("base_mse"),
        }
    raise ValueError(f"unsupported checkpoint kind: {kind}")


def _checkpoint_run_metadata(
    *,
    sample_root: Path,
    args: argparse.Namespace,
    split_source: str,
    split_counts: dict[str, int],
    model_config: dict[str, Any],
    loss_config: dict[str, Any],
    lr_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    seed_config: dict[str, Any],
    batch_config: dict[str, Any],
    graph_config: dict[str, Any],
    global_context_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "subset": str(sample_root),
        "split_map_path": str(args.split_map) if args.split_map is not None else None,
        "split_source": split_source,
        "split_counts": split_counts,
        "epochs": int(args.epochs),
        "output_dir": str(args.output_dir),
        "model_config": model_config,
        "loss_config": loss_config,
        "lr_config": lr_config,
        "optimizer_config": optimizer_config,
        "seed_config": seed_config,
        "batch_config": batch_config,
        "graph_config": graph_config,
        "global_context": global_context_payload,
        "boundary_mask_fallback": bool(args.boundary_mask_fallback),
        "prediction_split": args.prediction_split,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "checkpoint_load_mode": "params_only" if args.init_checkpoint is not None else None,
        "checkpoint_load_strict": args.checkpoint_load_strict,
        "partial_load_policy": args.partial_load_policy,
    }


def _write_params_checkpoint(
    path: Path,
    *,
    params: Any,
    model_config: dict[str, Any],
    stats: dict[str, Any],
    kind: str,
    epoch: int | None,
    record: dict[str, Any],
    run_metadata: dict[str, Any],
) -> None:
    host_params = _host_params(params)
    param_tree_summary = _param_tree_summary(host_params)
    train_stats = _stats_payload(stats)
    payload = {
        "schema_version": "heat3d_v3_params_checkpoint_v1",
        "checkpoint_format_version": 1,
        "checkpoint_kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _current_git_commit(),
        "params": host_params,
        "model_config": _json_safe(model_config),
        "train_only_normalization": train_stats,
        "epoch": int(epoch) if epoch is not None else None,
        "record": _json_safe(record),
        "run_config_metadata": _json_safe(run_metadata),
        "param_tree_summary": param_tree_summary,
        "param_count": param_tree_summary["param_count"],
        "param_shapes": param_tree_summary["param_shapes"],
        "model_config_hash": _stable_json_hash(model_config),
        "train_stats_hash": _stable_json_hash(train_stats),
        "load_policy_intended": "params_only",
        "optimizer_state_saved": False,
        "warm_start_supported": True,
        "warm_start_mode": "params_only",
    }
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def _stats_payload(stats: dict) -> dict[str, Any]:
    return {
        "feature_names": list(stats["feature_names"]),
        "target_delta_mean": float(stats["target_delta_mean"].reshape(-1)[0]),
        "target_delta_std": float(stats["target_delta_std"].reshape(-1)[0]),
        "condition_mean": [float(value) for value in stats["condition_mean"].reshape(-1)],
        "condition_std": [float(value) for value in stats["condition_std"].reshape(-1)],
    }


def _metrics_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
        for key, value in metrics.items()
    }


def _validation_metric_scalars(metrics: dict[str, Any]) -> dict[str, Any]:
    relative_pct = _metric_error_pct(metrics)
    return {
        "valid_raw_deltaT_mse": float(metrics["raw_delta_mse"]),
        "valid_raw_deltaT_rmse_K": float(metrics["raw_rmse_K"]),
        "valid_recovered_T_mse": float(metrics["recovered_temperature_mse"]),
        "valid_recovered_temperature_mse": float(metrics["recovered_temperature_mse"]),
        "valid_recovered_T_rmse_K": float(metrics["recovered_T_rmse_K"]),
        "valid_relative_rmse_pct_v4": relative_pct,
        "rel_rmse_v4_pct": relative_pct,
        "valid_raw_deltaT_relative_rmse_pct_v4": relative_pct,
        "raw_deltaT_relative_rmse_pct_v4": relative_pct,
        "relative_metric_denominator_mean_abs_true_deltaT": float(metrics["mean_abs_true_deltaT"]),
    }


def _print_startup_summary(
    args: argparse.Namespace,
    *,
    sample_root: Path,
    split_counts: dict[str, int],
    split_source: str,
    primary_validation_split: str,
    stress_validation_split: str | None,
    output_dir: Path,
    loss_config: dict[str, Any],
    lr_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    seed_config: dict[str, int],
    model_config: dict[str, Any],
    batch_config: dict[str, Any],
) -> None:
    if args.log_mode == "quiet":
        return

    _emit("Heat3D v1 medium controlled training export smoke")
    _emit("  scope: research reference diagnostics only; not formal model performance")
    _emit(f"  subset: {sample_root}")
    _emit(f"  split counts: {split_counts}")
    _emit(
        "  split mode: "
        f"source={split_source} split_map={args.split_map} "
        f"primary_validation_split={primary_validation_split} "
        f"stress_validation_split={stress_validation_split}"
    )
    _emit(
        "  run: "
        f"epochs={args.epochs} lr={args.lr} lr_schedule={lr_config['lr_schedule']} "
        f"optimizer={optimizer_config['optimizer']} seed={seed_config['seed']} "
        f"model_seed={seed_config['model_seed']} "
        f"batch_order_seed={seed_config['batch_order_seed']} "
        f"graph_seed={seed_config['graph_seed']} "
        f"report_every={args.report_every}"
    )
    _emit(
        "  model: "
        f"node_latent_size={model_config['node_latent_size']} "
        f"edge_latent_size={model_config['edge_latent_size']} "
        f"processor_steps={model_config['processor_steps']} "
        f"mlp_hidden_layers={model_config['mlp_hidden_layers']} "
        f"p_edge_masking={model_config.get('p_edge_masking', 0.0)}"
    )
    _emit(
        "  output: "
        f"dir={output_dir} save_predictions={bool(args.save_predictions)} "
        f"save_best_predictions={bool(args.save_best_predictions)}"
    )
    _emit(
        "  batching: "
        f"mode={'mini_batch' if batch_config['batch_size'] is not None else 'legacy_full_batch'} "
        f"batch_plan={batch_config['batch_plan']} "
        f"batch_build_seed={batch_config['batch_build_seed']} "
        f"batch_size={batch_config['batch_size']} "
        f"validation_batch_size={batch_config['validation_batch_size']} "
        f"prediction_batch_size={batch_config['prediction_batch_size']} "
        f"shuffle_train_batches={batch_config['shuffle_train_batches']} "
        f"drop_last={batch_config['drop_last']}"
    )
    _emit(
        "  logging: "
        f"log_mode={args.log_mode} progress_log={bool(args.progress_log)} "
        f"progress_detail={args.progress_detail}"
    )
    _emit(
        "  selection: "
        f"metric={args.selection_metric} best_predictions_name={args.best_predictions_name}"
    )
    _emit(
        "  checkpoint init: "
        f"init_checkpoint={args.init_checkpoint if args.init_checkpoint is not None else 'none'} "
        f"load_mode={'params_only' if args.init_checkpoint is not None else 'none'} "
        f"strict={args.checkpoint_load_strict} "
        f"partial_load_policy={args.partial_load_policy}"
    )
    if args.log_mode == "compact":
        _emit(
            "  loss: "
            f"mode={loss_config['loss_mode']} weight_schedule={loss_config['loss_weight_schedule']} "
            f"bg_q={loss_config['background_quantile']} hot_q={loss_config['hotspot_quantile']} "
            f"strong_q={loss_config['strong_q_quantile']} "
            f"rel_w={loss_config['background_relative_weight']} hot_w={loss_config['hotspot_weight']} "
            f"strong_q_w={loss_config['strong_q_weight']} "
            f"pn_type={loss_config['pseudo_negative_loss_type']} pn_w={loss_config['pseudo_negative_weight']}"
        )
    else:
        _emit(
            "  lr schedule params: "
            f"warmup_epochs={lr_config['warmup_epochs']} min_lr={lr_config['min_lr']} "
            f"second_stage_epoch={lr_config['second_stage_epoch']} "
            f"second_stage_lr={lr_config['second_stage_lr']} "
            f"lr_init={lr_config['lr_init']} lr_peak={lr_config['lr_peak']} "
            f"lr_base={lr_config['lr_base']} lr_lowr={lr_config['lr_lowr']} "
            f"pct_start={lr_config['pct_start']} pct_final={lr_config['pct_final']}"
        )
        _emit(
            "  optimizer params: "
            f"optimizer={optimizer_config['optimizer']} "
            f"gradient_clip_norm={optimizer_config['gradient_clip_norm']} "
            f"weight_decay={optimizer_config['weight_decay']}"
        )
        _emit(f"  loss mode: {loss_config['loss_mode']}")
        _emit(f"  loss space: {loss_config['loss_space']}")
        _emit(
            "  loss params: "
            f"background_quantile={loss_config['background_quantile']} "
            f"hotspot_quantile={loss_config['hotspot_quantile']} "
            f"strong_q_quantile={loss_config['strong_q_quantile']} "
            f"background_weight={loss_config['background_weight']} "
            f"hotspot_weight={loss_config['hotspot_weight']} "
            f"strong_q_weight={loss_config['strong_q_weight']} "
            f"background_l1_weight={loss_config['background_l1_weight']} "
            f"background_bias_weight={loss_config['background_bias_weight']} "
            f"background_over_weight={loss_config['background_over_weight']} "
            f"background_relative_weight={loss_config['background_relative_weight']} "
            f"relative_floor={loss_config['relative_floor']} "
            f"relative_floor_mode={loss_config['relative_floor_mode']} "
            f"pseudo_negative_quantile={loss_config['pseudo_negative_quantile']} "
            f"pseudo_negative_delta_threshold={loss_config['pseudo_negative_delta_threshold']} "
            f"pseudo_negative_weight={loss_config['pseudo_negative_weight']} "
            f"pseudo_negative_over_margin={loss_config['pseudo_negative_over_margin']} "
            f"pseudo_negative_min_count={loss_config['pseudo_negative_min_count']} "
            f"pseudo_negative_loss_type={loss_config['pseudo_negative_loss_type']} "
            f"pseudo_negative_relative_floor={loss_config['pseudo_negative_relative_floor']}"
        )
        _emit(f"  loss weight schedule: {_loss_weight_schedule_payload(loss_config)}")
    _emit("  feature mode: relative BC features, diag3 k encoding, zero_delta_u_bridge")
    _emit(
        "  target mode: normalized DeltaT target; normalized 0 is train mean raw DeltaT, "
        "not raw DeltaT=0"
    )


def _print_final_summary(
    args: argparse.Namespace,
    *,
    result: dict[str, Any],
    loss_config: dict[str, Any],
    lr_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    model_config: dict[str, Any],
    predictions_path: Path,
    predictions_saved: bool,
    prediction_count: int,
    best_predictions_path: Path | None,
    best_predictions_saved: bool,
    best_prediction_count: int,
    final_checkpoint_path: Path | None,
    final_checkpoint_saved: bool,
    best_checkpoint_path: Path | None,
    best_checkpoint_saved: bool,
    final_prediction_export_skipped: bool,
    final_prediction_export_skip_reason: str | None,
    timings: dict[str, float],
) -> None:
    lr_history_summary = _sequence_summary(result["lr_history"])
    relative_weight_summary = _history_field_summary(
        result["loss_weight_history"], "current_background_relative_weight"
    )
    hotspot_weight_summary = _history_field_summary(result["loss_weight_history"], "current_hotspot_weight")
    strong_q_weight_summary = _history_field_summary(result["loss_weight_history"], "current_strong_q_weight")
    best = result.get("best_record") or {}

    _emit("")
    _emit("summary")
    _emit(
        "  final: "
        f"epoch={result['final_epoch']} valid_loss={result['final_valid_loss']:.8e} "
        f"valid_base_mse={result['final_valid_loss_components']['base_mse']:.8e} "
        f"valid_raw_deltaT_mse={result['valid_metrics']['raw_delta_mse']:.8e} "
        f"raw_rmse_K={_format_progress_sigfig_decimal(result['valid_metrics']['raw_rmse_K'])} "
        f"rel_rmse_v4_pct={_format_progress_percent(result['valid_metrics'].get('rel_rmse_v4_pct'))}"
    )
    if result.get("primary_validation_split") == "valid_iid":
        _emit(
            "  final stratified: "
            f"valid_iid_loss={result['final_valid_iid_loss']:.8e} "
            f"valid_stress_loss={_format_progress_value(result.get('final_valid_stress_loss'))} "
            f"valid_stress_raw_deltaT_mse={_format_progress_value(result.get('final_valid_stress_raw_deltaT_mse'))}"
        )
    _emit(
        "  best-valid: "
        f"metric={args.selection_metric} epoch={best.get('epoch')} "
        f"valid_loss={best.get('valid_loss'):.8e} "
        f"valid_base_mse={best.get('valid_base_mse'):.8e} "
        f"valid_raw_deltaT_mse={best.get('valid_raw_deltaT_mse'):.8e} "
        f"raw_rmse_K={_format_progress_sigfig_decimal(best.get('valid_raw_deltaT_rmse_K'))} "
        f"rel_rmse_v4_pct={_format_progress_percent(best.get('valid_rel_rmse_v4_pct'))}"
    )
    _emit(
        "  predictions: "
        f"final_saved={bool(predictions_saved)} final_path={predictions_path if predictions_saved else 'not_written'} "
        f"final_count={prediction_count} best_saved={bool(best_predictions_saved)} "
        f"best_path={best_predictions_path if best_predictions_saved else 'not_written'} "
        f"best_count={best_prediction_count} "
        f"final_export_skipped={bool(final_prediction_export_skipped)} "
        f"final_export_skip_reason={final_prediction_export_skip_reason}"
    )
    _emit(
        "  checkpoints: "
        f"final_saved={bool(final_checkpoint_saved)} "
        f"final_path={final_checkpoint_path if final_checkpoint_saved else 'not_written'} "
        f"best_saved={bool(best_checkpoint_saved)} "
        f"best_path={best_checkpoint_path if best_checkpoint_saved else 'not_written'}"
    )
    _emit(
        "  status: "
        f"grad_finite={result['grad_finite']} "
        f"checkpoint_saved={bool(final_checkpoint_saved or best_checkpoint_saved)} "
        f"export_smoke_ok={result['status_ok']}"
    )

    if args.log_mode == "full":
        _emit("  loss/optimization")
        _emit(f"    model config: {model_config}")
        _emit(f"    loss mode: {loss_config['loss_mode']}")
        _emit(f"    loss weight schedule: {loss_config['loss_weight_schedule']}")
        _emit(f"    relative weight summary: {relative_weight_summary}")
        _emit(f"    hotspot weight summary: {hotspot_weight_summary}")
        _emit(f"    strong-q weight summary: {strong_q_weight_summary}")
        _emit(f"    lr schedule: {lr_config['lr_schedule']}")
        _emit(f"    lr history summary: {lr_history_summary}")
        _emit("  loss initial/final")
        _emit(f"    train loss initial/final: {result['train_losses'][0]:.8e} -> {result['train_losses'][-1]:.8e}")
        _emit(f"    valid loss initial/final: {result['valid_losses'][0]:.8e} -> {result['valid_losses'][-1]:.8e}")
        _emit("  final base/raw/recovered metrics")
        _emit(f"    final train base MSE: {result['final_train_loss_components']['base_mse']:.8e}")
        _emit(f"    final valid base MSE: {result['final_valid_loss_components']['base_mse']:.8e}")
        _emit(f"    final train raw DeltaT MSE: {result['train_metrics']['raw_delta_mse']:.8e}")
        _emit(f"    final valid raw DeltaT MSE: {result['valid_metrics']['raw_delta_mse']:.8e}")
        _emit(f"    final train recovered temperature MSE: {result['train_metrics']['recovered_temperature_mse']:.8e}")
        _emit(f"    final valid recovered temperature MSE: {result['valid_metrics']['recovered_temperature_mse']:.8e}")
        _emit("  final background metrics")
        _emit(f"    final train background penalty: {result['final_train_loss_components']['background_penalty']:.8e}")
        _emit(f"    final valid background penalty: {result['final_valid_loss_components']['background_penalty']:.8e}")
        _emit(f"    final train background L1: {result['final_train_loss_components']['background_l1']:.8e}")
        _emit(f"    final valid background L1: {result['final_valid_loss_components']['background_l1']:.8e}")
        _emit(
            "    final train background signed bias loss: "
            f"{result['final_train_loss_components']['background_signed_bias_loss']:.8e}"
        )
        _emit(
            "    final valid background signed bias loss: "
            f"{result['final_valid_loss_components']['background_signed_bias_loss']:.8e}"
        )
        _emit(
            "    final train background overprediction loss: "
            f"{result['final_train_loss_components']['background_overprediction_loss']:.8e}"
        )
        _emit(
            "    final valid background overprediction loss: "
            f"{result['final_valid_loss_components']['background_overprediction_loss']:.8e}"
        )
        _emit(
            "    final train background relative abs: "
            f"{result['final_train_loss_components']['background_relative_abs']:.8e}"
        )
        _emit(
            "    final valid background relative abs: "
            f"{result['final_valid_loss_components']['background_relative_abs']:.8e}"
        )
        _emit(f"    final train pseudo-negative count: {result['final_train_loss_components']['pseudo_negative_count']:.8e}")
        _emit(f"    final valid pseudo-negative count: {result['final_valid_loss_components']['pseudo_negative_count']:.8e}")
        _emit(
            "    final train pseudo-negative over loss: "
            f"{result['final_train_loss_components']['pseudo_negative_over_loss']:.8e}"
        )
        _emit(
            "    final valid pseudo-negative over loss: "
            f"{result['final_valid_loss_components']['pseudo_negative_over_loss']:.8e}"
        )
        _emit(
            "    final train pseudo-negative weighted loss: "
            f"{result['final_train_loss_components']['pseudo_negative_weighted_loss']:.8e}"
        )
        _emit(
            "    final valid pseudo-negative weighted loss: "
            f"{result['final_valid_loss_components']['pseudo_negative_weighted_loss']:.8e}"
        )
        _emit(
            "    final train pseudo-negative weighted fraction: "
            f"{result['final_train_loss_components']['pseudo_negative_weighted_fraction_of_total_loss']:.8e}"
        )
        _emit(
            "    final valid pseudo-negative weighted fraction: "
            f"{result['final_valid_loss_components']['pseudo_negative_weighted_fraction_of_total_loss']:.8e}"
        )
        _emit(f"    final train pseudo-negative bias: {result['final_train_loss_components']['pseudo_negative_bias']:.8e}")
        _emit(f"    final valid pseudo-negative bias: {result['final_valid_loss_components']['pseudo_negative_bias']:.8e}")
        _emit(
            "    final train pseudo-negative over ratio: "
            f"{result['final_train_loss_components']['pseudo_negative_over_ratio']:.8e}"
        )
        _emit(
            "    final valid pseudo-negative over ratio: "
            f"{result['final_valid_loss_components']['pseudo_negative_over_ratio']:.8e}"
        )
        _emit(f"    final train bg pred raw mean: {result['final_train_loss_components']['bg_pred_raw_mean']:.8e}")
        _emit(f"    final valid bg pred raw mean: {result['final_valid_loss_components']['bg_pred_raw_mean']:.8e}")
        _emit(f"    final train bg signed bias: {result['final_train_loss_components']['bg_signed_bias']:.8e}")
        _emit(f"    final valid bg signed bias: {result['final_valid_loss_components']['bg_signed_bias']:.8e}")
        _emit(f"    final train bg abs mean: {result['final_train_loss_components']['bg_abs_mean']:.8e}")
        _emit(f"    final valid bg abs mean: {result['final_valid_loss_components']['bg_abs_mean']:.8e}")
        _emit("  final hotspot metrics")
        _emit(f"    final train hotspot retention loss: {result['final_train_loss_components']['hotspot_retention_loss']:.8e}")
        _emit(f"    final valid hotspot retention loss: {result['final_valid_loss_components']['hotspot_retention_loss']:.8e}")
        _emit(f"    final train hotspot MSE: {result['final_train_loss_components']['hotspot_mse']:.8e}")
        _emit(f"    final valid hotspot MSE: {result['final_valid_loss_components']['hotspot_mse']:.8e}")
        _emit(f"    final train strong-q MSE: {result['final_train_loss_components']['strong_q_mse']:.8e}")
        _emit(f"    final valid strong-q MSE: {result['final_valid_loss_components']['strong_q_mse']:.8e}")
        _emit(
            "    final valid hotspot mask fraction: "
            f"{result['final_valid_loss_components']['hotspot_mask_fraction']:.8e}"
        )
        _emit(
            "    final valid strong-q mask fraction: "
            f"{result['final_valid_loss_components']['strong_q_mask_fraction']:.8e}"
        )
        _emit(f"    final train hotspot raw MAE: {result['final_train_loss_components']['hotspot_raw_mae']:.8e}")
        _emit(f"    final valid hotspot raw MAE: {result['final_valid_loss_components']['hotspot_raw_mae']:.8e}")
    else:
        _emit(
            "  optimization: "
            f"optimizer={optimizer_config['optimizer']} "
            f"loss_mode={loss_config['loss_mode']} loss_weight_schedule={loss_config['loss_weight_schedule']} "
            f"lr_schedule={lr_config['lr_schedule']} lr_summary={lr_history_summary}"
        )
        _emit(
            "  final background/hotspot: "
            f"valid_bg_bias={result['final_valid_loss_components']['bg_signed_bias']:.8e} "
            f"valid_bg_rel={result['final_valid_loss_components']['background_relative_abs']:.8e} "
            f"valid_pn_bias={result['final_valid_loss_components']['pseudo_negative_bias']:.8e} "
            f"valid_pn_over={result['final_valid_loss_components']['pseudo_negative_over_loss']:.8e} "
            f"valid_pn_over_ratio={result['final_valid_loss_components']['pseudo_negative_over_ratio']:.8e} "
            f"valid_pn_weighted_fraction={result['final_valid_loss_components']['pseudo_negative_weighted_fraction_of_total_loss']:.8e} "
            f"valid_hotspot_mae={result['final_valid_loss_components']['hotspot_raw_mae']:.8e} "
            f"valid_hotspot_mse={result['final_valid_loss_components']['hotspot_mse']:.8e} "
            f"valid_strong_q_mse={result['final_valid_loss_components']['strong_q_mse']:.8e}"
        )

    _progress(_progress_enabled(args), "startup-summary", _timing_summary(timings))
    _progress(_progress_enabled(args), "done", "script complete")


def _run_logged_diagnostic_command(
    command: list[str],
    *,
    log_prefix: str,
    cwd: Path = REPO_DIR,
) -> int:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.stdout:
        for line in completed.stdout.rstrip().splitlines():
            _emit(f"[{log_prefix}] {line}")
    if completed.stderr:
        for line in completed.stderr.rstrip().splitlines():
            _emit(f"[{log_prefix}:stderr] {line}", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"post-training diagnostic command failed with returncode={completed.returncode}: "
            + " ".join(command)
        )
    return int(completed.returncode)


def _run_post_training_prediction_diagnostics(
    args: argparse.Namespace,
    *,
    sample_root: Path,
    output_dir: Path,
    predictions_path: Path,
    predictions_saved: bool,
    best_predictions_path: Path | None,
    best_predictions_saved: bool,
    timings: dict[str, float],
    progress_enabled: bool,
) -> dict[str, Any]:
    if not args.post_training_diagnostics:
        return {"enabled": False, "reason": "disabled"}
    if args.prediction_split != "all":
        diagnostics_dir = args.post_training_diagnostics_output_dir or (
            output_dir / "post_training_diagnostics"
        )
        return {
            "enabled": False,
            "reason": "non_all_prediction_split",
            "prediction_split": args.prediction_split,
            "output_dir": str(diagnostics_dir),
        }

    diag_start = time.perf_counter()
    diagnostics_dir = args.post_training_diagnostics_output_dir or (
        output_dir / "post_training_diagnostics"
    )
    entries: list[tuple[str, Path]] = []
    if predictions_saved and predictions_path.is_file():
        entries.append(("final", predictions_path))
    if best_predictions_saved and best_predictions_path is not None and best_predictions_path.is_file():
        entries.append(("best", best_predictions_path))
    if not entries:
        result = {
            "enabled": True,
            "reason": "no_saved_predictions",
            "output_dir": str(diagnostics_dir),
            "entries": [],
        }
        _record_timing(timings, "post_training_diagnostics", diag_start)
        return result

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    loss_summary_path = output_dir / "loss_summary.json"
    entry_payloads: list[dict[str, Any]] = []
    _progress(
        progress_enabled,
        "diagnostics",
        (
            "running post-training diagnostics "
            f"labels={[label for label, _ in entries]} output_dir={diagnostics_dir}"
        ),
    )
    split_map_args = (
        ["--split-map", str(args.split_map)]
        if args.split_map is not None
        else []
    )

    for label, prediction_path in entries:
        baseline_json = diagnostics_dir / f"baseline_comparison_{label}.json"
        error_json = diagnostics_dir / f"error_bins_{label}.json"
        error_md = diagnostics_dir / f"error_bins_{label}.md"
        condition_json = diagnostics_dir / f"condition_diagnostics_{label}.json"
        condition_md = diagnostics_dir / f"condition_diagnostics_{label}.md"
        field_json = diagnostics_dir / f"field_shape_diagnostics_{label}.json"
        field_md = diagnostics_dir / f"field_shape_diagnostics_{label}.md"
        mechanism_json = diagnostics_dir / f"mechanism_{label}.json"
        mechanism_md = diagnostics_dir / f"mechanism_{label}.md"
        run_analysis_json = diagnostics_dir / f"run_analysis_{label}.json"
        run_analysis_md = diagnostics_dir / f"run_analysis_{label}.md"

        commands = [
            (
                "baseline",
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "compare_heat3d_v1_medium_baselines.py"),
                    "--subset",
                    str(sample_root),
                    *split_map_args,
                    "--trained-predictions",
                    str(prediction_path),
                    "--output-json",
                    str(baseline_json),
                    "--stdout-mode",
                    "compact",
                ],
            ),
            (
                "error-bins",
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "analyze_heat3d_v1_medium_error_bins.py"),
                    "--subset",
                    str(sample_root),
                    *split_map_args,
                    "--trained-predictions",
                    str(prediction_path),
                    "--output-json",
                    str(error_json),
                    "--output-md",
                    str(error_md),
                    "--stdout-mode",
                    "compact",
                ],
            ),
            (
                "condition",
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "analyze_heat3d_v1_medium_condition_diagnostics.py"),
                    "--subset",
                    str(sample_root),
                    *split_map_args,
                    "--trained-predictions",
                    str(prediction_path),
                    "--prediction-label",
                    label,
                    "--output-json",
                    str(condition_json),
                    "--output-md",
                    str(condition_md),
                    "--stdout-mode",
                    "compact",
                ],
            ),
            (
                "field-shape",
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "analyze_heat3d_v2_field_shape_diagnostics.py"),
                    "--subset",
                    str(sample_root),
                    *split_map_args,
                    "--trained-predictions",
                    str(prediction_path),
                    "--prediction-label",
                    label,
                    "--output-json",
                    str(field_json),
                    "--output-md",
                    str(field_md),
                    "--stdout-mode",
                    "compact",
                ],
            ),
            (
                "mechanism",
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "analyze_heat3d_v3_prediction_mechanisms.py"),
                    "--run-dir",
                    str(output_dir),
                    "--prediction-name",
                    prediction_path.name,
                    "--prediction-label",
                    label,
                    "--subset",
                    str(sample_root),
                    *split_map_args,
                    "--output-json",
                    str(mechanism_json),
                    "--output-md",
                    str(mechanism_md),
                ],
            ),
            (
                "run-summary",
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "analyze_heat3d_v1_medium_run_summary.py"),
                    "--run-dir",
                    str(output_dir),
                    "--loss-summary",
                    str(loss_summary_path),
                    "--baseline-comparison-json",
                    str(baseline_json),
                    "--error-bins-json",
                    str(error_json),
                    "--prediction-label",
                    label,
                    "--output-json",
                    str(run_analysis_json),
                    "--output-md",
                    str(run_analysis_md),
                    "--stdout-mode",
                    "compact",
                ],
            ),
        ]
        for command_label, command in commands:
            _run_logged_diagnostic_command(
                command,
                log_prefix=f"diagnostics:{label}:{command_label}",
            )

        mechanism_payload = json.loads(mechanism_json.read_text(encoding="utf-8"))
        overall = mechanism_payload.get("overall") or {}
        _emit(
            f"post_diagnostics {label}: "
            f"raw_RMSE={_format_progress_loss(overall.get('rmse'))} "
            f"zRMSE={_format_progress_loss(overall.get('zscore_rmse'))} "
            f"top_k={_format_progress_loss(overall.get('top_k_overlap'))} "
            f"peak_rel={_format_progress_loss(overall.get('peak_rel_error'))} "
            f"dir={diagnostics_dir}"
        )
        entry_payloads.append(
            {
                "label": label,
                "prediction_path": str(prediction_path),
                "baseline_comparison_json": str(baseline_json),
                "error_bins_json": str(error_json),
                "error_bins_md": str(error_md),
                "condition_diagnostics_json": str(condition_json),
                "condition_diagnostics_md": str(condition_md),
                "field_shape_diagnostics_json": str(field_json),
                "field_shape_diagnostics_md": str(field_md),
                "mechanism_json": str(mechanism_json),
                "mechanism_md": str(mechanism_md),
                "run_analysis_json": str(run_analysis_json),
                "run_analysis_md": str(run_analysis_md),
                "summary": {
                    "raw_RMSE": overall.get("rmse"),
                    "zRMSE": overall.get("zscore_rmse"),
                    "top_k_overlap": overall.get("top_k_overlap"),
                    "peak_rel": overall.get("peak_rel_error"),
                },
            }
        )

    region_json = diagnostics_dir / "region_error_decomposition.json"
    region_md = diagnostics_dir / "region_error_decomposition.md"
    region_command = [
        sys.executable,
        str(SCRIPTS_DIR / "analyze_heat3d_v3_region_error_decomposition.py"),
        "--subset",
        str(sample_root),
        *split_map_args,
        "--output-json",
        str(region_json),
        "--output-md",
        str(region_md),
    ]
    for label, prediction_path in entries:
        region_command.extend(["--entry", f"{label}={output_dir}:{prediction_path.name}"])
    _run_logged_diagnostic_command(
        region_command,
        log_prefix="diagnostics:region",
    )

    result = {
        "enabled": True,
        "output_dir": str(diagnostics_dir),
        "entries": entry_payloads,
        "region_error_decomposition_json": str(region_json),
        "region_error_decomposition_md": str(region_md),
    }
    _record_timing(timings, "post_training_diagnostics", diag_start)
    _progress(progress_enabled, "diagnostics", "post-training diagnostics complete", diag_start)
    return result


def _final_probe_eval_kinds(kind: str) -> tuple[str, ...]:
    if kind == "both":
        return ("best", "final")
    if kind in {"best", "final"}:
        return (kind,)
    raise ValueError(f"Unsupported final probe checkpoint kind: {kind}")


def _final_probe_checkpoint_entries(
    args: argparse.Namespace,
    *,
    final_checkpoint_path: Path | None,
    final_checkpoint_saved: bool,
    best_checkpoint_path: Path | None,
    best_checkpoint_saved: bool,
) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for kind in _final_probe_eval_kinds(args.final_probe_checkpoint_kind):
        if kind == "best":
            if not best_checkpoint_saved or best_checkpoint_path is None:
                raise RuntimeError(
                    "final probe eval requested best checkpoint, but params_best.pkl was not saved"
                )
            entries.append(("best", best_checkpoint_path))
        elif kind == "final":
            if not final_checkpoint_saved or final_checkpoint_path is None:
                raise RuntimeError(
                    "final probe eval requested final checkpoint, but params_final.pkl was not saved"
                )
            entries.append(("final", final_checkpoint_path))
    return entries


def _print_final_probe_eval_table(label: str, metrics_path: Path) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("metrics") or []
    _emit(f"final_probe {label}: sample_count={len(rows)} metrics={metrics_path}")
    for row in rows:
        probe_id = row.get("probe_id", "unknown")
        flags = ""
        if probe_id == "P10":
            flags = (
                " localized_top_contact_supported="
                f"{row.get('localized_top_contact_supported')} "
                "side_asymmetry_supported="
                f"{row.get('side_asymmetry_supported')}"
            )
        _emit(
            f"final_probe {label} {probe_id} "
            f"RMSE={_format_progress_loss(row.get('RMSE'))} "
            f"MAE={_format_progress_loss(row.get('MAE'))} "
            f"relRMSE_DeltaT={_format_progress_loss(row.get('relative_RMSE_on_DeltaT'))} "
            f"Tmax_error={_format_progress_loss(row.get('Tmax_error'))} "
            f"top5_RMSE={_format_progress_loss(row.get('top_5_percent_RMSE'))}"
            f"{flags}"
        )


def _run_post_training_final_probe_eval(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    final_checkpoint_path: Path | None,
    final_checkpoint_saved: bool,
    best_checkpoint_path: Path | None,
    best_checkpoint_saved: bool,
    timings: dict[str, float],
    progress_enabled: bool,
) -> dict[str, Any]:
    if not args.final_probe_eval_after_training:
        return {
            "enabled": False,
            "reason": "disabled",
            "checkpoint_kind": args.final_probe_checkpoint_kind,
        }

    eval_start = time.perf_counter()
    run_config_path = output_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"final probe eval requires run_config.json: {run_config_path}")
    if not args.final_probe_subset.is_dir():
        raise FileNotFoundError(f"final probe subset not found: {args.final_probe_subset}")
    if not args.final_probe_provenance.is_file():
        raise FileNotFoundError(f"final probe provenance not found: {args.final_probe_provenance}")

    eval_output_dir = args.final_probe_output_dir or (output_dir / "final_probe_eval")
    entries = _final_probe_checkpoint_entries(
        args,
        final_checkpoint_path=final_checkpoint_path,
        final_checkpoint_saved=final_checkpoint_saved,
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_saved=best_checkpoint_saved,
    )
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "run_heat3d_v3_final_probe_checkpoint_smoke.py"),
        "--subset",
        str(args.final_probe_subset),
        "--provenance",
        str(args.final_probe_provenance),
        "--output-dir",
        str(eval_output_dir),
        "--batch-size",
        str(args.final_probe_batch_size),
    ]
    for label, checkpoint_path in entries:
        command.extend(
            [
                "--checkpoint-entry",
                f"{label}={checkpoint_path}={run_config_path}",
            ]
        )

    _progress(
        progress_enabled,
        "final-probe",
        (
            "running post-training final probe checkpoint inference "
            f"kinds={[label for label, _ in entries]} output_dir={eval_output_dir}"
        ),
    )
    completed = subprocess.run(
        command,
        cwd=REPO_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.stdout:
        for line in completed.stdout.rstrip().splitlines():
            _emit(f"[final-probe] {line}")
    if completed.stderr:
        for line in completed.stderr.rstrip().splitlines():
            _emit(f"[final-probe:stderr] {line}", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            "post-training final probe inference failed with "
            f"returncode={completed.returncode}"
        )

    entry_payloads = []
    for label, checkpoint_path in entries:
        metrics_path = eval_output_dir / label / "metrics" / "s5_probe_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"final probe metrics not found: {metrics_path}")
        _print_final_probe_eval_table(label, metrics_path)
        entry_payloads.append(
            {
                "label": label,
                "checkpoint_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
                "output_dir": str(eval_output_dir / label),
            }
        )

    result = {
        "enabled": True,
        "checkpoint_kind": args.final_probe_checkpoint_kind,
        "output_dir": str(eval_output_dir),
        "subset": str(args.final_probe_subset),
        "provenance": str(args.final_probe_provenance),
        "batch_size": int(args.final_probe_batch_size),
        "command": command,
        "returncode": int(completed.returncode),
        "entries": entry_payloads,
        "comparison_json": str(eval_output_dir / "s5_family_final_probe_comparison.json"),
        "comparison_md": str(eval_output_dir / "s5_family_final_probe_comparison.md"),
    }
    _record_timing(timings, "final_probe_eval", eval_start)
    _progress(progress_enabled, "final-probe", "post-training final probe inference complete", eval_start)
    return result


def _sample_weight_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    default_weight = float(args.sample_weight_default)
    if default_weight < 0.0:
        raise ValueError("--sample-weight-default must be >= 0")
    if args.sample_weight_policy == "none":
        if args.sample_weight_json is not None:
            raise ValueError("--sample-weight-json requires --sample-weight-policy hard_sample_list")
    elif args.sample_weight_policy == "hard_sample_list":
        if args.sample_weight_json is None:
            raise ValueError("--sample-weight-policy hard_sample_list requires --sample-weight-json")
    return {
        "policy": args.sample_weight_policy,
        "json_path": str(args.sample_weight_json) if args.sample_weight_json is not None else None,
        "default": default_weight,
        "normalize": bool(args.sample_weight_normalize),
    }


def _load_hard_sample_weights(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if isinstance(payload.get("sample_weights"), dict):
            source = payload["sample_weights"]
        elif isinstance(payload.get("weights"), dict):
            source = payload["weights"]
        elif isinstance(payload.get("hard_samples"), list):
            source = payload["hard_samples"]
        elif isinstance(payload.get("samples"), list):
            source = payload["samples"]
        else:
            source = payload
    elif isinstance(payload, list):
        source = payload
    else:
        raise ValueError(f"Unsupported sample-weight JSON root type: {type(payload).__name__}")

    weights: dict[str, float] = {}
    if isinstance(source, dict):
        for sample_id, weight in source.items():
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Invalid sample id in sample-weight JSON: {sample_id!r}")
            weight_value = float(weight)
            if weight_value < 0.0:
                raise ValueError(f"Negative weight for sample {sample_id!r}: {weight_value}")
            weights[sample_id] = weight_value
        return weights

    if isinstance(source, list):
        for item in source:
            if isinstance(item, str):
                sample_id = item
                weight_value = 1.25
            elif isinstance(item, dict):
                sample_id = str(item.get("sample_id") or item.get("id") or "")
                weight_value = float(item.get("weight", 1.25))
            else:
                raise ValueError(f"Invalid sample-weight list item: {item!r}")
            if not sample_id:
                raise ValueError(f"Invalid sample id in sample-weight JSON list item: {item!r}")
            if weight_value < 0.0:
                raise ValueError(f"Negative weight for sample {sample_id!r}: {weight_value}")
            weights[sample_id] = weight_value
        return weights

    raise ValueError(f"Unsupported sample-weight JSON source type: {type(source).__name__}")


def _prepare_train_sample_weights(
    train_ids: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    policy = config["policy"]
    default_weight = float(config["default"])
    if policy == "none":
        summary = {
            "policy": "none",
            "json_path": None,
            "default": default_weight,
            "normalize": bool(config["normalize"]),
            "train_sample_count": int(len(train_ids)),
            "weighted_sample_count": 0,
            "mean": 1.0,
            "min": 1.0,
            "max": 1.0,
            "sum": float(len(train_ids)),
        }
        return {}, summary

    json_path = Path(str(config["json_path"]))
    explicit = _load_hard_sample_weights(json_path)
    weights = {sample_id: default_weight for sample_id in train_ids}
    unknown = sorted(sample_id for sample_id in explicit if sample_id not in weights)
    for sample_id, value in explicit.items():
        if sample_id in weights:
            weights[sample_id] = float(value)
    if config["normalize"]:
        total = float(sum(weights.values()))
        if total > 0.0:
            scale = float(len(train_ids)) / total
            weights = {sample_id: value * scale for sample_id, value in weights.items()}
    values = list(weights.values())
    summary = {
        "policy": policy,
        "json_path": str(json_path),
        "default": default_weight,
        "normalize": bool(config["normalize"]),
        "train_sample_count": int(len(train_ids)),
        "weighted_sample_count": int(sum(1 for sample_id in train_ids if sample_id in explicit)),
        "unknown_sample_count": int(len(unknown)),
        "unknown_samples": unknown[:50],
        "mean": float(np.mean(values)) if values else 0.0,
        "min": float(np.min(values)) if values else 0.0,
        "max": float(np.max(values)) if values else 0.0,
        "sum": float(np.sum(values)) if values else 0.0,
    }
    return weights, summary


def _attach_sample_weights_to_groups(groups: list[dict], sample_weights: dict[str, float]) -> None:
    if not sample_weights:
        return
    for group in groups:
        group["sample_weights"] = np.asarray(
            [float(sample_weights.get(sample_id, 1.0)) for sample_id in group["sample_ids"]],
            dtype=np.float32,
        )


def _global_context_row_for_example(example: Any) -> dict[str, float]:
    """Build one inference-only FiLM context from the active bridge view."""

    relative_view = example.get_relative_bc_feature_view()
    feature_names = tuple(relative_view.condition_feature_names)
    raw_condition = np.asarray(relative_view.condition_features, dtype=np.float64)
    raw_coords = np.asarray(example.condition.coords, dtype=np.float64).reshape(-1, 3)
    reference_temperature = float(relative_view.t_ref_value)
    if not math.isfinite(reference_temperature):
        raise ValueError(f"{example.sample_id}: global context has invalid reference temperature")
    return global_context_from_raw_condition(
        coords=raw_coords,
        raw_condition=raw_condition,
        condition_feature_names=feature_names,
        reference_temperature_K=reference_temperature,
    )


def _prepare_global_context_lookup(
    model_config: Mapping[str, Any],
    *,
    train_examples: list[Any],
    required_examples: list[Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit V5 context standardization on train only and encode requested groups."""

    native_enabled = model_config.get("native_output_mode") == "native_shape_scale"
    if (
        model_config.get("global_context_mode", GLOBAL_CONTEXT_MODE_NONE) == GLOBAL_CONTEXT_MODE_NONE
        and not native_enabled
    ):
        return {}, {
            "enabled": False,
            "mode": GLOBAL_CONTEXT_MODE_NONE,
            "target_or_label_derived_inputs": False,
        }
    feature_names = tuple(model_config.get("global_context_feature_names") or ())
    validate_global_context_schema(feature_names)
    rows: dict[str, dict[str, float]] = {}
    for example in [*train_examples, *required_examples]:
        sample_id = str(example.sample_id)
        if sample_id not in rows:
            rows[sample_id] = _global_context_row_for_example(example)
    train_ids = [str(example.sample_id) for example in train_examples]
    standardizer = fit_train_only_standardizer(
        [rows[sample_id] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    encoded = {
        sample_id: standardize_contexts([row], standardizer)[0]
        for sample_id, row in rows.items()
    }
    return encoded, {
        "enabled": True,
        "mode": (
            GLOBAL_CONTEXT_MODE_FILM
            if model_config.get("global_context_mode") == GLOBAL_CONTEXT_MODE_FILM
            else "native_scale_head"
        ),
        "feature_names": list(GLOBAL_CONTEXT_FEATURES),
        "standardizer": standardizer,
        "target_or_label_derived_inputs": False,
    }


def _attach_global_context_to_groups(
    groups: list[dict],
    encoded_context_by_id: Mapping[str, np.ndarray],
    *,
    expected_feature_dim: int,
) -> None:
    if not encoded_context_by_id:
        return
    for group in groups:
        missing = [sample_id for sample_id in group["sample_ids"] if sample_id not in encoded_context_by_id]
        if missing:
            raise ValueError(f"{group['name']}: global context missing samples {missing[:5]}")
        context = np.stack([encoded_context_by_id[sample_id] for sample_id in group["sample_ids"]])
        if context.shape != (len(group["sample_ids"]), expected_feature_dim):
            raise ValueError(
                f"{group['name']}: global context shape {context.shape} does not match "
                f"batch/feature dimensions {len(group['sample_ids'])}/{expected_feature_dim}"
            )
        if not np.all(np.isfinite(context)):
            raise ValueError(f"{group['name']}: global context contains non-finite values")
        group["global_context"] = jnp.asarray(context, dtype=jnp.float32)


def _attach_native_physics_to_groups(
    groups: list[dict],
    examples_by_id: Mapping[str, Any],
) -> None:
    """Attach inference-only native physics tensors in each group's sample order."""

    for group in groups:
        volumes = []
        log_s_phys = []
        references = []
        masks = []
        prescribed = []
        for sample_id in group["sample_ids"]:
            example = examples_by_id[sample_id]
            relative = example.get_relative_bc_feature_view()
            names = tuple(relative.condition_feature_names)
            values = np.asarray(relative.condition_features, dtype=np.float64)
            coords = np.asarray(example.condition.coords, dtype=np.float64)
            if "is_bottom" not in names or "bottom_T_fixed_minus_T_ref" not in names:
                raise ValueError(f"{sample_id}: native branch lacks bottom Dirichlet features")
            bottom = values[:, names.index("is_bottom")] > 0.5
            if not np.any(bottom):
                raise ValueError(f"{sample_id}: native branch has no Dirichlet nodes")
            reference = float(relative.t_ref_value)
            offset = values[:, names.index("bottom_T_fixed_minus_T_ref")]
            context = _global_context_row_for_example(example)
            volumes.append(control_volume_weights(coords))
            log_s_phys.append(float(context["log_s_phys_K"]))
            references.append(np.full(coords.shape[0], reference, dtype=np.float32))
            masks.append(bottom.astype(np.float32))
            prescribed.append((reference + offset).astype(np.float32))
        group["native_physics"] = {
            "control_volumes": jnp.asarray(np.stack(volumes), dtype=jnp.float32),
            "log_s_phys": jnp.asarray(log_s_phys, dtype=jnp.float32),
            "reference_temperature": jnp.asarray(np.stack(references), dtype=jnp.float32),
            "dirichlet_mask": jnp.asarray(np.stack(masks), dtype=jnp.float32),
            "prescribed_temperature": jnp.asarray(np.stack(prescribed), dtype=jnp.float32),
        }


def _model_apply(model, params, group: Mapping[str, Any]):
    """Use the V4 call path unless a group explicitly carries Global FiLM data."""

    if "native_physics" in group:
        physics = group["native_physics"]
        return model.apply(
            {"params": params},
            inputs=group["inputs"],
            graphs=group["graphs"],
            global_context=group.get("global_context"),
            control_volumes=physics["control_volumes"],
            log_s_phys=physics["log_s_phys"],
            reference_temperature=physics["reference_temperature"],
            dirichlet_mask=physics["dirichlet_mask"],
            prescribed_temperature=physics["prescribed_temperature"],
            method=model.predict_native_shape_scale,
        )
    return model.apply(
        {"params": params},
        inputs=group["inputs"],
        graphs=group["graphs"],
        global_context=group.get("global_context"),
    )


def _model_init(model, key, group: Mapping[str, Any], inputs: Any):
    if "native_physics" in group:
        physics = group["native_physics"]
        return model.init(
            key,
            inputs=inputs,
            graphs=group["graphs"],
            global_context=group.get("global_context"),
            control_volumes=physics["control_volumes"],
            log_s_phys=physics["log_s_phys"],
            reference_temperature=physics["reference_temperature"],
            dirichlet_mask=physics["dirichlet_mask"],
            prescribed_temperature=physics["prescribed_temperature"],
            method=model.predict_native_shape_scale,
        )
    return model.init(
        key,
        inputs=inputs,
        graphs=group["graphs"],
        global_context=group.get("global_context"),
    )


def _metadata_key(graph_seed: int):
    return jax.random.PRNGKey(int(graph_seed))


def _build_batch_metadata_with_seed(
    builder: Heat3DGraphBuilder,
    coords_list: list[np.ndarray],
    *,
    graph_seed: int,
):
    metadata_list = [
        builder.build_metadata(coords, key=_metadata_key(graph_seed))
        for coords in coords_list
    ]
    same_coords = all(np.array_equal(coords_list[0], coords) for coords in coords_list[1:])
    if same_coords:
        return tree.tree_map(
            lambda value: jnp.repeat(value, repeats=len(coords_list), axis=0),
            metadata_list[0],
        ), True
    return tree.tree_map(lambda *values: jnp.concatenate(values, axis=0), *metadata_list), False


def _graph_coords_for_example(example, stats: dict) -> np.ndarray:
    if stats.get("coord_policy") != COORD_POLICY_SAMPLE_LOCAL_ISOTROPIC:
        return np.asarray(example.condition.coords)
    n_points = example.condition.coords.shape[0]
    raw_coords = np.asarray(example.condition.coords).reshape(1, 1, n_points, 3)
    return np.asarray(_normalize_coords(raw_coords, stats)).reshape(n_points, 3)


def _make_batch_group_with_seed(
    group_name: str,
    examples,
    stats: dict,
    builder: Heat3DGraphBuilder,
    *,
    graph_seed: int,
) -> dict:
    bridges = [_bridge_for(example) for example in examples]
    feature_names = bridges[0].condition_feature_names
    for bridge in bridges[1:]:
        if bridge.condition_feature_names != feature_names:
            raise ValueError(f"Feature-name mismatch in {group_name}")

    raw_u = jnp.concatenate([bridge.legacy_inputs.u for bridge in bridges], axis=0)
    raw_c = jnp.concatenate([bridge.legacy_inputs.c for bridge in bridges], axis=0)
    raw_coords = jnp.concatenate([bridge.legacy_inputs.x_inp for bridge in bridges], axis=0)
    target_delta = jnp.concatenate([bridge.target_delta_u for bridge in bridges], axis=0)
    t_ref = jnp.concatenate([bridge.t_ref for bridge in bridges], axis=0)

    c = normalize_condition(raw_c, stats)
    target = normalize_target_delta(target_delta, stats)
    coords = _normalize_coords(raw_coords, stats)
    inputs = Inputs(u=raw_u, c=c, x_inp=coords, x_out=coords, t=None, tau=None)
    metadata, shared = _build_batch_metadata_with_seed(
        builder=builder,
        coords_list=[_graph_coords_for_example(example, stats) for example in examples],
        graph_seed=graph_seed,
    )
    graphs = builder.build_graphs(metadata)
    return {
        "name": group_name,
        "sample_ids": tuple(example.sample_id for example in examples),
        "split": examples[0].meta.get("split"),
        "inputs": inputs,
        "target_normalized": target,
        "target_delta_raw": target_delta,
        "target_temperature": t_ref + target_delta,
        "t_ref": t_ref,
        "graphs": graphs,
        "metadata": metadata,
        "shared_metadata": shared,
        "feature_names": feature_names,
    }


def _make_groups_with_progress(
    examples,
    stats: dict,
    builder: Heat3DGraphBuilder,
    label: str,
    progress_enabled: bool,
    progress_detail: str,
    graph_seed: int,
    batch_size: int | None = None,
    drop_last: bool = False,
    profile_counts: dict[str, int] | None = None,
) -> list[dict]:
    start = time.perf_counter()
    sample_count = len(examples)
    batch_text = f" batch_size={batch_size}" if batch_size else " legacy_full_batch"
    _progress(progress_enabled, "startup", f"group build {label}: start samples={sample_count}{batch_text} ...")
    detail_mode = _progress_detail_mode(progress_detail)
    verbose_progress_enabled = progress_enabled and detail_mode == "full"

    grouped: dict[tuple[int, tuple[str, ...], tuple[tuple[int, ...], ...]], list] = {}
    checkpoints = _progress_checkpoints(sample_count)
    scan_start = time.perf_counter()
    for index, example in enumerate(examples, start=1):
        bridge = _bridge_for(example)
        signature = _metadata_shape_signature(
            builder.build_metadata(
                _graph_coords_for_example(example, stats),
                key=_metadata_key(graph_seed),
            )
        )
        _bump_profile_count(profile_counts, "graph_metadata_build_calls")
        _bump_profile_count(profile_counts, f"{label}_scan_metadata_build_calls")
        key = (
            example.condition.coords.shape[0],
            bridge.condition_feature_names,
            signature,
        )
        grouped.setdefault(key, []).append(example)
        if verbose_progress_enabled and index in checkpoints:
            _progress(
                True,
                "startup",
                f"group build {label}: {index}/{sample_count} samples scanned groups={len(grouped)}",
                scan_start,
            )

    _progress(
        progress_enabled,
        "startup",
        f"group build {label}: sample scan grouped={len(grouped)}",
        scan_start,
    )

    pending_batches: list[tuple[int, int, str, list]] = []
    grouped_count = len(grouped)
    for group_index, ((n_points, feature_names, _signature), group_examples) in enumerate(grouped.items(), start=1):
        group_name = f"group_{group_index}_N{n_points}_F{len(feature_names)}"
        batches = _chunk_examples(group_examples, batch_size=batch_size, drop_last=drop_last)
        for batch_index, batch_examples in enumerate(batches, start=1):
            if batch_size:
                batch_group_name = f"{group_name}_batch_{batch_index:04d}_B{len(batch_examples)}"
            else:
                batch_group_name = group_name
            pending_batches.append((group_index, grouped_count, batch_group_name, batch_examples))

    result = []
    bar = _ProgressBar(
        progress_enabled and detail_mode == "basic",
        f"[startup] group build {label}",
        len(pending_batches),
    )
    for current_index, (group_index, grouped_count, batch_group_name, batch_examples) in enumerate(pending_batches, start=1):
        batch_start = time.perf_counter()
        if verbose_progress_enabled:
            _progress(
                True,
                "startup",
                (
                    f"group build {label}: group {group_index}/{grouped_count} "
                    f"{batch_group_name} arrays+graph start samples={len(batch_examples)} ..."
                ),
            )
            _bump_profile_count(profile_counts, "graph_metadata_build_calls", len(batch_examples))
            _bump_profile_count(profile_counts, f"{label}_batch_metadata_build_calls", len(batch_examples))
            _bump_profile_count(profile_counts, "graph_build_graphs_calls")
            _bump_profile_count(profile_counts, f"{label}_build_graphs_calls")
        else:
            _bump_profile_count(profile_counts, "graph_metadata_build_calls", len(batch_examples))
            _bump_profile_count(profile_counts, f"{label}_batch_metadata_build_calls", len(batch_examples))
            _bump_profile_count(profile_counts, "graph_build_graphs_calls")
            _bump_profile_count(profile_counts, f"{label}_build_graphs_calls")
        result.append(
            _make_batch_group_with_seed(
                batch_group_name,
                batch_examples,
                stats,
                builder,
                graph_seed=graph_seed,
            )
        )
        if verbose_progress_enabled:
            _progress(
                True,
                "startup",
                f"group build {label}: {batch_group_name} arrays+graph built",
                batch_start,
            )
        else:
            bar.update(current_index)

    bar.close(current=len(result))
    _progress(progress_enabled, "startup", f"group build {label}: done groups={len(result)}", start)
    return result


def _sample_shuffle_failure_debug(
    batch_index: int,
    batch_group_name: str,
    batch_examples,
    stats: dict,
    builder: Heat3DGraphBuilder,
    graph_seed: int,
) -> dict[str, Any]:
    sample_debug = []
    signature_counts: dict[str, int] = {}
    for example in batch_examples:
        payload = {
            "sample_id": example.sample_id,
            "coords_shape": [int(dim) for dim in example.condition.coords.shape],
        }
        try:
            metadata = builder.build_metadata(
                _graph_coords_for_example(example, stats),
                key=_metadata_key(graph_seed),
            )
            signature = _metadata_shape_signature(metadata)
            payload["metadata_shape_signature"] = [list(shape) for shape in signature]
            key = json.dumps(payload["metadata_shape_signature"], sort_keys=True, separators=(",", ":"))
            signature_counts[key] = signature_counts.get(key, 0) + 1
        except Exception as exc:  # pragma: no cover - diagnostic path.
            payload["metadata_shape_error"] = f"{type(exc).__name__}: {exc}"
        sample_debug.append(payload)
    return {
        "batch_index": int(batch_index),
        "batch_group_name": batch_group_name,
        "sample_count": int(len(batch_examples)),
        "sample_ids": [example.sample_id for example in batch_examples],
        "metadata_shape_signature_counts": signature_counts,
        "samples": sample_debug,
    }


def _make_sample_shuffle_groups_with_progress(
    examples,
    stats: dict,
    builder: Heat3DGraphBuilder,
    label: str,
    progress_enabled: bool,
    progress_detail: str,
    graph_seed: int,
    batch_size: int,
    batch_build_seed: int,
    drop_last: bool = False,
    profile_counts: dict[str, int] | None = None,
) -> list[dict]:
    start = time.perf_counter()
    sample_count = len(examples)
    _progress(
        progress_enabled,
        "startup",
        (
            f"group build {label}: sample_shuffle start samples={sample_count} "
            f"batch_size={batch_size} batch_build_seed={batch_build_seed} ..."
        ),
    )
    rng = np.random.default_rng(int(batch_build_seed))
    indices = rng.permutation(sample_count)
    shuffled = [examples[int(index)] for index in indices]
    pending_batches = _chunk_examples(shuffled, batch_size=batch_size, drop_last=drop_last)
    result = []
    detail_mode = _progress_detail_mode(progress_detail)
    verbose_progress_enabled = progress_enabled and detail_mode == "full"
    bar = _ProgressBar(
        progress_enabled and detail_mode == "basic",
        f"[startup] group build {label} sample_shuffle",
        len(pending_batches),
    )
    for batch_index, batch_examples in enumerate(pending_batches, start=1):
        batch_group_name = f"{label}_sample_shuffle_batch_{batch_index:04d}_B{len(batch_examples)}"
        batch_start = time.perf_counter()
        if verbose_progress_enabled:
            _progress(
                True,
                "startup",
                f"group build {label}: {batch_group_name} arrays+graph start samples={len(batch_examples)} ...",
            )
        _bump_profile_count(profile_counts, "graph_metadata_build_calls", len(batch_examples))
        _bump_profile_count(profile_counts, f"{label}_sample_shuffle_batch_metadata_build_calls", len(batch_examples))
        _bump_profile_count(profile_counts, "graph_build_graphs_calls")
        _bump_profile_count(profile_counts, f"{label}_sample_shuffle_build_graphs_calls")
        try:
            result.append(
                _make_batch_group_with_seed(
                    batch_group_name,
                    batch_examples,
                    stats,
                    builder,
                    graph_seed=graph_seed,
                )
            )
        except Exception as exc:
            debug = _sample_shuffle_failure_debug(
                batch_index,
                batch_group_name,
                batch_examples,
                stats,
                builder,
                graph_seed,
            )
            raise RuntimeError(
                "sample_shuffle train batch build failed; "
                f"failure_stage=make_batch_group_with_seed; debug={json.dumps(_json_safe(debug), sort_keys=True)}"
            ) from exc
        if verbose_progress_enabled:
            _progress(True, "startup", f"group build {label}: {batch_group_name} arrays+graph built", batch_start)
        else:
            bar.update(batch_index)

    bar.close(current=len(result))
    _progress(
        progress_enabled,
        "startup",
        f"group build {label}: sample_shuffle done groups={len(result)}",
        start,
    )
    return result


def _chunk_examples(examples, *, batch_size: int | None, drop_last: bool) -> list:
    examples = list(examples)
    if batch_size is None:
        return [examples] if examples else []
    chunks = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        if len(chunk) < batch_size and drop_last:
            continue
        chunks.append(chunk)
    return chunks


def _require_nonempty_groups(groups: list[dict], label: str) -> None:
    if not groups:
        raise ValueError(f"{label} group build produced no groups; check batch_size/drop_last")


def _prediction_groups_for_split(
    prediction_split: str,
    *,
    all_groups: list[dict],
    train_groups: list[dict],
    valid_groups: list[dict],
    valid_stress_groups: list[dict],
    test_iid_groups: list[dict],
) -> list[dict]:
    groups_by_split = {
        "all": all_groups,
        "train": train_groups,
        "valid_iid": valid_groups,
        "valid_stress": valid_stress_groups,
        "test_iid": test_iid_groups,
    }
    groups = groups_by_split[prediction_split]
    _require_nonempty_groups(groups, f"prediction split {prediction_split}")
    return groups


def _epoch_train_groups(groups: list[dict], *, epoch: int, seed: int, shuffle: bool) -> list[dict]:
    if not shuffle or len(groups) <= 1:
        return groups
    rng = np.random.default_rng(seed + epoch)
    indices = rng.permutation(len(groups))
    return [groups[int(index)] for index in indices]


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.report_every < 1:
        raise ValueError("--report-every must be >= 1")
    if args.grad_norm_report_every < 0:
        raise ValueError("--grad-norm-report-every must be >= 0")
    _output_filename(args.best_predictions_name, "best-predictions-name")
    _output_filename(args.final_checkpoint_name, "final-checkpoint-name")
    _output_filename(args.best_checkpoint_name, "best-checkpoint-name")
    _output_filename(
        args.point_global_best_checkpoint_name,
        "point-global-best-checkpoint-name",
    )
    progress_enabled = _progress_enabled(args)
    progress_detail_enabled = _progress_detail_enabled(args)
    profile_enabled = _profile_timing_enabled(args)
    timings: dict[str, float] = {}
    profile_counts: dict[str, int] = {}
    script_start = time.perf_counter()
    _progress(
        progress_enabled,
        "startup",
        (
            f"script start subset={args.subset} epochs={args.epochs} lr={args.lr} "
            f"log_mode={args.log_mode} save_predictions={bool(args.save_predictions)}"
        ),
    )
    loss_config = _loss_config_from_args(args)
    lr_config = _lr_config_from_args(args)
    optimizer_config = _optimizer_config_from_args(args)
    seed_config = _seed_config_from_args(args)
    model_config = _model_config_from_args(args)
    batch_config = _batch_config_from_args(args)
    graph_config = _graph_config_from_args(args)
    sample_weight_config = _sample_weight_config_from_args(args)
    _validate_loss_config(loss_config)
    _validate_lr_config(lr_config)
    _validate_optimizer_config(optimizer_config)
    _validate_seed_config(seed_config)
    _validate_model_config(model_config)
    _validate_batch_config(batch_config)
    _validate_graph_config(graph_config)
    checkpoint_load_strict = args.checkpoint_load_strict == "true"
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        raise FileNotFoundError(f"--init-checkpoint not found: {args.init_checkpoint}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "model_config": _json_safe(model_config),
                    "batch_config": _json_safe(batch_config),
                    "graph_config": _json_safe(graph_config),
                    "loss_config": _json_safe(loss_config),
                    "optimizer_config": _json_safe(optimizer_config),
                    "training_runs": 0,
                    "output_writes": 0,
                },
                sort_keys=True,
            )
        )
        return 0

    output_start = time.perf_counter()
    output_dir = _ensure_ignored_output_dir(args.output_dir)
    profile_timing_json_path = (
        _ensure_ignored_output_file(args.profile_timing_json, "profile-timing-json")
        if args.profile_timing_json is not None
        else None
    )
    memory_audit_jsonl_path = (
        _ensure_ignored_output_file(args.memory_audit_jsonl, "memory-audit-jsonl")
        if args.memory_audit_jsonl is not None
        else None
    )
    memory_audit = (
        MemoryAudit(
            memory_audit_jsonl_path,
            every_batch=bool(args.memory_audit_every_batch),
            gc_enabled=bool(args.memory_audit_gc),
        )
        if memory_audit_jsonl_path is not None
        else None
    )
    if memory_audit is not None:
        memory_audit.record("startup_output_ready", detail={"output_dir": str(output_dir)})
    _progress(progress_enabled, "startup", f"output dir ready: {output_dir}", output_start)

    dataset_start = time.perf_counter()
    sample_root = _sample_root(args.subset)
    _progress(progress_enabled, "startup", f"loading dataset from {sample_root} ...")
    split_ids, split_source, primary_validation_split, stress_validation_split = _resolve_training_splits(
        sample_root,
        args.split_map,
    )

    all_ids = sorted(sample_id for ids in split_ids.values() for sample_id in ids)
    train_ids = split_ids["train"]
    valid_ids = split_ids[primary_validation_split]
    valid_stress_ids = split_ids.get(stress_validation_split, []) if stress_validation_split is not None else []
    test_iid_ids = split_ids.get("test_iid", [])
    split_counts = {split: len(ids) for split, ids in sorted(split_ids.items())}
    train_sample_weights, sample_weight_summary = _prepare_train_sample_weights(
        train_ids,
        sample_weight_config,
    )
    _progress(
        progress_enabled,
        "startup",
        (
            "sample weighting: "
            f"policy={sample_weight_summary['policy']} "
            f"weighted={sample_weight_summary['weighted_sample_count']}/"
            f"{sample_weight_summary['train_sample_count']} "
            f"mean={sample_weight_summary['mean']:.6g} "
            f"min={sample_weight_summary['min']:.6g} "
            f"max={sample_weight_summary['max']:.6g}"
        ),
    )
    for sample_id in all_ids:
        if not (sample_root / sample_id / "temperature.npy").is_file():
            raise FileNotFoundError(f"Missing temperature.npy for {sample_id}")

    dataset = Heat3DV1NativeSupervisedDataset(
        sample_root,
        k_encoding_mode="diag3",
        boundary_mask_fallback=args.boundary_mask_fallback,
    )
    index_by_id = dataset.sample_index_by_id()
    missing = [sample_id for sample_id in all_ids if sample_id not in index_by_id]
    if missing:
        raise FileNotFoundError(f"Dataset loader did not expose samples: {missing}")

    train_examples = [dataset[index_by_id[sample_id]] for sample_id in train_ids]
    valid_examples = [dataset[index_by_id[sample_id]] for sample_id in valid_ids]
    valid_stress_examples = [dataset[index_by_id[sample_id]] for sample_id in valid_stress_ids]
    _progress(
        progress_enabled,
        "startup",
        (
            f"dataset loaded: sample_count={len(dataset)} split_source={split_source} "
            f"primary_validation_split={primary_validation_split} split_counts={split_counts}"
        ),
        dataset_start,
    )
    _record_timing(timings, "dataset_load", dataset_start)

    builder = Heat3DGraphBuilder(**graph_config)
    norm_start = time.perf_counter()
    _progress(progress_enabled, "startup", "computing train-only target normalization ...")
    stats = _train_only_stats(train_examples)
    _progress(
        progress_enabled,
        "startup",
        (
            "target normalization done: "
            f"delta_mean={float(stats['target_delta_mean'].reshape(-1)[0]):.6e} "
            f"delta_std={float(stats['target_delta_std'].reshape(-1)[0]):.6e}"
        ),
        norm_start,
    )
    _record_timing(timings, "normalization", norm_start)
    model_config = _resolve_decoder_bypass_model_config(model_config, stats)
    _validate_model_config(model_config)
    group_start = time.perf_counter()
    _progress(progress_enabled, "startup", "building grouped JAX arrays and graphs ...")
    if batch_config["batch_plan"] == "sample_shuffle":
        train_groups = _make_sample_shuffle_groups_with_progress(
            train_examples,
            stats,
            builder,
            "train",
            progress_detail_enabled,
            args.progress_detail,
            seed_config["graph_seed"],
            batch_size=batch_config["batch_size"],
            batch_build_seed=batch_config["batch_build_seed"],
            drop_last=batch_config["drop_last"],
            profile_counts=profile_counts if profile_enabled else None,
        )
    else:
        train_groups = _make_groups_with_progress(
            train_examples,
            stats,
            builder,
            "train",
            progress_detail_enabled,
            args.progress_detail,
            seed_config["graph_seed"],
            batch_size=batch_config["batch_size"],
            drop_last=batch_config["drop_last"],
            profile_counts=profile_counts if profile_enabled else None,
        )
    valid_groups = _make_groups_with_progress(
        valid_examples,
        stats,
        builder,
        primary_validation_split,
        progress_detail_enabled,
        args.progress_detail,
        seed_config["graph_seed"],
        batch_size=batch_config["validation_batch_size"],
        drop_last=False,
        profile_counts=profile_counts if profile_enabled else None,
    )
    valid_stress_groups = (
        _make_groups_with_progress(
            valid_stress_examples,
            stats,
            builder,
            stress_validation_split,
            progress_detail_enabled,
            args.progress_detail,
            seed_config["graph_seed"],
            batch_size=batch_config["validation_batch_size"],
            drop_last=False,
            profile_counts=profile_counts if profile_enabled else None,
        )
        if stress_validation_split is not None and valid_stress_examples
        else []
    )
    build_all_groups = args.prediction_split == "all"
    build_test_iid_groups = args.prediction_split == "test_iid"
    all_groups: list[dict[str, Any]] = []
    test_iid_groups: list[dict[str, Any]] = []
    all_examples: list[Any] = []
    test_iid_examples: list[Any] = []
    if build_all_groups:
        all_examples = [dataset[index_by_id[sample_id]] for sample_id in all_ids]
        all_groups = _make_groups_with_progress(
            all_examples,
            stats,
            builder,
            "all",
            progress_detail_enabled,
            args.progress_detail,
            seed_config["graph_seed"],
            batch_size=batch_config["prediction_batch_size"],
            drop_last=False,
            profile_counts=profile_counts if profile_enabled else None,
        )
    if build_test_iid_groups:
        test_iid_examples = [dataset[index_by_id[sample_id]] for sample_id in test_iid_ids]
        test_iid_groups = _make_groups_with_progress(
            test_iid_examples,
            stats,
            builder,
            "test_iid",
            progress_detail_enabled,
            args.progress_detail,
            seed_config["graph_seed"],
            batch_size=batch_config["prediction_batch_size"],
            drop_last=False,
            profile_counts=profile_counts if profile_enabled else None,
        )
    global_context_lookup, global_context_payload = _prepare_global_context_lookup(
        model_config,
        train_examples=train_examples,
        required_examples=[
            *valid_examples,
            *valid_stress_examples,
            *all_examples,
            *test_iid_examples,
        ],
    )
    for groups in (train_groups, valid_groups, valid_stress_groups, all_groups, test_iid_groups):
        _attach_global_context_to_groups(
            groups,
            global_context_lookup,
            expected_feature_dim=int(model_config.get("global_context_feature_dim", 0)),
        )
    if model_config.get("native_output_mode") == "native_shape_scale":
        native_examples_by_id = {
            example.sample_id: example
            for example in (
                *train_examples,
                *valid_examples,
                *valid_stress_examples,
                *all_examples,
                *test_iid_examples,
            )
        }
        for groups in (train_groups, valid_groups, valid_stress_groups, all_groups, test_iid_groups):
            _attach_native_physics_to_groups(groups, native_examples_by_id)
    _attach_sample_weights_to_groups(train_groups, train_sample_weights)
    _require_nonempty_groups(train_groups, "train")
    _require_nonempty_groups(valid_groups, primary_validation_split)
    if build_all_groups:
        _require_nonempty_groups(all_groups, "all")
    if build_test_iid_groups:
        _require_nonempty_groups(test_iid_groups, "test_iid")
    alignment_groups = [*train_groups, *valid_groups, *valid_stress_groups]
    if build_all_groups:
        alignment_groups.extend(all_groups)
    if build_test_iid_groups:
        alignment_groups.extend(test_iid_groups)
    _check_decoder_bypass_input_alignment(model_config, alignment_groups)
    _record_timing(timings, "group_build", group_start)
    all_groups_status = str(len(all_groups)) if build_all_groups else "skipped"
    test_iid_groups_status = str(len(test_iid_groups)) if build_test_iid_groups else "skipped"
    _progress(
        progress_enabled,
        "startup",
        (
            "groups built: "
            f"train_groups={len(train_groups)} {primary_validation_split}_groups={len(valid_groups)} "
            f"valid_stress_groups={len(valid_stress_groups)} "
            f"test_iid_groups={test_iid_groups_status} all_groups={all_groups_status}"
        ),
        group_start,
    )
    if memory_audit is not None:
        built_group_signatures = {
            "train": _groups_memory_signature(train_groups),
            primary_validation_split: _groups_memory_signature(valid_groups),
            "valid_stress": _groups_memory_signature(valid_stress_groups),
        }
        if build_test_iid_groups:
            built_group_signatures["test_iid"] = _groups_memory_signature(test_iid_groups)
        if build_all_groups:
            built_group_signatures["all"] = _groups_memory_signature(all_groups)
        memory_audit.record(
            "groups_built",
            detail=built_group_signatures,
        )

    _print_startup_summary(
        args,
        sample_root=sample_root,
        split_counts=split_counts,
        split_source=split_source,
        primary_validation_split=primary_validation_split,
        stress_validation_split=stress_validation_split,
        output_dir=output_dir,
        loss_config=loss_config,
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        seed_config=seed_config,
        model_config=model_config,
        batch_config=batch_config,
    )

    result = _fit_once(
        train_groups,
        valid_groups,
        valid_stress_groups,
        stats,
        args.epochs,
        lr_config,
        seed_config["model_seed"],
        seed_config["batch_order_seed"],
        args.report_every,
        args.train_metrics_schedule,
        args.grad_norm_report_every,
        loss_config,
        optimizer_config,
        model_config,
        batch_config,
        args.selection_metric,
        args.log_mode,
        progress_enabled,
        init_mode=args.init_mode,
        init_checkpoint=args.init_checkpoint,
        checkpoint_load_strict=checkpoint_load_strict,
        partial_load_policy=args.partial_load_policy,
        timings=timings,
        profile_enabled=profile_enabled,
        memory_audit=memory_audit,
        primary_validation_split=primary_validation_split,
        stress_validation_split=stress_validation_split,
        track_point_global_best=bool(args.save_point_global_best_checkpoint),
    )
    prediction_groups = _prediction_groups_for_split(
        args.prediction_split,
        all_groups=all_groups,
        train_groups=train_groups,
        valid_groups=valid_groups,
        valid_stress_groups=valid_stress_groups,
        test_iid_groups=test_iid_groups,
    )
    predictions_path = output_dir / "predictions.npz"
    predictions: dict[str, np.ndarray] = {}
    final_prediction_export_skipped = not should_build_final_predictions(bool(args.save_predictions))
    final_prediction_export_skip_reason = "save_predictions_false" if final_prediction_export_skipped else None
    if should_build_final_predictions(bool(args.save_predictions)):
        prediction_start = time.perf_counter()
        if memory_audit is not None:
            memory_audit.record(
                "prediction_export_start",
                detail=_groups_memory_signature(prediction_groups),
            )
        _progress(progress_enabled, "export", "building recovered predictions ...")
        predictions = _predict_temperatures(result["model"], result["params"], prediction_groups, stats)
        _record_timing(timings, "prediction_export", prediction_start)
        if memory_audit is not None:
            memory_audit.record("prediction_export_end", detail={"prediction_count": len(predictions)})
        _progress(progress_enabled, "export", f"prediction arrays built: key_count={len(predictions)}", prediction_start)
    else:
        timings["prediction_export"] = 0.0
        _progress(
            progress_enabled,
            "export",
            "prediction export skipped: save_predictions=False",
        )

    best_predictions_path = output_dir / args.best_predictions_name if args.save_best_predictions else None
    best_predictions: dict[str, np.ndarray] = {}
    best_predictions_saved = False
    best_prediction_count = 0

    save_start = time.perf_counter()
    if args.save_predictions:
        _progress(progress_enabled, "export", f"saving predictions to {predictions_path} ...")
        np.savez_compressed(predictions_path, **predictions)
        _progress(progress_enabled, "export", f"predictions saved: key_count={len(predictions)} path={predictions_path}", save_start)
    else:
        _progress(progress_enabled, "export", f"prediction save skipped: key_count={len(predictions)}", save_start)
    _record_timing(timings, "prediction_save", save_start)
    if memory_audit is not None:
        memory_audit.record("prediction_save_end", detail={"prediction_count": len(predictions)})

    if args.save_best_predictions:
        if result.get("best_params") is None:
            raise RuntimeError("best params are unavailable; expected at least one training epoch")
        best_prediction_start = time.perf_counter()
        if memory_audit is not None:
            memory_audit.record(
                "best_prediction_export_start",
                detail=_groups_memory_signature(prediction_groups),
            )
        _progress(progress_enabled, "export", "building best-valid recovered predictions ...")
        best_params_device = _device_params(result["best_params"])
        try:
            best_predictions = _predict_temperatures(result["model"], best_params_device, prediction_groups, stats)
        finally:
            del best_params_device
        best_prediction_count = len(best_predictions)
        _record_timing(timings, "best_prediction_export", best_prediction_start)
        if memory_audit is not None:
            memory_audit.record(
                "best_prediction_export_end",
                detail={"prediction_count": best_prediction_count},
            )
        _progress(
            progress_enabled,
            "export",
            f"best-valid prediction arrays built: key_count={best_prediction_count}",
            best_prediction_start,
        )
        best_save_start = time.perf_counter()
        _progress(progress_enabled, "export", f"saving best predictions to {best_predictions_path} ...")
        np.savez_compressed(best_predictions_path, **best_predictions)
        best_predictions_saved = True
        _record_timing(timings, "best_prediction_save", best_save_start)
        if memory_audit is not None:
            memory_audit.record(
                "best_prediction_save_end",
                detail={"prediction_count": best_prediction_count},
            )
        _progress(
            progress_enabled,
            "export",
            f"best predictions saved: key_count={best_prediction_count} path={best_predictions_path}",
            best_save_start,
        )

    best_selection = _best_selection_payload(
        result,
        best_predictions_path=best_predictions_path,
        best_predictions_saved=best_predictions_saved,
    )
    checkpoint_run_metadata = _checkpoint_run_metadata(
        sample_root=sample_root,
        args=args,
        split_source=split_source,
        split_counts=split_counts,
        model_config=model_config,
        loss_config=loss_config,
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        seed_config=seed_config,
        batch_config=batch_config,
        graph_config=graph_config,
        global_context_payload=global_context_payload,
    )
    final_checkpoint_path = output_dir / args.final_checkpoint_name if args.save_final_checkpoint else None
    best_checkpoint_path = output_dir / args.best_checkpoint_name if args.save_best_checkpoint else None
    point_global_best_checkpoint_path = (
        output_dir / args.point_global_best_checkpoint_name
        if args.save_point_global_best_checkpoint
        else None
    )
    final_checkpoint_saved = False
    best_checkpoint_saved = False
    point_global_best_checkpoint_saved = False
    checkpoint_start = time.perf_counter()
    if args.save_final_checkpoint:
        _progress(progress_enabled, "export", f"saving final params checkpoint to {final_checkpoint_path} ...")
        _write_params_checkpoint(
            final_checkpoint_path,
            params=result["params"],
            model_config=model_config,
            stats=stats,
            kind="final",
            epoch=result.get("final_epoch"),
            record=_checkpoint_record_from_result(result, kind="final"),
            run_metadata=checkpoint_run_metadata,
        )
        final_checkpoint_saved = True
        _progress(progress_enabled, "export", f"final params checkpoint saved: {final_checkpoint_path}", checkpoint_start)
    if args.save_best_checkpoint:
        if result.get("best_params") is None:
            raise RuntimeError("best params are unavailable; expected at least one training epoch")
        best_checkpoint_start = time.perf_counter()
        _progress(progress_enabled, "export", f"saving best params checkpoint to {best_checkpoint_path} ...")
        _write_params_checkpoint(
            best_checkpoint_path,
            params=result["best_params"],
            model_config=model_config,
            stats=stats,
            kind="best",
            epoch=(result.get("best_record") or {}).get("epoch"),
            record=_checkpoint_record_from_result(result, kind="best"),
            run_metadata=checkpoint_run_metadata,
        )
        best_checkpoint_saved = True
        _progress(progress_enabled, "export", f"best params checkpoint saved: {best_checkpoint_path}", best_checkpoint_start)
    if args.save_point_global_best_checkpoint:
        if result.get("point_global_best_params") is None:
            raise RuntimeError("point-global best params are unavailable")
        point_global_start = time.perf_counter()
        _progress(
            progress_enabled,
            "export",
            f"saving point-global best params checkpoint to {point_global_best_checkpoint_path} ...",
        )
        _write_params_checkpoint(
            point_global_best_checkpoint_path,
            params=result["point_global_best_params"],
            model_config=model_config,
            stats=stats,
            kind="point_global_best",
            epoch=(result.get("point_global_best_record") or {}).get("epoch"),
            record=_checkpoint_record_from_result(result, kind="point_global_best"),
            run_metadata=checkpoint_run_metadata,
        )
        point_global_best_checkpoint_saved = True
        _progress(
            progress_enabled,
            "export",
            f"point-global best params checkpoint saved: {point_global_best_checkpoint_path}",
            point_global_start,
        )
    _record_timing(timings, "checkpoint_save", checkpoint_start)
    native_runtime_audit = _native_runtime_architecture_audit(
        result["model"], result["params"], train_groups[0]
    )
    reload_entries = []
    if final_checkpoint_saved and args.save_predictions:
        reload_entries.append(
            ("final", final_checkpoint_path, predictions_path, predictions, result["params"])
        )
    if best_checkpoint_saved and best_predictions_saved:
        reload_entries.append(
            ("best", best_checkpoint_path, best_predictions_path, best_predictions, result["best_params"])
        )
    checkpoint_prediction_reload_audit = _checkpoint_prediction_reload_audit(
        model=result["model"],
        groups=prediction_groups,
        stats=stats,
        entries=reload_entries,
    )
    memory_audit_summary = memory_audit.summary() if memory_audit is not None else None

    run_config = {
        "diagnostic_scope": "controlled training export smoke; not formal model performance",
        "subset": str(sample_root),
        "split_map_path": str(args.split_map) if args.split_map is not None else None,
        "split_source": split_source,
        "primary_validation_split": primary_validation_split,
        "stress_validation_split": stress_validation_split,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_schedule": lr_config["lr_schedule"],
        "warmup_epochs": lr_config["warmup_epochs"],
        "min_lr": lr_config["min_lr"],
        "second_stage_epoch": lr_config["second_stage_epoch"],
        "second_stage_lr": lr_config["second_stage_lr"],
        "lr_init": lr_config["lr_init"],
        "lr_peak": lr_config["lr_peak"],
        "lr_base": lr_config["lr_base"],
        "lr_lowr": lr_config["lr_lowr"],
        "pct_start": lr_config["pct_start"],
        "pct_final": lr_config["pct_final"],
        "optimizer": optimizer_config["optimizer"],
        "gradient_clip_norm": optimizer_config["gradient_clip_norm"],
        "weight_decay": optimizer_config["weight_decay"],
        "init_mode": args.init_mode,
        "model_config": model_config,
        **_batch_config_payload(batch_config),
        "seed": seed_config["seed"],
        "legacy_seed": seed_config["legacy_seed"],
        "model_seed": seed_config["model_seed"],
        "batch_order_seed": seed_config["batch_order_seed"],
        "graph_seed": seed_config["graph_seed"],
        "boundary_mask_fallback": bool(args.boundary_mask_fallback),
        "graph_config": graph_config,
        "global_context": global_context_payload,
        "native_runtime_architecture_audit": native_runtime_audit,
        "checkpoint_prediction_reload_audit": checkpoint_prediction_reload_audit,
        "memory_audit_summary": memory_audit_summary,
        "route": "relative BC features + zero_delta_u_bridge + normalized DeltaT target",
        **_decoder_bypass_payload(model_config),
        "output_dir": str(output_dir),
        "save_predictions": bool(args.save_predictions),
        "prediction_split": args.prediction_split,
        "predictions_path": str(predictions_path) if args.save_predictions else None,
        "save_best_predictions": bool(args.save_best_predictions),
        "best_predictions_name": args.best_predictions_name,
        "save_final_checkpoint": bool(args.save_final_checkpoint),
        "final_checkpoint_name": args.final_checkpoint_name,
        "final_checkpoint_saved": bool(final_checkpoint_saved),
        "final_checkpoint_path": str(final_checkpoint_path) if final_checkpoint_path is not None else None,
        "save_best_checkpoint": bool(args.save_best_checkpoint),
        "best_checkpoint_name": args.best_checkpoint_name,
        "best_checkpoint_saved": bool(best_checkpoint_saved),
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        "save_point_global_best_checkpoint": bool(args.save_point_global_best_checkpoint),
        "point_global_best_checkpoint_name": args.point_global_best_checkpoint_name,
        "point_global_best_checkpoint_saved": bool(point_global_best_checkpoint_saved),
        "point_global_best_checkpoint_path": (
            str(point_global_best_checkpoint_path)
            if point_global_best_checkpoint_path is not None
            else None
        ),
        "point_global_best_epoch": (result.get("point_global_best_record") or {}).get("epoch"),
        "point_global_best_relative_rmse_pct": result.get("point_global_best_score"),
        "final_probe_eval_after_training": bool(args.final_probe_eval_after_training),
        "final_probe_checkpoint_kind": args.final_probe_checkpoint_kind,
        "final_probe_output_dir": str(args.final_probe_output_dir) if args.final_probe_output_dir is not None else str(output_dir / "final_probe_eval"),
        "final_probe_subset": str(args.final_probe_subset),
        "final_probe_provenance": str(args.final_probe_provenance),
        "final_probe_batch_size": int(args.final_probe_batch_size),
        "post_training_diagnostics": bool(args.post_training_diagnostics),
        "post_training_diagnostics_output_dir": (
            str(args.post_training_diagnostics_output_dir)
            if args.post_training_diagnostics_output_dir is not None
            else str(output_dir / "post_training_diagnostics")
        ),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "checkpoint_load_mode": result["checkpoint_load_info"].get("checkpoint_load_mode"),
        "checkpoint_load_strict": bool(result["checkpoint_load_info"].get("checkpoint_load_strict")),
        "partial_load_policy": result["checkpoint_load_info"].get("partial_load_policy"),
        "checkpoint_loaded": bool(result["checkpoint_load_info"].get("loaded")),
        "checkpoint_loaded_key_count": int(result["checkpoint_load_info"].get("loaded_key_count", 0)),
        "checkpoint_skipped_key_count": int(result["checkpoint_load_info"].get("skipped_key_count", 0)),
        "checkpoint_missing_key_count": int(result["checkpoint_load_info"].get("missing_key_count", 0)),
        "checkpoint_unused_key_count": int(result["checkpoint_load_info"].get("unused_key_count", 0)),
        "checkpoint_shape_mismatch_count": int(result["checkpoint_load_info"].get("shape_mismatch_count", 0)),
        "checkpoint_load_info": result["checkpoint_load_info"],
        "log_mode": args.log_mode,
        "progress_log": bool(args.progress_log),
        "progress_detail": args.progress_detail,
        "profile_timing": bool(args.profile_timing),
        "profile_timing_json": str(profile_timing_json_path) if profile_timing_json_path is not None else None,
        "memory_audit_jsonl": str(memory_audit_jsonl_path) if memory_audit_jsonl_path is not None else None,
        "memory_audit_every_batch": bool(args.memory_audit_every_batch),
        "memory_audit_gc": bool(args.memory_audit_gc),
        "train_metrics_schedule": result["train_metrics_schedule"],
        "train_metrics_epochs": [int(epoch) for epoch in result["train_metrics_epochs"]],
        "initial_valid_loss": result["initial_valid_loss"],
        "initial_valid_iid_loss": result["initial_valid_iid_loss"],
        "initial_valid_stress_loss": result["initial_valid_stress_loss"],
        "initial_valid_raw_deltaT_mse": result["initial_valid_raw_deltaT_mse"],
        "initial_valid_iid_raw_deltaT_mse": result["initial_valid_iid_raw_deltaT_mse"],
        "initial_valid_stress_raw_deltaT_mse": result["initial_valid_stress_raw_deltaT_mse"],
        "updates_per_epoch": result["updates_per_epoch"],
        "total_update_count": result["total_update_count"],
        "train_group_count": result["train_group_count"],
        "train_group_sample_counts": result["train_group_sample_counts"],
        "train_group_names": result["train_group_names"],
        "valid_iid_group_count": result["valid_iid_group_count"],
        "valid_stress_group_count": result["valid_stress_group_count"],
        "test_iid_group_count": len(test_iid_groups) if build_test_iid_groups else 0,
        "all_groups_count": len(all_groups),
        "all_groups_status": "built" if build_all_groups else "skipped",
        "prediction_group_count": len(prediction_groups),
        "train_group_sample_id_hash": result["train_group_sample_id_hash"],
        "valid_iid_sample_id_hash": result["valid_iid_sample_id_hash"],
        "valid_stress_sample_id_hash": result["valid_stress_sample_id_hash"],
        "deterministic_audit_enabled": result["deterministic_audit_enabled"],
        "epoch_train_batch_order_hashes": result["epoch_train_batch_order_hashes"],
        "code_version_or_git_commit": result["code_version_or_git_commit"],
        "final_best_ratio": result["final_best_ratio"],
        "epoch_lrs": result["epoch_lrs"],
        "grad_norm_report_every": result["grad_norm_report_every"],
        "grad_norm_reported_batch_count": result["grad_norm_reported_batch_count"],
        "grad_norm_skipped_batch_count": result["grad_norm_skipped_batch_count"],
        "grad_norm_reporting_mode": result["grad_norm_reporting_mode"],
        "final_metrics_reused": result["final_metrics_reused"],
        "final_metrics_reuse_source": result["final_metrics_reuse_source"],
        "final_metrics_time_s": result["final_metrics_time_s"],
        **_validation_metric_scalars(result["valid_metrics"]),
        "best_params_storage": result.get("best_params_storage"),
        "final_prediction_export_skipped": bool(final_prediction_export_skipped),
        "final_prediction_export_skip_reason": final_prediction_export_skip_reason,
        **best_selection,
        "checkpoint_saved": bool(
            final_checkpoint_saved
            or best_checkpoint_saved
            or point_global_best_checkpoint_saved
        ),
        "loss_mode": loss_config["loss_mode"],
        "background_quantile": loss_config["background_quantile"],
        "hotspot_quantile": loss_config["hotspot_quantile"],
        "strong_q_quantile": loss_config["strong_q_quantile"],
        "background_weight": loss_config["background_weight"],
        "hotspot_weight": loss_config["hotspot_weight"],
        "strong_q_weight": loss_config["strong_q_weight"],
        "background_l1_weight": loss_config["background_l1_weight"],
        "background_bias_weight": loss_config["background_bias_weight"],
        "background_over_weight": loss_config["background_over_weight"],
        "background_relative_weight": loss_config["background_relative_weight"],
        "relative_floor": loss_config["relative_floor"],
        "relative_floor_mode": loss_config["relative_floor_mode"],
        "pseudo_negative_quantile": loss_config["pseudo_negative_quantile"],
        "pseudo_negative_delta_threshold": loss_config["pseudo_negative_delta_threshold"],
        "pseudo_negative_weight": loss_config["pseudo_negative_weight"],
        "pseudo_negative_over_margin": loss_config["pseudo_negative_over_margin"],
        "pseudo_negative_min_count": loss_config["pseudo_negative_min_count"],
        "pseudo_negative_loss_type": loss_config["pseudo_negative_loss_type"],
        "pseudo_negative_relative_floor": loss_config["pseudo_negative_relative_floor"],
        **_loss_weight_schedule_payload(loss_config),
        "loss": loss_config,
        "lr_config": lr_config,
        "optimizer_config": optimizer_config,
        "model_config": model_config,
        "batch_config": batch_config,
        "sample_weight_config": sample_weight_config,
        "sample_weight_summary": sample_weight_summary,
        "graph_config": graph_config,
        "split_counts": split_counts,
        "boundary_mask_fallback": bool(args.boundary_mask_fallback),
        "timing_diagnostics": dict(timings),
        "timing_profile_counts": dict(profile_counts),
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "valid_iid_ids": valid_ids if primary_validation_split == "valid_iid" else [],
        "valid_stress_ids": valid_stress_ids,
        "test_iid_ids": test_iid_ids,
        "ignored_candidate_ids": sorted(
            sample_id
            for split, ids in split_ids.items()
            if split not in {"train", primary_validation_split, stress_validation_split}
            for sample_id in ids
        ),
    }

    loss_summary = {
        "status_ok": bool(result["status_ok"]),
        "grad_finite": bool(result["grad_finite"]),
        "split_map_path": str(args.split_map) if args.split_map is not None else None,
        "split_source": split_source,
        "primary_validation_split": primary_validation_split,
        "stress_validation_split": stress_validation_split,
        "prediction_split": args.prediction_split,
        "split_counts": split_counts,
        "global_context": global_context_payload,
        "native_runtime_architecture_audit": native_runtime_audit,
        "checkpoint_prediction_reload_audit": checkpoint_prediction_reload_audit,
        "memory_audit_summary": memory_audit_summary,
        **_decoder_bypass_payload(model_config),
        "memory_audit_jsonl": str(memory_audit_jsonl_path) if memory_audit_jsonl_path is not None else None,
        "memory_audit_every_batch": bool(args.memory_audit_every_batch),
        "memory_audit_gc": bool(args.memory_audit_gc),
        "train_losses": [float(value) for value in result["train_losses"]],
        "train_loss_epochs": [int(value) for value in result["train_loss_epochs"]],
        "valid_losses": [float(value) for value in result["valid_losses"]],
        "valid_iid_losses": [float(value) for value in result["valid_iid_losses"]],
        "valid_stress_losses": [float(value) for value in result["valid_stress_losses"]],
        "initial_train_loss": result["initial_train_loss"],
        "initial_valid_loss": result["initial_valid_loss"],
        "initial_valid_iid_loss": result["initial_valid_iid_loss"],
        "initial_valid_stress_loss": result["initial_valid_stress_loss"],
        "initial_valid_base_mse": result["initial_valid_base_mse"],
        "initial_valid_raw_deltaT_mse": result["initial_valid_raw_deltaT_mse"],
        "initial_valid_iid_raw_deltaT_mse": result["initial_valid_iid_raw_deltaT_mse"],
        "initial_valid_stress_raw_deltaT_mse": result["initial_valid_stress_raw_deltaT_mse"],
        "updates_per_epoch": result["updates_per_epoch"],
        "total_update_count": result["total_update_count"],
        "train_group_count": result["train_group_count"],
        "train_group_sample_counts": result["train_group_sample_counts"],
        "train_group_names": result["train_group_names"],
        "valid_iid_group_count": result["valid_iid_group_count"],
        "valid_stress_group_count": result["valid_stress_group_count"],
        "test_iid_group_count": len(test_iid_groups) if build_test_iid_groups else 0,
        "all_groups_count": len(all_groups),
        "all_groups_status": "built" if build_all_groups else "skipped",
        "prediction_group_count": len(prediction_groups),
        "train_group_sample_id_hash": result["train_group_sample_id_hash"],
        "valid_iid_sample_id_hash": result["valid_iid_sample_id_hash"],
        "valid_stress_sample_id_hash": result["valid_stress_sample_id_hash"],
        "deterministic_audit_enabled": result["deterministic_audit_enabled"],
        "seed": seed_config["seed"],
        "legacy_seed": seed_config["legacy_seed"],
        "model_seed": result["model_seed"],
        "batch_order_seed": result["batch_order_seed"],
        "graph_seed": seed_config["graph_seed"],
        "shuffle_train_batches": bool(batch_config["shuffle_train_batches"]),
        "epoch_train_batch_order_hashes": result["epoch_train_batch_order_hashes"],
        "code_version_or_git_commit": result["code_version_or_git_commit"],
        "final_best_ratio": result["final_best_ratio"],
        "epoch_lrs": result["epoch_lrs"],
        "epoch_mean_train_batch_loss": [
            record.get("epoch_mean_train_batch_loss") for record in result["epoch_history"]
        ],
        "epoch_min_train_batch_loss": [
            record.get("epoch_min_train_batch_loss") for record in result["epoch_history"]
        ],
        "epoch_max_train_batch_loss": [
            record.get("epoch_max_train_batch_loss") for record in result["epoch_history"]
        ],
        "epoch_mean_grad_norm": [
            record.get("epoch_mean_grad_norm") for record in result["epoch_history"]
        ],
        "epoch_max_grad_norm": [
            record.get("epoch_max_grad_norm") for record in result["epoch_history"]
        ],
        "native_gradient_group_norms": {
            group_name: {
                "epoch_mean": [
                    record.get(f"epoch_mean_{group_name}_grad_norm")
                    for record in result["epoch_history"]
                ],
                "epoch_max": [
                    record.get(f"epoch_max_{group_name}_grad_norm")
                    for record in result["epoch_history"]
                ],
            }
            for group_name in ("backbone", "shape_decoder", "scale_head")
        },
        "epoch_mean_update_norm": [
            record.get("epoch_mean_update_norm") for record in result["epoch_history"]
        ],
        "epoch_max_update_norm": [
            record.get("epoch_max_update_norm") for record in result["epoch_history"]
        ],
        "epoch_mean_param_norm": [
            record.get("epoch_mean_param_norm") for record in result["epoch_history"]
        ],
        "epoch_update_to_param_norm_ratio": [
            record.get("epoch_update_to_param_norm_ratio") for record in result["epoch_history"]
        ],
        "epoch_max_update_to_param_norm_ratio": [
            record.get("epoch_max_update_to_param_norm_ratio") for record in result["epoch_history"]
        ],
        "grad_norms": [float(value) for value in result["grad_norms"]],
        "lr_history": [float(value) for value in result["lr_history"]],
        "lr_history_summary": _sequence_summary(result["lr_history"]),
        "loss_weight_history": result["loss_weight_history"],
        "loss_weight_history_summary": {
            "current_background_l1_weight": _history_field_summary(
                result["loss_weight_history"], "current_background_l1_weight"
            ),
            "current_background_bias_weight": _history_field_summary(
                result["loss_weight_history"], "current_background_bias_weight"
            ),
            "current_background_over_weight": _history_field_summary(
                result["loss_weight_history"], "current_background_over_weight"
            ),
            "current_background_relative_weight": _history_field_summary(
                result["loss_weight_history"], "current_background_relative_weight"
            ),
            "current_hotspot_weight": _history_field_summary(result["loss_weight_history"], "current_hotspot_weight"),
            "current_strong_q_weight": _history_field_summary(
                result["loss_weight_history"], "current_strong_q_weight"
            ),
        },
        "train_metrics": _metrics_payload(result["train_metrics"]),
        "valid_metrics": _metrics_payload(result["valid_metrics"]),
        "valid_iid_metrics": _metrics_payload(result["valid_metrics"]) if primary_validation_split == "valid_iid" else {},
        "valid_stress_metrics": (
            _metrics_payload(result["valid_stress_metrics"])
            if result["valid_stress_metrics"] is not None
            else {}
        ),
        "log_mode": args.log_mode,
        "progress_log": bool(args.progress_log),
        "progress_detail": args.progress_detail,
        "profile_timing": bool(args.profile_timing),
        "profile_timing_json": str(profile_timing_json_path) if profile_timing_json_path is not None else None,
        "train_metrics_schedule": result["train_metrics_schedule"],
        "train_metrics_epochs": [int(epoch) for epoch in result["train_metrics_epochs"]],
        "grad_norm_report_every": result["grad_norm_report_every"],
        "grad_norm_reported_batch_count": result["grad_norm_reported_batch_count"],
        "grad_norm_skipped_batch_count": result["grad_norm_skipped_batch_count"],
        "grad_norm_reporting_mode": result["grad_norm_reporting_mode"],
        "final_metrics_reused": result["final_metrics_reused"],
        "final_metrics_reuse_source": result["final_metrics_reuse_source"],
        "final_metrics_time_s": result["final_metrics_time_s"],
        **_validation_metric_scalars(result["valid_metrics"]),
        "best_params_storage": result.get("best_params_storage"),
        "final_prediction_export_skipped": bool(final_prediction_export_skipped),
        "final_prediction_export_skip_reason": final_prediction_export_skip_reason,
        **best_selection,
        "checkpoint_saved": bool(
            final_checkpoint_saved
            or best_checkpoint_saved
            or point_global_best_checkpoint_saved
        ),
        "save_final_checkpoint": bool(args.save_final_checkpoint),
        "final_checkpoint_name": args.final_checkpoint_name,
        "final_checkpoint_saved": bool(final_checkpoint_saved),
        "final_checkpoint_path": str(final_checkpoint_path) if final_checkpoint_path is not None else None,
        "save_best_checkpoint": bool(args.save_best_checkpoint),
        "best_checkpoint_name": args.best_checkpoint_name,
        "best_checkpoint_saved": bool(best_checkpoint_saved),
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        "save_point_global_best_checkpoint": bool(args.save_point_global_best_checkpoint),
        "point_global_best_checkpoint_name": args.point_global_best_checkpoint_name,
        "point_global_best_checkpoint_saved": bool(point_global_best_checkpoint_saved),
        "point_global_best_checkpoint_path": (
            str(point_global_best_checkpoint_path)
            if point_global_best_checkpoint_path is not None
            else None
        ),
        "point_global_best_epoch": (result.get("point_global_best_record") or {}).get("epoch"),
        "point_global_best_relative_rmse_pct": result.get("point_global_best_score"),
        "final_probe_eval_after_training": bool(args.final_probe_eval_after_training),
        "final_probe_checkpoint_kind": args.final_probe_checkpoint_kind,
        "final_probe_output_dir": str(args.final_probe_output_dir) if args.final_probe_output_dir is not None else str(output_dir / "final_probe_eval"),
        "final_probe_subset": str(args.final_probe_subset),
        "final_probe_provenance": str(args.final_probe_provenance),
        "final_probe_batch_size": int(args.final_probe_batch_size),
        "post_training_diagnostics": bool(args.post_training_diagnostics),
        "post_training_diagnostics_output_dir": (
            str(args.post_training_diagnostics_output_dir)
            if args.post_training_diagnostics_output_dir is not None
            else str(output_dir / "post_training_diagnostics")
        ),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "checkpoint_load_mode": result["checkpoint_load_info"].get("checkpoint_load_mode"),
        "checkpoint_load_strict": bool(result["checkpoint_load_info"].get("checkpoint_load_strict")),
        "partial_load_policy": result["checkpoint_load_info"].get("partial_load_policy"),
        "checkpoint_loaded": bool(result["checkpoint_load_info"].get("loaded")),
        "checkpoint_loaded_key_count": int(result["checkpoint_load_info"].get("loaded_key_count", 0)),
        "checkpoint_skipped_key_count": int(result["checkpoint_load_info"].get("skipped_key_count", 0)),
        "checkpoint_missing_key_count": int(result["checkpoint_load_info"].get("missing_key_count", 0)),
        "checkpoint_unused_key_count": int(result["checkpoint_load_info"].get("unused_key_count", 0)),
        "checkpoint_shape_mismatch_count": int(result["checkpoint_load_info"].get("shape_mismatch_count", 0)),
        "checkpoint_load_info": result["checkpoint_load_info"],
        "loss_mode": loss_config["loss_mode"],
        "background_quantile": loss_config["background_quantile"],
        "hotspot_quantile": loss_config["hotspot_quantile"],
        "strong_q_quantile": loss_config["strong_q_quantile"],
        "background_weight": loss_config["background_weight"],
        "hotspot_weight": loss_config["hotspot_weight"],
        "strong_q_weight": loss_config["strong_q_weight"],
        "background_l1_weight": loss_config["background_l1_weight"],
        "background_bias_weight": loss_config["background_bias_weight"],
        "background_over_weight": loss_config["background_over_weight"],
        "background_relative_weight": loss_config["background_relative_weight"],
        "relative_floor": loss_config["relative_floor"],
        "relative_floor_mode": loss_config["relative_floor_mode"],
        "pseudo_negative_quantile": loss_config["pseudo_negative_quantile"],
        "pseudo_negative_delta_threshold": loss_config["pseudo_negative_delta_threshold"],
        "pseudo_negative_weight": loss_config["pseudo_negative_weight"],
        "pseudo_negative_over_margin": loss_config["pseudo_negative_over_margin"],
        "pseudo_negative_min_count": loss_config["pseudo_negative_min_count"],
        "pseudo_negative_loss_type": loss_config["pseudo_negative_loss_type"],
        "pseudo_negative_relative_floor": loss_config["pseudo_negative_relative_floor"],
        **_loss_weight_schedule_payload(loss_config),
        "lr": lr_config["lr"],
        "lr_schedule": lr_config["lr_schedule"],
        "warmup_epochs": lr_config["warmup_epochs"],
        "min_lr": lr_config["min_lr"],
        "second_stage_epoch": lr_config["second_stage_epoch"],
        "second_stage_lr": lr_config["second_stage_lr"],
        "lr_init": lr_config["lr_init"],
        "lr_peak": lr_config["lr_peak"],
        "lr_base": lr_config["lr_base"],
        "lr_lowr": lr_config["lr_lowr"],
        "pct_start": lr_config["pct_start"],
        "pct_final": lr_config["pct_final"],
        "optimizer": optimizer_config["optimizer"],
        "gradient_clip_norm": optimizer_config["gradient_clip_norm"],
        "weight_decay": optimizer_config["weight_decay"],
        "init_mode": args.init_mode,
        "model_config": model_config,
        **_batch_config_payload(batch_config),
        "epoch_batch_counts": [int(value) for value in result["epoch_batch_counts"]],
        "train_loss_selected": _selected_steps(result["train_losses"], args.report_every),
        "train_loss_selected_epochs": [
            (int(result["train_loss_epochs"][index]), float(result["train_losses"][index]))
            for index, _ in _selected_steps(result["train_losses"], args.report_every)
        ],
        "valid_loss_selected": _selected_steps(result["valid_losses"], args.report_every),
        "grad_norm_selected": _selected_steps_or_empty(result["grad_norms"], args.report_every),
        "lr_config": lr_config,
        "optimizer_config": optimizer_config,
        "model_config": model_config,
        "batch_config": batch_config,
        "sample_weight_config": sample_weight_config,
        "sample_weight_summary": sample_weight_summary,
        "graph_config": graph_config,
        "epoch_history": result["epoch_history"],
        "train_batch_records": result["train_batch_records"] if profile_enabled else [],
        "validation_batch_records": result["validation_batch_records"] if profile_enabled else [],
        "loss": loss_config,
        "final_train_loss_components": result["final_train_loss_components"],
        "final_valid_loss_components": result["final_valid_loss_components"],
        "final_valid_iid_loss_components": result["final_valid_iid_loss_components"],
        "final_valid_stress_loss_components": result["final_valid_stress_loss_components"],
        "train_only_normalization": _stats_payload(stats),
    }
    summary_write_start = time.perf_counter()
    _progress(progress_enabled, "export", "writing run_config.json and loss_summary.json ...")
    loss_summary["timing_diagnostics"] = dict(timings)
    run_config["timing_diagnostics"] = dict(timings)
    _write_json(output_dir / "run_config.json", run_config)
    _write_json(output_dir / "loss_summary.json", loss_summary)
    _record_timing(timings, "summary_write", summary_write_start)
    _progress(progress_enabled, "export", "run summary files written", summary_write_start)

    post_training_diagnostics_result = _run_post_training_prediction_diagnostics(
        args,
        sample_root=sample_root,
        output_dir=output_dir,
        predictions_path=predictions_path,
        predictions_saved=bool(args.save_predictions),
        best_predictions_path=best_predictions_path,
        best_predictions_saved=best_predictions_saved,
        timings=timings,
        progress_enabled=progress_enabled,
    )
    run_config["post_training_diagnostics_result"] = post_training_diagnostics_result
    loss_summary["post_training_diagnostics_result"] = post_training_diagnostics_result
    loss_summary["timing_diagnostics"] = dict(timings)
    run_config["timing_diagnostics"] = dict(timings)
    diagnostics_summary_write_start = time.perf_counter()
    _write_json(output_dir / "run_config.json", run_config)
    _write_json(output_dir / "loss_summary.json", loss_summary)
    _record_timing(timings, "post_training_diagnostics_summary_write", diagnostics_summary_write_start)

    final_probe_eval_result = _run_post_training_final_probe_eval(
        args,
        output_dir=output_dir,
        final_checkpoint_path=final_checkpoint_path,
        final_checkpoint_saved=final_checkpoint_saved,
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_saved=best_checkpoint_saved,
        timings=timings,
        progress_enabled=progress_enabled,
    )
    run_config["final_probe_eval_result"] = final_probe_eval_result
    loss_summary["final_probe_eval_result"] = final_probe_eval_result
    loss_summary["timing_diagnostics"] = dict(timings)
    run_config["timing_diagnostics"] = dict(timings)
    final_probe_summary_write_start = time.perf_counter()
    _write_json(output_dir / "run_config.json", run_config)
    _write_json(output_dir / "loss_summary.json", loss_summary)
    _record_timing(timings, "final_probe_summary_write", final_probe_summary_write_start)

    profile_payload = _profile_timing_payload(
        timings=timings,
        profile_counts=profile_counts,
        epoch_records=result["epoch_history"],
        train_batch_records=result["train_batch_records"] if profile_enabled else [],
        validation_batch_records=result["validation_batch_records"] if profile_enabled else [],
        train_group_count=len(train_groups),
        valid_group_count=len(valid_groups),
        all_group_count=len(all_groups),
        prediction_group_count=len(prediction_groups),
        all_groups_built=build_all_groups,
        train_batch_counts=[int(value) for value in result["epoch_batch_counts"]],
        subset=sample_root,
        output_dir=output_dir,
        train_metrics_schedule=result["train_metrics_schedule"],
        train_metrics_epoch_values=[int(epoch) for epoch in result["train_metrics_epochs"]],
        grad_norm_report_every=result["grad_norm_report_every"],
        grad_norm_reported_batch_count=result["grad_norm_reported_batch_count"],
        grad_norm_skipped_batch_count=result["grad_norm_skipped_batch_count"],
        final_metrics_reused=result["final_metrics_reused"],
        final_metrics_reuse_source=result["final_metrics_reuse_source"],
        final_prediction_export_skipped=final_prediction_export_skipped,
        final_prediction_export_skip_reason=final_prediction_export_skip_reason,
        total_run_time_so_far=time.perf_counter() - script_start,
    )
    if profile_enabled:
        _print_profile_timing(profile_payload)
    if profile_timing_json_path is not None:
        profile_json_start = time.perf_counter()
        _write_json(profile_timing_json_path, profile_payload)
        _record_timing(timings, "profile_timing_json_write", profile_json_start)
        _progress(
            progress_enabled,
            "export",
            f"profile timing json written: {profile_timing_json_path}",
            profile_json_start,
        )

    _print_final_summary(
        args,
        result=result,
        loss_config=loss_config,
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        model_config=model_config,
        predictions_path=predictions_path,
        predictions_saved=bool(args.save_predictions),
        prediction_count=len(predictions),
        best_predictions_path=best_predictions_path,
        best_predictions_saved=best_predictions_saved,
        best_prediction_count=best_prediction_count,
        final_checkpoint_path=final_checkpoint_path,
        final_checkpoint_saved=final_checkpoint_saved,
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_saved=best_checkpoint_saved,
        final_prediction_export_skipped=final_prediction_export_skipped,
        final_prediction_export_skip_reason=final_prediction_export_skip_reason,
        timings=timings,
    )
    _progress(progress_enabled, "done", "script complete", script_start)
    return 0 if result["status_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
