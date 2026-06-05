"""Controlled Heat3D v1 medium training export smoke.

This runner reuses the existing v1 train/valid smoke path and writes recovered
temperature predictions to an ignored output directory for downstream
diagnostic comparison. It is not a formal training experiment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

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
    _bridge_for,
    _global_norm,
    _make_batch_group,
    _metadata_shape_signature,
    _metrics,
    _sample_root,
    _selected_steps,
    _subset_split_ids,
    _train_only_stats,
)
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


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
TRAIN_METRICS_SCHEDULE_CHOICES = ("every_epoch", "half_and_final", "final_only", "none")
RADIUS_POLICY_CHOICES = ("legacy_kdtree_mean4", "discrete_physical_coverage")
COVERAGE_REPAIR_POLICY_CHOICES = ("none", "nearest_rnode")


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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "warmup_cosine", "rapid_decay", "two_stage", "second_stage"),
        default="warmup_cosine",
    )
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--second-stage-epoch", type=int, default=0)
    parser.add_argument("--second-stage-lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("manual_gd", "adam", "adamw"), default="adamw")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--node-latent-size", type=int, default=MODEL_CONFIG["node_latent_size"])
    parser.add_argument("--edge-latent-size", type=int, default=MODEL_CONFIG["edge_latent_size"])
    parser.add_argument("--processor-steps", type=int, default=MODEL_CONFIG["processor_steps"])
    parser.add_argument("--mlp-hidden-layers", type=int, default=MODEL_CONFIG["mlp_hidden_layers"])
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--validation-batch-size", type=int, default=0)
    parser.add_argument("--prediction-batch-size", type=int, default=0)
    parser.add_argument(
        "--prediction-split",
        choices=("all", "train", "valid_iid", "valid_stress"),
        default="all",
        help="Limit final/best prediction export to one split; training behavior is unchanged.",
    )
    parser.add_argument("--shuffle-train-batches", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
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
        default="legacy_kdtree_mean4",
        help="Heat3D graph radius policy. Default preserves legacy graph behavior.",
    )
    parser.add_argument(
        "--coverage-repair-policy",
        choices=COVERAGE_REPAIR_POLICY_CHOICES,
        default="none",
        help="Optional Heat3D graph coverage repair policy. Default disables repair.",
    )
    parser.add_argument("--repair-p2r", dest="repair_p2r", action="store_true", default=True)
    parser.add_argument("--no-repair-p2r", dest="repair_p2r", action="store_false")
    parser.add_argument("--repair-r2p", dest="repair_r2p", action="store_true", default=True)
    parser.add_argument("--no-repair-r2p", dest="repair_r2p", action="store_false")
    parser.add_argument("--min-physical-coverage", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=("valid_loss", "valid_raw_deltaT_mse", "valid_base_mse"),
        default="valid_loss",
        help="Validation metric used to track the best epoch for optional best prediction export.",
    )
    parser.add_argument("--save-best-predictions", action="store_true")
    parser.add_argument("--best-predictions-name", type=str, default="best_predictions.npz")
    parser.add_argument("--report-every", type=int, default=1)
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
        choices=("mse", "background_hotspot", "background_l1_bias", "background_l1_relative", "background_pseudo_negative"),
        default="mse",
    )
    parser.add_argument("--background-quantile", type=float, default=0.50)
    parser.add_argument("--hotspot-quantile", type=float, default=0.90)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--hotspot-weight", type=float, default=0.1)
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


def _deltaT_error_pct(raw_delta_mse: Any, mean_abs_true_deltaT: Any) -> float | None:
    if raw_delta_mse is None or mean_abs_true_deltaT is None:
        return None
    try:
        mse = float(raw_delta_mse)
        denominator = float(mean_abs_true_deltaT)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(mse) or not math.isfinite(denominator) or denominator <= 0.0:
        return None
    return 100.0 * math.sqrt(max(mse, 0.0)) / denominator


def _metric_error_pct(metrics: dict[str, Any] | None) -> float | None:
    if metrics is None:
        return None
    if metrics.get("raw_deltaT_relative_rmse_pct") is not None:
        return float(metrics["raw_deltaT_relative_rmse_pct"])
    return _deltaT_error_pct(metrics.get("raw_delta_mse"), metrics.get("mean_abs_true_deltaT"))


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
    numeric_keys = (
        "normalized_loss",
        "raw_delta_mse",
        "recovered_temperature_mse",
        "mean_abs_true_deltaT",
    )
    combined = {
        key: float(
            sum(float(metrics.get(key, 0.0)) * count for count, metrics in weighted_entries)
            / total_count
        )
        for key in numeric_keys
    }
    combined["raw_deltaT_relative_rmse_pct"] = _metric_error_pct(combined)
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
            "prediction_batches": int(all_group_count),
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
        f"all_groups={counts['all_groups']} valid_batches={counts['valid_batches']} "
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


def _run_text_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _gpu_memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "gpus": [],
        "jax_devices": [],
        "process_gpu_memory_mb": None,
    }
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

    gpu_output = _run_text_command(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_output:
        for index, line in enumerate(gpu_output.splitlines()):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2:
                try:
                    snapshot["gpus"].append(
                        {
                            "index": int(index),
                            "memory_used_mb": int(float(parts[0])),
                            "memory_total_mb": int(float(parts[1])),
                        }
                    )
                except ValueError:
                    continue

    process_output = _run_text_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if process_output:
        current_pid = os.getpid()
        total = 0
        found = False
        for line in process_output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                used = int(float(parts[1]))
            except ValueError:
                continue
            if pid == current_pid:
                total += used
                found = True
        if found:
            snapshot["process_gpu_memory_mb"] = int(total)
    return snapshot


class MemoryAudit:
    def __init__(self, path: Path, *, every_batch: bool = False, gc_enabled: bool = False):
        self.path = path
        self.every_batch = bool(every_batch)
        self.gc_enabled = bool(gc_enabled)
        self.event_index = 0
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
        gpu = _gpu_memory_snapshot()
        payload = {
            "event_index": self.event_index,
            "time_unix": time.time(),
            "stage": stage,
            "epoch": epoch,
            "batch_index": batch_index,
            "split": split,
            "rss_mb": _current_rss_mb(),
            "gpu_memory": gpu,
            "detail": detail or {},
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")
            file.flush()

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
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    mapping = loaded.get("sample_splits", loaded)
    if not isinstance(mapping, dict):
        raise ValueError(f"--split-map must be a mapping or contain sample_splits: {path}")
    split_ids: dict[str, list[str]] = {}
    for sample_id, split in mapping.items():
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"--split-map contains invalid sample_id: {sample_id!r}")
        if not isinstance(split, str) or not split:
            raise ValueError(f"--split-map contains invalid split for {sample_id!r}: {split!r}")
        split_ids.setdefault(split, []).append(sample_id)
    return {split: sorted(ids) for split, ids in split_ids.items()}


def _resolve_training_splits(
    sample_root: Path,
    split_map_path: Path | None,
) -> tuple[dict[str, list[str]], str, str, str | None]:
    if split_map_path is None:
        split_ids = _subset_split_ids(sample_root)
        _require_train_valid_splits(split_ids)
        return split_ids, "sample_meta", "valid", None

    split_ids = _load_external_split_map(split_map_path)
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
            "base and hotspot terms use normalized_deltaT; background MSE/L1/bias/overprediction/relative "
            "terms use raw_deltaT_K"
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
    }


def _optimizer_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "optimizer": args.optimizer,
        "gradient_clip_norm": (
            None if args.gradient_clip_norm is None else float(args.gradient_clip_norm)
        ),
        "weight_decay": float(args.weight_decay),
    }


def _model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    model_config = dict(MODEL_CONFIG)
    model_config.update(
        {
            "node_latent_size": int(args.node_latent_size),
            "edge_latent_size": int(args.edge_latent_size),
            "processor_steps": int(args.processor_steps),
            "mlp_hidden_layers": int(args.mlp_hidden_layers),
        }
    )
    return model_config


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
    }


def _graph_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
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


def _validate_optimizer_config(config: dict[str, Any]) -> None:
    if config["optimizer"] not in {"manual_gd", "adam", "adamw"}:
        raise ValueError("--optimizer must be manual_gd, adam, or adamw")
    gradient_clip_norm = config.get("gradient_clip_norm")
    if gradient_clip_norm is not None and float(gradient_clip_norm) <= 0.0:
        raise ValueError("--gradient-clip-norm must be > 0 when provided")
    if float(config["weight_decay"]) < 0.0:
        raise ValueError("--weight-decay must be >= 0")


def _validate_model_config(config: dict[str, Any]) -> None:
    for key in ("node_latent_size", "edge_latent_size", "processor_steps", "mlp_hidden_layers"):
        if int(config[key]) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be >= 1")


def _validate_batch_config(config: dict[str, Any]) -> None:
    for key in ("batch_size", "validation_batch_size", "prediction_batch_size"):
        value = config.get(key)
        if value is not None and int(value) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be >= 1 or 0 for legacy full-batch")


def _validate_graph_config(config: dict[str, Any]) -> None:
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
        if second_stage_epoch <= 0 or epoch < second_stage_epoch:
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
    raise ValueError(f"Unsupported lr schedule: {schedule}")


def _loss_weight_keys() -> tuple[str, ...]:
    return (
        "background_l1_weight",
        "background_bias_weight",
        "background_over_weight",
        "background_relative_weight",
        "hotspot_weight",
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


def _normalized_delta_to_raw(pred_normalized, stats: dict):
    return pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]


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


def _loss_components(model, params, groups: list[dict], stats: dict, loss_config: dict[str, Any]) -> dict[str, Any]:
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
        "total_loss": 0.0,
        "bg_pred_raw_mean": 0.0,
        "bg_signed_bias": 0.0,
        "bg_abs_mean": 0.0,
        "hotspot_raw_mae": 0.0,
    }
    count = 0
    pseudo_negative_count = jnp.asarray(0.0)
    for group in groups:
        pred = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        target = group["target_normalized"]
        target_raw = group["target_delta_raw"]
        pred_raw_delta = _normalized_delta_to_raw(pred, stats)
        base_mse = jnp.mean(jnp.square(pred - target))
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
        raw_error = pred_raw_delta - target_raw
        if loss_config["loss_mode"] == "background_hotspot":
            background_penalty = _masked_mean(jnp.square(pred_raw_delta), background_mask)
            hotspot_retention_loss = _masked_mean(jnp.square(pred - target), hotspot_mask)
            total_loss = (
                base_mse
                + loss_config["background_weight"] * background_penalty
                + loss_config["hotspot_weight"] * hotspot_retention_loss
            )
        elif loss_config["loss_mode"] in {"background_l1_bias", "background_l1_relative", "background_pseudo_negative"}:
            background_l1 = _masked_mean(jnp.abs(pred_raw_delta), background_mask)
            background_signed_bias_loss = jnp.abs(_masked_mean(raw_error, background_mask))
            background_overprediction_loss = _masked_mean(jnp.maximum(raw_error, 0.0), background_mask)
            hotspot_retention_loss = _masked_mean(jnp.square(pred - target), hotspot_mask)
            if loss_config["loss_mode"] in {"background_l1_relative", "background_pseudo_negative"}:
                denom = _safe_relative_denominator(target_raw, loss_config)
                background_relative_abs = _masked_mean(jnp.abs(raw_error) / denom, background_mask)
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
                    _masked_mean(raw_error, pseudo_negative_mask),
                    jnp.asarray(0.0, dtype=base_mse.dtype),
                )
                pseudo_negative_over_ratio = jnp.where(
                    enough_points,
                    _masked_mean((raw_error > loss_config["pseudo_negative_over_margin"]).astype(base_mse.dtype), pseudo_negative_mask),
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
        else:
            total_loss = base_mse
        bg_pred_raw_mean = _masked_mean(pred_raw_delta, background_mask)
        bg_signed_bias = _masked_mean(raw_error, background_mask)
        bg_abs_mean = _masked_mean(jnp.abs(raw_error), background_mask)
        hotspot_raw_mae = _masked_mean(jnp.abs(raw_error), hotspot_mask)
        n = target.shape[0]
        weighted["base_mse"] = weighted["base_mse"] + base_mse * n
        weighted["background_penalty"] = weighted["background_penalty"] + background_penalty * n
        weighted["background_l1"] = weighted["background_l1"] + background_l1 * n
        weighted["background_signed_bias_loss"] = (
            weighted["background_signed_bias_loss"] + background_signed_bias_loss * n
        )
        weighted["background_overprediction_loss"] = (
            weighted["background_overprediction_loss"] + background_overprediction_loss * n
        )
        weighted["background_relative_abs"] = weighted["background_relative_abs"] + background_relative_abs * n
        weighted["pseudo_negative_over_loss"] = weighted["pseudo_negative_over_loss"] + pseudo_negative_over_loss * n
        weighted["pseudo_negative_unweighted_loss"] = (
            weighted["pseudo_negative_unweighted_loss"] + pseudo_negative_unweighted_loss * n
        )
        weighted["pseudo_negative_weighted_loss"] = (
            weighted["pseudo_negative_weighted_loss"] + pseudo_negative_weighted_loss * n
        )
        weighted["pseudo_negative_weighted_fraction_of_total_loss"] = (
            weighted["pseudo_negative_weighted_fraction_of_total_loss"] + pseudo_negative_weighted_fraction * n
        )
        weighted["pseudo_negative_bias"] = weighted["pseudo_negative_bias"] + pseudo_negative_bias * n
        weighted["pseudo_negative_over_ratio"] = weighted["pseudo_negative_over_ratio"] + pseudo_negative_over_ratio * n
        weighted["hotspot_retention_loss"] = weighted["hotspot_retention_loss"] + hotspot_retention_loss * n
        weighted["total_loss"] = weighted["total_loss"] + total_loss * n
        weighted["bg_pred_raw_mean"] = weighted["bg_pred_raw_mean"] + bg_pred_raw_mean * n
        weighted["bg_signed_bias"] = weighted["bg_signed_bias"] + bg_signed_bias * n
        weighted["bg_abs_mean"] = weighted["bg_abs_mean"] + bg_abs_mean * n
        weighted["hotspot_raw_mae"] = weighted["hotspot_raw_mae"] + hotspot_raw_mae * n
        count += int(n)
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
    count = 0
    finite_ok = True
    shape_ok = True
    for group in groups:
        pred_normalized = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        pred_delta = pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]
        recovered = group["t_ref"] + pred_delta
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
    raw_delta_mse = float(weighted_raw_delta_mse / divisor)
    return {
        "normalized_loss": float(weighted_normalized_loss / divisor),
        "raw_delta_mse": raw_delta_mse,
        "recovered_temperature_mse": float(weighted_recovered_mse / divisor),
        "mean_abs_true_deltaT": mean_abs_true_delta,
        "raw_deltaT_relative_rmse_pct": _deltaT_error_pct(raw_delta_mse, mean_abs_true_delta),
        "finite_ok": finite_ok,
        "shape_ok": shape_ok,
    }


def _optax_learning_rate_schedule(epochs: int, lr_config: dict[str, Any]):
    schedule = lr_config["lr_schedule"]
    base_lr = float(lr_config["lr"])
    updates_per_epoch = max(int(lr_config.get("updates_per_epoch", 1)), 1)

    if schedule == "constant":
        return base_lr

    def learning_rate(count):
        update_count = jnp.asarray(count, dtype=jnp.float32)
        epoch = jnp.floor(update_count / float(updates_per_epoch)) + 1.0
        legacy_epoch = update_count + 1.0
        base = jnp.asarray(base_lr, dtype=jnp.float32)
        if schedule == "two_stage":
            second_stage_epoch = int(lr_config["second_stage_epoch"])
            if second_stage_epoch <= 0:
                return base
            second_lr = jnp.asarray(float(lr_config["second_stage_lr"]), dtype=jnp.float32)
            return jnp.where(legacy_epoch <= float(second_stage_epoch), base, second_lr)

        if schedule == "second_stage":
            second_stage_epoch = int(lr_config["second_stage_epoch"])
            if second_stage_epoch <= 0:
                return base
            second_lr = jnp.asarray(float(lr_config["second_stage_lr"]), dtype=jnp.float32)
            return jnp.where(legacy_epoch < float(second_stage_epoch), base, second_lr)

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
        "train_error_pct": _metric_error_pct(train_metrics),
        "valid_error_pct": _metric_error_pct(valid_metrics),
        "valid_iid_error_pct": _metric_error_pct(valid_metrics) if primary_validation_split == "valid_iid" else None,
        "train_recovered_T_mse": _maybe_float(train_metrics, "recovered_temperature_mse"),
        "valid_recovered_T_mse": float(valid_metrics["recovered_temperature_mse"]),
        "valid_iid_recovered_T_mse": (
            float(valid_metrics["recovered_temperature_mse"]) if primary_validation_split == "valid_iid" else None
        ),
        "train_full_metrics_computed": train_components is not None and train_metrics is not None,
    }
    if valid_stress_components is not None and valid_stress_metrics is not None:
        record.update(
            {
                "stress_validation_split": stress_validation_split or "valid_stress",
                "valid_stress_loss": float(valid_stress_components["total_loss"]),
                "valid_stress_base_mse": float(valid_stress_components["base_mse"]),
                "valid_stress_raw_deltaT_mse": float(valid_stress_metrics["raw_delta_mse"]),
                "valid_stress_error_pct": _metric_error_pct(valid_stress_metrics),
                "valid_stress_recovered_T_mse": float(valid_stress_metrics["recovered_temperature_mse"]),
                "valid_stress_bg_signed_bias": float(valid_stress_components["bg_signed_bias"]),
                "valid_stress_hotspot_raw_mae": float(valid_stress_components["hotspot_raw_mae"]),
            }
        )
    record.update(_current_weight_payload(current_loss_config))
    return record


def _print_epoch_progress(record: dict[str, Any], epochs: int, log_mode: str) -> None:
    if log_mode == "quiet":
        return
    if log_mode == "compact":
        train_loss = _first_progress_numeric(
            record.get("train_loss"),
            record.get("epoch_mean_train_batch_loss"),
        )
        valid_iid_loss = record.get("valid_iid_loss")
        if valid_iid_loss is None:
            valid_iid_loss = record.get("valid_loss")
        valid_iid_error_pct = record.get("valid_iid_error_pct")
        if valid_iid_error_pct is None:
            valid_iid_error_pct = record.get("valid_error_pct")
        _emit(
            f"epoch {record['epoch']}/{epochs} "
            f"lr={record['lr']:.2e} "
            f"train={_format_progress_decimal(train_loss)} "
            f"iid={_format_progress_decimal(valid_iid_loss)} "
            f"iid_err={_format_progress_percent(valid_iid_error_pct)} "
            f"stress={_format_progress_decimal(record.get('valid_stress_loss'))} "
            f"stress_err={_format_progress_percent(record.get('valid_stress_error_pct'))} "
            f"best=e{_format_progress_int(record.get('best_epoch'))}/"
            f"{_format_progress_decimal(record.get('best_valid_iid_loss'))}"
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
        f"train_recovered_T_mse={record['train_recovered_T_mse']:.8e} "
        f"valid_recovered_T_mse={record['valid_recovered_T_mse']:.8e} "
        f"current_background_l1_weight={record['current_background_l1_weight']:.8e} "
        f"current_background_bias_weight={record['current_background_bias_weight']:.8e} "
        f"current_background_over_weight={record['current_background_over_weight']:.8e} "
        f"current_background_relative_weight={record['current_background_relative_weight']:.8e} "
        f"current_hotspot_weight={record['current_hotspot_weight']:.8e}"
    )


def _fit_once(
    train_groups: list[dict],
    valid_groups: list[dict],
    valid_stress_groups: list[dict],
    stats: dict,
    epochs: int,
    lr_config: dict[str, Any],
    seed: int,
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
    timings: dict[str, float] | None = None,
    profile_enabled: bool = False,
    memory_audit: MemoryAudit | None = None,
    primary_validation_split: str = "valid",
    stress_validation_split: str | None = None,
) -> dict:
    timings = timings if timings is not None else {}
    init_start = time.perf_counter()
    if memory_audit is not None:
        memory_audit.record("model_init_start")
    _progress(progress_enabled, "startup", "initializing model parameters ...")
    model = GraphNeuralOperator(**model_config)
    params = model.init(
        jax.random.PRNGKey(seed),
        inputs=train_groups[0]["inputs"],
        graphs=train_groups[0]["graphs"],
    )["params"]
    _record_timing(timings, "model_init", init_start)
    if memory_audit is not None:
        memory_audit.record("model_init_end")
    _progress(progress_enabled, "startup", "model parameters initialized", init_start)

    batch_enabled = batch_config.get("batch_size") is not None
    metrics_fn = _weighted_metrics if batch_enabled else _metrics
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
        epoch_grad_norms: list[float] = []
        epoch_update_norms: list[float] = []
        epoch_param_norms: list[float] = []
        epoch_update_to_param_ratios: list[float] = []
        if batch_enabled:
            train_epoch_groups = _epoch_train_groups(
                train_groups,
                epoch=epoch,
                seed=seed,
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
                    return _loss_components(model, current_params, [group], stats, current_loss_config)["total_loss"]

                batch_start = time.perf_counter()
                loss_grad_start = time.perf_counter()
                loss_value, grads = jax.value_and_grad(loss_fn)(params)
                if profile_enabled:
                    _block_until_ready_tree((loss_value, grads))
                loss_grad_time = time.perf_counter() - loss_grad_start
                batch_loss_value = float(loss_value)
                epoch_train_batch_losses.append(batch_loss_value)

                grad_norm_reported = should_report_grad_norm(grad_norm_report_every, batch_index)
                compute_batch_norms = bool(grad_norm_reported or profile_enabled)
                grad_norm = None
                grad_norm_time = 0.0
                if compute_batch_norms:
                    grad_norm_start = time.perf_counter()
                    grad_norm = _global_norm(grads)
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
                        "total_batch_time": float(total_batch_time),
                        "loss_grad_time": float(loss_grad_time),
                        "grad_norm_time": float(grad_norm_time),
                        "grad_norm": float(grad_norm) if grad_norm is not None else None,
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
                del grads, updates, loss_value
            epoch_batch_counts.append(len(train_epoch_groups))
            if batch_grad_norms:
                grad_norms.append(float(np.mean(batch_grad_norms)))
        else:
            epoch_train_batch_order_hashes.append(_group_sample_id_hash(train_groups))
            def loss_fn(current_params):
                return _loss_components(model, current_params, train_groups, stats, current_loss_config)["total_loss"]

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
            loss_value, grads = jax.value_and_grad(loss_fn)(params)
            if profile_enabled:
                _block_until_ready_tree((loss_value, grads))
            loss_grad_time = time.perf_counter() - loss_grad_start
            batch_loss_value = float(loss_value)
            epoch_train_batch_losses.append(batch_loss_value)

            grad_norm_reported = should_report_grad_norm(grad_norm_report_every, 1)
            compute_batch_norms = bool(grad_norm_reported or profile_enabled)
            grad_norm = None
            grad_norm_time = 0.0
            if compute_batch_norms:
                grad_norm_start = time.perf_counter()
                grad_norm = _global_norm(grads)
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
                    "total_batch_time": float(total_batch_time),
                    "loss_grad_time": float(loss_grad_time),
                    "grad_norm_time": float(grad_norm_time),
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
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
            del grads, updates, loss_value
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
        record["epoch_min_train_batch_loss"] = batch_loss_summary["min"]
        record["epoch_max_train_batch_loss"] = batch_loss_summary["max"]
        record["epoch_mean_grad_norm"] = grad_norm_summary["mean"]
        record["epoch_max_grad_norm"] = grad_norm_summary["max"]
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
            best_params = _copy_params(params)
            if memory_audit is not None:
                memory_audit.record("best_params_copy_end", epoch=epoch)
        record["best_epoch"] = best_record.get("epoch") if best_record is not None else None
        record["best_valid_iid_loss"] = best_record.get("valid_iid_loss") if best_record is not None else None
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
        "updates_per_epoch": int(updates_per_epoch),
        "total_update_count": int(sum(epoch_batch_counts)),
        "train_group_count": int(len(train_groups)),
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
        "best_record": best_record,
        "best_params": best_params,
        "best_score": best_score,
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
        pred_normalized = model.apply({"params": params}, inputs=group["inputs"], graphs=group["graphs"])
        pred_delta = pred_normalized * stats["target_delta_std"] + stats["target_delta_mean"]
        recovered = np.asarray(group["t_ref"] + pred_delta)
        if not np.all(np.isfinite(recovered)):
            raise ValueError(f"Non-finite recovered predictions in group {group['name']}")
        for batch_index, sample_id in enumerate(group["sample_ids"]):
            predictions[sample_id] = recovered[batch_index, 0, :, :].astype(np.float64)
    return predictions


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
        f"optimizer={optimizer_config['optimizer']} seed={args.seed} report_every={args.report_every}"
    )
    _emit(
        "  model: "
        f"node_latent_size={model_config['node_latent_size']} "
        f"edge_latent_size={model_config['edge_latent_size']} "
        f"processor_steps={model_config['processor_steps']} "
        f"mlp_hidden_layers={model_config['mlp_hidden_layers']}"
    )
    _emit(
        "  output: "
        f"dir={output_dir} save_predictions={bool(args.save_predictions)} "
        f"save_best_predictions={bool(args.save_best_predictions)}"
    )
    _emit(
        "  batching: "
        f"mode={'mini_batch' if batch_config['batch_size'] is not None else 'legacy_full_batch'} "
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
    if args.log_mode == "compact":
        _emit(
            "  loss: "
            f"mode={loss_config['loss_mode']} weight_schedule={loss_config['loss_weight_schedule']} "
            f"bg_q={loss_config['background_quantile']} hot_q={loss_config['hotspot_quantile']} "
            f"rel_w={loss_config['background_relative_weight']} hot_w={loss_config['hotspot_weight']} "
            f"pn_type={loss_config['pseudo_negative_loss_type']} pn_w={loss_config['pseudo_negative_weight']}"
        )
    else:
        _emit(
            "  lr schedule params: "
            f"warmup_epochs={lr_config['warmup_epochs']} min_lr={lr_config['min_lr']} "
            f"second_stage_epoch={lr_config['second_stage_epoch']} "
            f"second_stage_lr={lr_config['second_stage_lr']}"
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
            f"background_weight={loss_config['background_weight']} "
            f"hotspot_weight={loss_config['hotspot_weight']} "
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
    final_prediction_export_skipped: bool,
    final_prediction_export_skip_reason: str | None,
    timings: dict[str, float],
) -> None:
    lr_history_summary = _sequence_summary(result["lr_history"])
    relative_weight_summary = _history_field_summary(
        result["loss_weight_history"], "current_background_relative_weight"
    )
    hotspot_weight_summary = _history_field_summary(result["loss_weight_history"], "current_hotspot_weight")
    best = result.get("best_record") or {}

    _emit("")
    _emit("summary")
    _emit(
        "  final: "
        f"epoch={result['final_epoch']} valid_loss={result['final_valid_loss']:.8e} "
        f"valid_base_mse={result['final_valid_loss_components']['base_mse']:.8e} "
        f"valid_raw_deltaT_mse={result['valid_metrics']['raw_delta_mse']:.8e}"
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
        f"valid_raw_deltaT_mse={best.get('valid_raw_deltaT_mse'):.8e}"
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
        "  status: "
        f"grad_finite={result['grad_finite']} checkpoint_saved=False export_smoke_ok={result['status_ok']}"
    )

    if args.log_mode == "full":
        _emit("  loss/optimization")
        _emit(f"    model config: {model_config}")
        _emit(f"    loss mode: {loss_config['loss_mode']}")
        _emit(f"    loss weight schedule: {loss_config['loss_weight_schedule']}")
        _emit(f"    relative weight summary: {relative_weight_summary}")
        _emit(f"    hotspot weight summary: {hotspot_weight_summary}")
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
            f"valid_hotspot_mae={result['final_valid_loss_components']['hotspot_raw_mae']:.8e}"
        )

    _progress(_progress_enabled(args), "startup-summary", _timing_summary(timings))
    _progress(_progress_enabled(args), "done", "script complete")


def _make_groups_with_progress(
    examples,
    stats: dict,
    builder: Heat3DGraphBuilder,
    label: str,
    progress_enabled: bool,
    progress_detail: str,
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
        signature = _metadata_shape_signature(builder.build_metadata(example.condition.coords))
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
        result.append(_make_batch_group(batch_group_name, batch_examples, stats, builder))
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
) -> list[dict]:
    groups_by_split = {
        "all": all_groups,
        "train": train_groups,
        "valid_iid": valid_groups,
        "valid_stress": valid_stress_groups,
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
    model_config = _model_config_from_args(args)
    batch_config = _batch_config_from_args(args)
    graph_config = _graph_config_from_args(args)
    _validate_loss_config(loss_config)
    _validate_lr_config(lr_config)
    _validate_optimizer_config(optimizer_config)
    _validate_model_config(model_config)
    _validate_batch_config(batch_config)
    _validate_graph_config(graph_config)

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
    split_counts = {split: len(ids) for split, ids in sorted(split_ids.items())}
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
    all_examples = [dataset[index_by_id[sample_id]] for sample_id in all_ids]
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
    group_start = time.perf_counter()
    _progress(progress_enabled, "startup", "building grouped JAX arrays and graphs ...")
    train_groups = _make_groups_with_progress(
        train_examples,
        stats,
        builder,
        "train",
        progress_detail_enabled,
        args.progress_detail,
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
            batch_size=batch_config["validation_batch_size"],
            drop_last=False,
            profile_counts=profile_counts if profile_enabled else None,
        )
        if stress_validation_split is not None and valid_stress_examples
        else []
    )
    all_groups = _make_groups_with_progress(
        all_examples,
        stats,
        builder,
        "all",
        progress_detail_enabled,
        args.progress_detail,
        batch_size=batch_config["prediction_batch_size"],
        drop_last=False,
        profile_counts=profile_counts if profile_enabled else None,
    )
    _require_nonempty_groups(train_groups, "train")
    _require_nonempty_groups(valid_groups, primary_validation_split)
    _require_nonempty_groups(all_groups, "all")
    _record_timing(timings, "group_build", group_start)
    _progress(
        progress_enabled,
        "startup",
        (
            "groups built: "
            f"train_groups={len(train_groups)} {primary_validation_split}_groups={len(valid_groups)} "
            f"valid_stress_groups={len(valid_stress_groups)} all_groups={len(all_groups)}"
        ),
        group_start,
    )
    if memory_audit is not None:
        memory_audit.record(
            "groups_built",
            detail={
                "train": _groups_memory_signature(train_groups),
                primary_validation_split: _groups_memory_signature(valid_groups),
                "valid_stress": _groups_memory_signature(valid_stress_groups),
                "all": _groups_memory_signature(all_groups),
            },
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
        args.seed,
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
        timings,
        profile_enabled=profile_enabled,
        memory_audit=memory_audit,
        primary_validation_split=primary_validation_split,
        stress_validation_split=stress_validation_split,
    )
    prediction_groups = _prediction_groups_for_split(
        args.prediction_split,
        all_groups=all_groups,
        train_groups=train_groups,
        valid_groups=valid_groups,
        valid_stress_groups=valid_stress_groups,
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
        best_predictions = _predict_temperatures(result["model"], result["best_params"], prediction_groups, stats)
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
        "optimizer": optimizer_config["optimizer"],
        "gradient_clip_norm": optimizer_config["gradient_clip_norm"],
        "weight_decay": optimizer_config["weight_decay"],
        "model_config": model_config,
        **_batch_config_payload(batch_config),
        "seed": args.seed,
        "boundary_mask_fallback": bool(args.boundary_mask_fallback),
        "graph_config": graph_config,
        "route": "relative BC features + zero_delta_u_bridge + normalized DeltaT target",
        "output_dir": str(output_dir),
        "save_predictions": bool(args.save_predictions),
        "prediction_split": args.prediction_split,
        "predictions_path": str(predictions_path) if args.save_predictions else None,
        "save_best_predictions": bool(args.save_best_predictions),
        "best_predictions_name": args.best_predictions_name,
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
        "valid_iid_group_count": result["valid_iid_group_count"],
        "valid_stress_group_count": result["valid_stress_group_count"],
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
        "final_prediction_export_skipped": bool(final_prediction_export_skipped),
        "final_prediction_export_skip_reason": final_prediction_export_skip_reason,
        **best_selection,
        "checkpoint_saved": False,
        "loss_mode": loss_config["loss_mode"],
        "background_quantile": loss_config["background_quantile"],
        "hotspot_quantile": loss_config["hotspot_quantile"],
        "background_weight": loss_config["background_weight"],
        "hotspot_weight": loss_config["hotspot_weight"],
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
        "graph_config": graph_config,
        "split_counts": split_counts,
        "boundary_mask_fallback": bool(args.boundary_mask_fallback),
        "timing_diagnostics": dict(timings),
        "timing_profile_counts": dict(profile_counts),
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "valid_iid_ids": valid_ids if primary_validation_split == "valid_iid" else [],
        "valid_stress_ids": valid_stress_ids,
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
        "valid_iid_group_count": result["valid_iid_group_count"],
        "valid_stress_group_count": result["valid_stress_group_count"],
        "train_group_sample_id_hash": result["train_group_sample_id_hash"],
        "valid_iid_sample_id_hash": result["valid_iid_sample_id_hash"],
        "valid_stress_sample_id_hash": result["valid_stress_sample_id_hash"],
        "deterministic_audit_enabled": result["deterministic_audit_enabled"],
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
        "final_prediction_export_skipped": bool(final_prediction_export_skipped),
        "final_prediction_export_skip_reason": final_prediction_export_skip_reason,
        **best_selection,
        "loss_mode": loss_config["loss_mode"],
        "background_quantile": loss_config["background_quantile"],
        "hotspot_quantile": loss_config["hotspot_quantile"],
        "background_weight": loss_config["background_weight"],
        "hotspot_weight": loss_config["hotspot_weight"],
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
        "optimizer": optimizer_config["optimizer"],
        "gradient_clip_norm": optimizer_config["gradient_clip_norm"],
        "weight_decay": optimizer_config["weight_decay"],
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

    profile_payload = _profile_timing_payload(
        timings=timings,
        profile_counts=profile_counts,
        epoch_records=result["epoch_history"],
        train_batch_records=result["train_batch_records"] if profile_enabled else [],
        validation_batch_records=result["validation_batch_records"] if profile_enabled else [],
        train_group_count=len(train_groups),
        valid_group_count=len(valid_groups),
        all_group_count=len(all_groups),
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
        final_prediction_export_skipped=final_prediction_export_skipped,
        final_prediction_export_skip_reason=final_prediction_export_skip_reason,
        timings=timings,
    )
    _progress(progress_enabled, "done", "script complete", script_start)
    return 0 if result["status_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
