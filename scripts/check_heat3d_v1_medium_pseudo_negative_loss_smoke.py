#!/usr/bin/env python3
"""Smoke-check Heat3D v1 pseudo-negative background loss components."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


class _FakeModel:
    def apply(self, _params, *, inputs, graphs):  # noqa: D401
        del graphs
        return inputs["pred_normalized"]


def _parse(argv: list[str]):
    original = sys.argv[:]
    try:
        sys.argv = ["run_heat3d_v1_medium_controlled_training_export.py", *argv]
        return runner.parse_args()
    finally:
        sys.argv = original


def _components(loss_mode: str, loss_type: str = "mse"):
    args = _parse(
        [
            "--loss-mode",
            loss_mode,
            "--background-quantile",
            "0.50",
            "--hotspot-quantile",
            "0.90",
            "--background-relative-weight",
            "0.05",
            "--pseudo-negative-quantile",
            "0.50",
            "--pseudo-negative-weight",
            "0.10",
            "--pseudo-negative-over-margin",
            "0.0",
            "--pseudo-negative-min-count",
            "1",
            "--pseudo-negative-loss-type",
            loss_type,
            "--pseudo-negative-relative-floor",
            "0.02",
        ]
    )
    loss_config = runner._loss_config_from_args(args)
    runner._validate_loss_config(loss_config)
    target_raw = jnp.asarray([[[[0.00], [0.01], [0.12], [0.35]]]], dtype=jnp.float32)
    pred_raw = jnp.asarray([[[[0.04], [0.05], [0.10], [0.30]]]], dtype=jnp.float32)
    group = {
        "inputs": {"pred_normalized": pred_raw},
        "graphs": None,
        "target_normalized": target_raw,
        "target_delta_raw": target_raw,
    }
    stats = {
        "target_delta_mean": jnp.asarray(0.0, dtype=jnp.float32),
        "target_delta_std": jnp.asarray(1.0, dtype=jnp.float32),
    }
    return runner._loss_components(_FakeModel(), {}, [group], stats, loss_config), loss_config


def main() -> int:
    default_args = _parse([])
    mse_components, _ = _components("mse")
    rel_components, rel_config = _components("background_l1_relative")
    pn_components, pn_config = _components("background_pseudo_negative", "mse")
    pn_l1_components, pn_l1_config = _components("background_pseudo_negative", "l1")
    pn_rel_l1_components, pn_rel_l1_config = _components("background_pseudo_negative", "relative_l1")
    pn_rel_mse_components, pn_rel_mse_config = _components("background_pseudo_negative", "relative_mse")
    metrics = {"raw_delta_mse": 1.0, "recovered_temperature_mse": 1.0}
    record = runner._epoch_history_record(
        1,
        1e-3,
        runner._loss_config_for_epoch(pn_config, 1),
        pn_components,
        pn_components,
        metrics,
        metrics,
    )
    loss_summary_stub = {
        "loss_mode": pn_config["loss_mode"],
        "pseudo_negative_quantile": pn_config["pseudo_negative_quantile"],
        "pseudo_negative_delta_threshold": pn_config["pseudo_negative_delta_threshold"],
        "pseudo_negative_weight": pn_config["pseudo_negative_weight"],
        "pseudo_negative_over_margin": pn_config["pseudo_negative_over_margin"],
        "pseudo_negative_loss_type": pn_config["pseudo_negative_loss_type"],
        "pseudo_negative_relative_floor": pn_config["pseudo_negative_relative_floor"],
        "epoch_history": [record],
        "final_train_loss_components": runner._loss_components_payload(pn_components),
    }
    json.dumps(loss_summary_stub, sort_keys=True)
    expected_mse = 0.0016
    expected_l1 = 0.04
    expected_relative_l1 = 2.0
    expected_relative_mse = 4.0
    checks = {
        "default_mse_unchanged": default_args.loss_mode == "mse",
        "default_pseudo_negative_loss_type_unchanged": default_args.pseudo_negative_loss_type == "mse",
        "l1_relative_mode_still_valid": rel_config["loss_mode"] == "background_l1_relative",
        "pseudo_negative_mode_valid": pn_config["loss_mode"] == "background_pseudo_negative",
        "pseudo_negative_mse_type_valid": pn_config["pseudo_negative_loss_type"] == "mse",
        "pseudo_negative_l1_type_valid": pn_l1_config["pseudo_negative_loss_type"] == "l1",
        "pseudo_negative_relative_l1_type_valid": pn_rel_l1_config["pseudo_negative_loss_type"] == "relative_l1",
        "pseudo_negative_relative_mse_type_valid": pn_rel_mse_config["pseudo_negative_loss_type"] == "relative_mse",
        "pseudo_negative_count_positive": float(pn_components["pseudo_negative_count"]) > 0.0,
        "pseudo_negative_over_loss_positive": float(pn_components["pseudo_negative_over_loss"]) > 0.0,
        "pseudo_negative_increases_total_loss": float(pn_components["total_loss"]) > float(rel_components["total_loss"]),
        "pseudo_negative_mse_matches_old_formula": abs(float(pn_components["pseudo_negative_unweighted_loss"]) - expected_mse)
        < 1e-6,
        "pseudo_negative_l1_formula": abs(float(pn_l1_components["pseudo_negative_unweighted_loss"]) - expected_l1)
        < 1e-6,
        "pseudo_negative_relative_l1_formula": abs(
            float(pn_rel_l1_components["pseudo_negative_unweighted_loss"]) - expected_relative_l1
        )
        < 1e-6,
        "pseudo_negative_relative_mse_formula": abs(
            float(pn_rel_mse_components["pseudo_negative_unweighted_loss"]) - expected_relative_mse
        )
        < 1e-6,
        "pseudo_negative_weighted_loss_recorded": abs(
            float(pn_components["pseudo_negative_weighted_loss"])
            - pn_config["pseudo_negative_weight"] * float(pn_components["pseudo_negative_unweighted_loss"])
        )
        < 1e-6,
        "pseudo_negative_weighted_fraction_recorded": float(
            pn_components["pseudo_negative_weighted_fraction_of_total_loss"]
        )
        > 0.0,
        "mse_pseudo_fields_zero": float(mse_components["pseudo_negative_over_loss"]) == 0.0
        and float(mse_components["pseudo_negative_count"]) == 0.0,
        "relative_pseudo_fields_zero": float(rel_components["pseudo_negative_over_loss"]) == 0.0
        and float(rel_components["pseudo_negative_count"]) == 0.0,
        "epoch_history_fields": all(
            key in record
            for key in (
                "train_pseudo_negative_count",
                "valid_pseudo_negative_count",
                "train_pseudo_negative_over_loss",
                "valid_pseudo_negative_over_loss",
                "train_pseudo_negative_unweighted_loss",
                "valid_pseudo_negative_unweighted_loss",
                "train_pseudo_negative_weighted_loss",
                "valid_pseudo_negative_weighted_loss",
                "train_pseudo_negative_weighted_fraction_of_total_loss",
                "valid_pseudo_negative_weighted_fraction_of_total_loss",
                "train_pseudo_negative_bias",
                "valid_pseudo_negative_bias",
                "train_pseudo_negative_over_ratio",
                "valid_pseudo_negative_over_ratio",
                "valid_pn_bias",
                "valid_pn_over",
                "valid_pn_over_ratio",
            )
        ),
        "summary_fields": all(
            key in loss_summary_stub
            for key in (
                "pseudo_negative_quantile",
                "pseudo_negative_delta_threshold",
                "pseudo_negative_weight",
                "pseudo_negative_over_margin",
                "pseudo_negative_loss_type",
                "pseudo_negative_relative_floor",
            )
        ),
    }
    ok = all(checks.values())
    print("Heat3D v1 pseudo-negative background loss smoke")
    print(f"checks: {checks}")
    print(f"pseudo_negative_loss_smoke_ok: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
