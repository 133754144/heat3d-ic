#!/usr/bin/env python3
"""Render the frozen full-field publication figures for selected cases.

The compact fixture is produced from the exact evaluator prediction artifacts
and contains canonical x_y_z flattened arrays.  This script never changes
values, interpolates, smooths, or chooses cases; it only reshapes the frozen
65x65x57 field and applies the already-frozen input-only selection receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/heat3d-v7-g2-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GRID = (65, 65, 57)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_rows(receipt: dict) -> dict[str, dict]:
    rows = {}
    for role, key in (("valid_iid", "valid_iid"), ("test_iid", "test_iid_visualization_only")):
        for label, entry in receipt[key].items():
            row = dict(entry)
            row["role"] = role
            row["selection_label"] = label
            rows[str(row["sample_id"])] = row
    return rows


def render_case(
    *, output_dir: Path, sample_id: str, role: str, z_index: int,
    truth: np.ndarray, heat3d: np.ndarray, thermfm: np.ndarray,
) -> dict:
    truth_grid = np.asarray(truth, dtype=np.float64).reshape(GRID)
    heat_grid = np.asarray(heat3d, dtype=np.float64).reshape(GRID)
    therm_grid = np.asarray(thermfm, dtype=np.float64).reshape(GRID)
    ref = truth_grid[:, :, z_index]
    preds = {"Heat3D U-v2": heat_grid[:, :, z_index], "Therm-FM": therm_grid[:, :, z_index]}
    errors = {name: value - ref for name, value in preds.items()}
    tmin, tmax = float(np.min(ref)), float(np.max(ref))
    pooled = np.concatenate([np.abs(value).reshape(-1) for value in errors.values()])
    emax = max(5.0, float(np.quantile(pooled, 0.99)))
    if not np.isfinite([tmin, tmax, emax]).all() or tmax <= tmin:
        raise ValueError(f"invalid display range for {sample_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), squeeze=False)
    temp_cmap = plt.get_cmap("turbo")
    err_cmap = plt.get_cmap("RdBu_r")
    diagnostics = {}
    for row_index, (name, pred) in enumerate(preds.items()):
        error = errors[name]
        diagnostics[name] = {
            "prediction_min_K": float(np.min(pred)),
            "prediction_max_K": float(np.max(pred)),
            "slice_rmse_K": float(np.sqrt(np.mean(np.square(error)))),
            "slice_mae_K": float(np.mean(np.abs(error))),
            "slice_peak_error_K": float(np.max(np.abs(error))),
            "prediction_below_colorbar": bool(np.any(pred < tmin)),
            "prediction_above_colorbar": bool(np.any(pred > tmax)),
        }
        for col, image in enumerate((ref, pred, error)):
            ax = axes[row_index, col]
            if col < 2:
                extend = "both" if diagnostics[name]["prediction_below_colorbar"] and diagnostics[name]["prediction_above_colorbar"] else ("min" if diagnostics[name]["prediction_below_colorbar"] else ("max" if diagnostics[name]["prediction_above_colorbar"] else "neither"))
                plot = ax.imshow(image, origin="lower", interpolation="nearest", cmap=temp_cmap, vmin=tmin, vmax=tmax)
                fig.colorbar(plot, ax=ax, fraction=0.047, pad=0.035, extend=extend, label="ΔT (K)")
            else:
                plot = ax.imshow(image, origin="lower", interpolation="nearest", cmap=err_cmap, vmin=-emax, vmax=emax)
                fig.colorbar(plot, ax=ax, fraction=0.047, pad=0.035, extend="both", label="error (K)")
            ax.set_xlabel("x index")
            ax.set_ylabel("y index")
            if row_index == 0:
                ax.set_title(("FVM reference", "prediction", "prediction − reference")[col])
        axes[row_index, 0].text(
            -0.43, 0.5,
            f"{name}\nRMSE {diagnostics[name]['slice_rmse_K']:.3f} K\nMAE {diagnostics[name]['slice_mae_K']:.3f} K\npeak {diagnostics[name]['slice_peak_error_K']:.3f} K",
            transform=axes[row_index, 0].transAxes, ha="right", va="center", fontsize=8.5,
        )
    fig.suptitle(f"V7 G2 · {role} · {sample_id} · exact dense slice z-index={z_index}", fontsize=14, y=0.99)
    fig.text(0.5, 0.01, f"shared reference color range [{tmin:.3f}, {tmax:.3f}] K; shared error range ±{emax:.3f} K; no interpolation or per-model rescaling", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0.12, 0.04, 1.0, 0.96))
    stem = f"{role}_{sample_id}_fullfield_exact_dense"
    fig.savefig(output_dir / f"{stem}.png", dpi=220)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)
    return {
        "sample_id": sample_id,
        "role": role,
        "z_index": int(z_index),
        "mode": "exact_dense_slice",
        "temperature_limits_K": [tmin, tmax],
        "error_limits_K": [-emax, emax],
        "metrics": diagnostics,
        "figure_png": str(output_dir / f"{stem}.png"),
        "figure_pdf": str(output_dir / f"{stem}.pdf"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.fixture, allow_pickle=False)
    receipt = json.loads(args.selection_receipt.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    selection = selected_rows(receipt)
    ids = [str(value) for value in data["sample_ids"]]
    roles = [str(value) for value in data["roles"]]
    z_indices = [int(value) for value in data["z_index"]]
    if len(ids) != len(set(ids)) or len(ids) != 6:
        raise ValueError("compact figure fixture must contain exactly six frozen cases")
    if any(sample_id not in selection for sample_id in ids):
        raise ValueError("fixture contains a case absent from the frozen selection receipt")
    for sample_id, role, z_index in zip(ids, roles, z_indices, strict=True):
        row = selection[sample_id]
        if row["role"] != role or int(row["selected_z_index"]) != z_index:
            raise ValueError(f"selection/provenance mismatch for {sample_id}")
    coords = np.asarray(data["coords"], dtype=np.float64)
    if coords.shape != (65 * 65 * 57, 3):
        raise ValueError("shared coordinate domain mismatch")
    figures = []
    for index, (sample_id, role, z_index) in enumerate(zip(ids, roles, z_indices, strict=True)):
        figures.append(render_case(output_dir=args.output, sample_id=sample_id, role=role, z_index=z_index, truth=data["truth"][index], heat3d=data["heat3d"][index], thermfm=data["thermfm"][index]))
    audit = {
        "schema_version": "heat3d_v7_g2_publication_figure_audit_v2",
        "fixture_sha256": sha256(args.fixture),
        "selection_receipt_sha256": sha256(args.selection_receipt),
        "protocol_sha256": sha256(args.protocol),
        "selection_status": receipt.get("status"),
        "visualization_only_test": True,
        "used_for_model_selection": False,
        "sealed_access": False,
        "prediction_layout": "canonical x_y_z flattened; reshape (65,65,57), exact z slice",
        "models": ["Heat3D V7 P1i-e200 U-v2", "Therm-FM"],
        "figures": figures,
        "thermfm_layout_check": {
            "source_formal_layout": "z_x_y",
            "display_layout": "x_y_z",
            "transpose_applied_once": True,
            "denormalization_in_plotter": False,
            "interpolation": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "publication_figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
