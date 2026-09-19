#!/usr/bin/env python3
"""Render fixed valid-only V7 G2 figures from a compact frozen fixture.

Full-field rows use exact 65x65 slices. Native rows are explicitly labelled
``interpolated diagnostic`` and are not treated as benchmark-equivalent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/heat3d-v7-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dense_grid(coords: np.ndarray, values: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sel = np.isclose(coords[:, 2], z, atol=1e-12, rtol=0.0)
    xs = np.unique(coords[sel, 0])
    ys = np.unique(coords[sel, 1])
    ix = np.searchsorted(xs, coords[sel, 0])
    iy = np.searchsorted(ys, coords[sel, 1])
    out = np.empty((len(ys), len(xs)), dtype=np.float64)
    out[iy, ix] = values[sel]
    return out, xs, ys


def save_triptych(out: Path, name: str, sample: str, z: float, truth: np.ndarray, pred_map: dict[str, np.ndarray], coords: np.ndarray, labels: dict[str, str], dense: bool) -> dict[str, object]:
    truth_img, xs, ys = dense_grid(coords, truth, z)
    imgs = {model: dense_grid(coords, pred, z)[0] for model, pred in pred_map.items()}
    errs = {model: imgs[model] - truth_img for model in imgs}
    lo = min(float(truth_img.min()), *(float(v.min()) for v in imgs.values()))
    hi = max(float(truth_img.max()), *(float(v.max()) for v in imgs.values()))
    err_lim = max(float(np.max(np.abs(v))) for v in errs.values())
    norm = Normalize(lo, hi)
    en = Normalize(-err_lim, err_lim)
    tcmap = plt.get_cmap("turbo").copy()
    ecmap = plt.get_cmap("RdBu_r").copy()
    fig = plt.figure(figsize=(13, 3.4 * len(imgs) + 2.0), facecolor="white")
    gs = fig.add_gridspec(len(imgs), 3, left=0.12, right=0.96, top=0.88, bottom=0.12, hspace=0.58, wspace=0.36)
    reference_hashes = []
    for i, (model, img) in enumerate(imgs.items()):
        err = errs[model]
        for j, arr in enumerate((truth_img, img, err)):
            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(arr, origin="lower", aspect="equal", interpolation="nearest", cmap=tcmap if j < 2 else ecmap, norm=norm if j < 2 else en)
            cb = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.035)
            cb.set_label("K", fontsize=8)
            ax.set_xlabel("x grid index", fontsize=8)
            ax.set_ylabel("y grid index", fontsize=8)
            if i == 0:
                ax.set_title(("Reference ΔT (K)", "Prediction ΔT (K)", "Prediction − reference (K)")[j], fontsize=11)
            if j == 0:
                reference_hashes.append(hashlib.sha256(np.asarray(im.to_rgba(im.get_array())).tobytes()).hexdigest())
                ax.text(-0.40, 0.5, labels[model], transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
    fig.suptitle(f"V7 G2 · {name} · {sample}", fontsize=16, fontweight="bold", y=0.97)
    fig.text(0.5, 0.925, f"valid_iid · exact dense 65×65 slice at z={z:.9g} m · common temperature/error scales", ha="center", fontsize=9)
    fig.text(0.12, 0.035, "Full-field models only: exact dense slice; native models are not included in this figure.", fontsize=8.5, color="#44515d")
    fig.savefig(out / f"{name}.png", dpi=220)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    return {
        "figure": name,
        "sample_id": sample,
        "mode": "exact_dense_slice" if dense else "interpolated_diagnostic",
        "temperature_limits": [lo, hi],
        "error_limits": [-err_lim, err_lim],
        "reference_rgba_sha256_by_row": reference_hashes,
        "reference_rgba_identical": len(set(reference_hashes)) == 1,
        "models": list(imgs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--fixture-meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = np.load(args.fixture, allow_pickle=False)
    meta = json.loads(args.fixture_meta.read_text(encoding="utf-8"))
    coords = np.asarray(d["coords"], dtype=np.float64)
    truth = np.asarray(d["truth"], dtype=np.float64)
    z = float(coords[int(np.argmax(truth)), 2])
    audit = {
        "schema_version": "heat3d_v7_g2_plotting_audit_v1",
        "sample_id": meta["sample_id"],
        "split_role": meta["split_role"],
        "fixture_sha256": sha256(args.fixture),
        "fixture_meta_sha256": sha256(args.fixture_meta),
        "source_audit": meta,
        "figures": [],
        "test_iid": {"status": "BLOCKED_NO_FROZEN_TEST_PREDICTION_ARTIFACT", "read": False, "reason": "No authorized fixed test visualization prediction archive was present; no fresh inference was run."},
    }
    dense_preds = {"Heat3D U-v2": np.asarray(d["heat3d_u_v2"], dtype=np.float64), "Therm-FM": np.asarray(d["thermfm"], dtype=np.float64)}
    audit["figures"].append(save_triptych(args.out, "valid_fullfield_exact_dense", meta["sample_id"], z, truth, dense_preds, coords, {"Heat3D U-v2": "Heat3D U-v2 (exact dense)", "Therm-FM": "Therm-FM (exact dense)"}, True))

    native_preds = {}
    for key, label in (("heat3d_native", "Heat3D native1024"), ("gino_native", "GINO native1024"), ("transolver_native", "Transolver native1024")):
        if key in d:
            native_preds[label] = np.asarray(d[key], dtype=np.float64)
    # Interpolate native same-layer points only for a diagnostic picture.
    native_coords = np.asarray(d["native_coords"], dtype=np.float64)
    native_layer = np.asarray(d["native_layer_id"], dtype=np.int32)
    layer = int(np.asarray(d["layer_id"])[int(np.argmax(truth))])
    support = native_layer == layer
    origin = coords.min(axis=0)
    scale = np.ptp(coords, axis=0).max()
    query_img, xs, ys = dense_grid(coords, truth, z)
    xx, yy = np.meshgrid(xs, ys)
    query = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
    p = (native_coords[support] - origin) / scale
    q = (query - origin) / scale
    outside = float(np.mean(Delaunay(p).find_simplex(q) < 0))
    interp = {}
    dense_slice_mask = np.isclose(coords[:, 2], z, atol=1e-12, rtol=0.0)
    for label, values in native_preds.items():
        slice_values = RBFInterpolator(p, values[support], kernel="thin_plate_spline", smoothing=0)(q).reshape(-1)
        full_values = np.full(coords.shape[0], np.nan, dtype=np.float64)
        full_values[dense_slice_mask] = slice_values
        interp[label] = full_values
    audit["native_interpolated_display"] = {"status": "DIAGNOSTIC_ONLY", "outside_support_fraction": outside, "label": "interpolated diagnostic; not benchmark-equivalent"}
    audit["figures"].append(save_triptych(args.out, "valid_native1024_interpolated_diagnostic", meta["sample_id"], z, truth, interp, coords, {k: k for k in interp}, False))

    # Native point cloud figure is kept separate from the interpolated image.
    fig, axes = plt.subplots(1, len(native_preds), figsize=(5 * len(native_preds), 4), squeeze=False)
    for ax, (label, values) in zip(axes[0], native_preds.items()):
        sc = ax.scatter(native_coords[support, 0] * 1000, native_coords[support, 1] * 1000, c=values[support], s=9, cmap="turbo")
        ax.set_title(label + "\nnative-point scatter")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); fig.colorbar(sc, ax=ax, label="ΔT (K)")
    fig.suptitle(f"V7 G2 · {meta['sample_id']} · native1024 diagnostic", fontsize=14)
    fig.tight_layout(); fig.savefig(args.out / "valid_native1024_scatter.png", dpi=220); fig.savefig(args.out / "valid_native1024_scatter.pdf"); plt.close(fig)
    audit["figures"].append({"figure": "valid_native1024_scatter", "mode": "native_point_scatter", "models": list(native_preds)})
    (args.out / "plotting_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
