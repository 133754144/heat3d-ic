#!/usr/bin/env python3
"""Synthetic smoke checks for coordinate-derived boundary mask fallback."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.dataset_Heat3D_v1 import Heat3DV1MetadataDataset


def main() -> int:
    coords = np.asarray(list(product((0.0, 0.5, 1.0), repeat=3)), dtype=np.float64)
    fallback = Heat3DV1MetadataDataset._encode_boundary_conditions(
        coords,
        {},
        boundary_mask_fallback=True,
    )
    legacy = Heat3DV1MetadataDataset._encode_boundary_conditions(
        coords,
        {},
        boundary_mask_fallback=False,
    )

    if [int(np.sum(fallback[:, index])) for index in range(4)] != [9, 9, 24, 1]:
        raise AssertionError("coordinate fallback did not reconstruct expected top/bottom/side/interior masks")
    if not np.all(legacy[:, :3] == 0.0) or not np.all(legacy[:, 3] == 1.0):
        raise AssertionError("disabled fallback must preserve legacy all-interior masks")

    explicit_meta = {
        "boundary_regions": [
            {"name": "top", "point_indices": [0, 1]},
            {"name": "bottom", "point_indices": [2]},
            {"name": "sides", "point_indices": [3, 4]},
        ]
    }
    explicit_enabled = Heat3DV1MetadataDataset._encode_boundary_conditions(
        coords,
        explicit_meta,
        boundary_mask_fallback=True,
    )
    explicit_disabled = Heat3DV1MetadataDataset._encode_boundary_conditions(
        coords,
        explicit_meta,
        boundary_mask_fallback=False,
    )
    if not np.array_equal(explicit_enabled, explicit_disabled):
        raise AssertionError("fallback must not change the explicit boundary_regions path")

    print("Heat3D v2 boundary mask fallback smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
