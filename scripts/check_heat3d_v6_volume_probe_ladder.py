#!/usr/bin/env python3
"""Deterministically check the V6 volume-representative probe ladder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from prepare_heat3d_v6_volume_probe_ladder import (  # noqa: E402
    DEFAULT_ACCEPTANCE,
    DEFAULT_LADDER,
    DEFAULT_MANIFEST,
    DEFAULT_PROBE4096,
    RESOLUTIONS,
    _array_sha256,
    build,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--probe4096", type=Path, default=DEFAULT_PROBE4096)
    args = parser.parse_args()

    frozen = _read_json(args.ladder)
    rebuilt = build(args.acceptance.resolve(), args.manifest.resolve())
    _assert(frozen == rebuilt, "frozen ladder is not deterministic")
    _assert(
        frozen["resolutions"] == list(RESOLUTIONS),
        "resolution ladder drifted",
    )
    _assert(
        frozen["supports_are_nested"]
        and frozen["label_independent"]
        and frozen["source_dense_quota_fraction"] == 0.0,
        "selection contract drifted",
    )
    _assert(
        frozen["evaluation_role"] == "valid_iid"
        and not frozen["test_hard_accessed"]
        and not frozen["training_executed"]
        and not frozen["formal_inference_executed"],
        "role or execution guard failed",
    )

    prior: set[int] = set()
    summary: dict[str, Any] = {}
    old_probe = _read_json(
        ROOT / "configs/heat3d_v6/v6_valid_common_probe4096.json"
    )
    for resolution in RESOLUTIONS:
        probe = frozen["probes"][str(resolution)]
        indices = np.asarray(probe["indices"], dtype=np.int32)
        inclusion = np.asarray(
            probe["inclusion_probabilities"], dtype=np.float64
        )
        expansion = np.asarray(
            probe["expansion_weights_m3"], dtype=np.float64
        )
        _assert(
            indices.shape == inclusion.shape == expansion.shape == (resolution,),
            f"{resolution}: probe array shapes drifted",
        )
        _assert(
            len(np.unique(indices)) == resolution,
            f"{resolution}: indices are not unique",
        )
        current = set(indices.tolist())
        _assert(not prior or prior < current, f"{resolution}: support is not nested")
        prior = current
        _assert(
            _array_sha256(indices) == probe["support_index_sha256"],
            f"{resolution}: support hash drifted",
        )
        _assert(
            _array_sha256(inclusion)
            == probe["inclusion_probability_sha256"],
            f"{resolution}: inclusion hash drifted",
        )
        _assert(
            _array_sha256(expansion) == probe["expansion_weight_sha256"],
            f"{resolution}: expansion hash drifted",
        )
        total = float(probe["total_control_volume_m3"])
        _assert(
            np.allclose(
                expansion,
                total / resolution,
                rtol=1.0e-13,
                atol=0.0,
            ),
            f"{resolution}: Horvitz-Thompson weights drifted",
        )
        _assert(
            math.isclose(float(np.sum(expansion)), total, rel_tol=1.0e-13),
            f"{resolution}: expansion weights do not recover total volume",
        )
        coverage = probe["coverage"]
        _assert(
            coverage["all_layers_covered"]
            and coverage["all_interfaces_covered"]
            and coverage["top_point_count"] > 0
            and coverage["bottom_point_count"] > 0,
            f"{resolution}: required physical coverage failed",
        )
        _assert(
            probe["label_independent"]
            and probe["source_dense_quota_fraction"] == 0.0
            and not probe["test_hard_accessed"]
            and not probe["training_executed"]
            and not probe["inference_executed"],
            f"{resolution}: leakage/execution contract failed",
        )
        selected_text = " ".join(probe["selection_inputs"]).lower()
        _assert(
            "temperature" not in selected_text
            and "source metadata" not in selected_text
            and "prediction" not in selected_text,
            f"{resolution}: label-derived selection input",
        )
        summary[str(resolution)] = {
            "support_index_sha256": probe["support_index_sha256"],
            "top_point_count": coverage["top_point_count"],
            "bottom_point_count": coverage["bottom_point_count"],
            "max_layer_volume_fraction_error": coverage[
                "max_layer_volume_fraction_error"
            ],
        }

    probe4096 = _read_json(args.probe4096)
    expected4096 = dict(frozen["probes"]["4096"])
    expected4096["ladder_parent"] = str(DEFAULT_LADDER.relative_to(ROOT))
    _assert(probe4096 == expected4096, "standalone 4096 probe drifted")
    _assert(
        probe4096["support_index_sha256"] != old_probe["support_index_sha256"],
        "volume probe unexpectedly reuses the old source-dense support",
    )

    print(
        json.dumps(
            {
                "status": "passed",
                "selection_policy": frozen["selection_policy"],
                "supports_are_nested": True,
                "source_dense_quota_fraction": 0.0,
                "resolutions": summary,
                "test_hard_accessed": False,
                "formal_inference_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
