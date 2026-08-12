#!/usr/bin/env bash
set -euo pipefail

# Valid-only paired replay for U4.  This exports predictions that the historical
# aggregate-only P5-R artifacts did not retain; it does not replace their frozen
# accuracy or timing records.
ROOT=${HEAT3D_REPO_ROOT:-$HOME/myCodeGitOnly/heat3d-ic}
ARTIFACT_ROOT=${P1I_PREFLIGHT_ROOT:-/tmp/v6_p1i_gpu_only_high_n_valid32_b935713}
OUT=${P1I_U4_PAIRED_ROOT:-/tmp/v6_p1i_u4_paired_replay}
DATASET_ROOT=${P1I_DATASET_ROOT:-$ROOT/data/heat3d_v6_p1i_continuous_physics1024_v1}
FULL_FIELDS=${P1I_FULL_FIELDS:-$ROOT/data/heat3d_v6_p1i_continuous_physics1024_v1_full_fields/full_fields.h5}
RUN_DIR=${P1I_SEED0_RUN_DIR:-/tmp/p1i-seed0-timing-run}
CHECKPOINT_SHA=51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e
BINDING=$ROOT/configs/heat3d_v6_p1i/v6_p1i_high_n_implementation_binding.json
MANIFEST=$ROOT/configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_manifest.json
P5R_PROTOCOL=$ROOT/configs/heat3d_v6_p1i/v6_p1i_p5r_resolution_sweep_protocol.json
U4_PROTOCOL=$ROOT/configs/heat3d_v6_p1i/v6_p1i_u4_direct240825_protocol.json
RAW=$ROOT/configs/heat3d_v6_p1i/v6_p1i_p5r_raw

mkdir -p "$OUT"

for ROUTE in native1024_reconstruction E16384_reconstruction E240825_direct; do
  python "$ROOT/scripts/run_heat3d_v6_p1i_p5r_resolution_cell.py" \
    --protocol "$P5R_PROTOCOL" --binding "$BINDING" \
    --artifact-root "$ARTIFACT_ROOT" --dataset-root "$DATASET_ROOT" \
    --manifest "$MANIFEST" --full-fields "$FULL_FIELDS" --run-dir "$RUN_DIR" \
    --padding-result "$RAW/$ROUTE.json" \
    --native-padding-result "$RAW/native1024_reconstruction.json" \
    --output "$OUT/$ROUTE.json" --prediction-output "$OUT/$ROUTE.npz" \
    --route "$ROUTE" --checkpoint-sha256 "$CHECKPOINT_SHA" --sample-count 32
done

python "$ROOT/scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py" \
  --protocol "$U4_PROTOCOL" --binding "$BINDING" \
  --artifact-root "$ARTIFACT_ROOT" --dataset-root "$DATASET_ROOT" \
  --manifest "$MANIFEST" --full-fields "$FULL_FIELDS" --run-dir "$RUN_DIR" \
  --native-padding-result "$RAW/native1024_reconstruction.json" \
  --query-padding-result "$ROOT/configs/heat3d_v6_p1i/v6_p1i_u3_runtime_raw/u1_240825_valid32.json" \
  --output "$OUT/U_direct240825.json" --prediction-output "$OUT/U_direct240825.npz" \
  --resolution 240825 --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --sample-count 32 --repeats 1 --batch-sizes 1
