# Heat3D v3 B96 Nearest Repair Readiness

## Purpose

Prepare a manual B96 nearest_repair A/B candidate without running the long
e400 job in this task. The only intended training-path change is graph policy:
`coverage_repair_policy=nearest_rnode` with legacy radius policy.

## Config Support

The controlled runner now accepts:

- `--radius-policy`
- `--coverage-repair-policy`
- `--repair-p2r` / `--no-repair-p2r`
- `--repair-r2p` / `--no-repair-r2p`
- `--min-physical-coverage`

Defaults preserve legacy behavior:

- `radius_policy=legacy_kdtree_mean4`
- `coverage_repair_policy=none`
- `repair_p2r=true`
- `repair_r2p=true`
- `min_physical_coverage=1`

The v2 YAML command builder maps optional `graph:` fields to these CLI flags.

## Probe Config

Prepared and intended to run on SSH devbox:

`configs/heat3d_v2/frozen_v1_e002_adamw_m2width_B96_base_mse_warmup_cosine_nearest_repair_probe_seed0.yaml`

Scope: two-epoch OOM/compatibility probe only. No benchmark claim.

Probe result: passed on SSH devbox.

Key probe fields:

- `status_ok=true`
- `grad_finite=true`
- `best_epoch=2`
- `best_valid_loss=1.0683878660202026`
- `final_valid_loss=1.0683878660202026`
- `valid_iid raw_deltaT_relative_rmse_pct=159.77`
- `valid_stress raw_deltaT_relative_rmse_pct=146.13`
- `graph_config={"radius_policy": "legacy_kdtree_mean4", "coverage_repair_policy": "nearest_rnode", "repair_p2r": true, "repair_r2p": true, "min_physical_coverage": 1}`

The probe completed graph/group build, two training epochs, final/best prediction
export, and `run_config.json` / `loss_summary.json` writing without OOM.

## e400 Config

Prepared but not run:

`configs/heat3d_v2/frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_nearest_repair_seed0.yaml`

Manual command after probe passes:

```bash
python3 - <<'PY'
import subprocess
from rigno.heat3d_v2_config import load_v2_config
from rigno.heat3d_v2_runner_command import build_training_command

config_path = "configs/heat3d_v2/frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_nearest_repair_seed0.yaml"
command = build_training_command(load_v2_config(config_path), python_executable="python3")
print(" ".join(command))
subprocess.run(command, check=True)
PY
```

## P2 Follow-Up

legacy vs nearest_repair 1/4/16-sample Adam lr=1e-3 constant e1000:

| sample_count | legacy best relative RMSE | nearest_repair best relative RMSE | nearest better |
| ---: | ---: | ---: | --- |
| 1 | 37.04% | 37.84% | no |
| 4 | 50.49% | 58.48% | no |
| 16 | 54.78% | 55.30% | no |

Optional B96 AdamW warmup-cosine e1000:

| sample_count | legacy best relative RMSE | nearest_repair best relative RMSE | nearest better |
| ---: | ---: | ---: | --- |
| 1 | 19.68% | 19.62% | slight |
| 4 | 55.39% | 17.76% | yes |
| 16 | 60.81% | 59.71% | slight |

Interpretation: nearest_repair is not optimizer-agnostically stable, because
Adam lr=1e-3 constant makes it slightly worse than legacy. Under the B96 AdamW
schedule, it is the better B96 A/B candidate and passes the 2-epoch full
medium1024 compatibility probe. It should be treated as a diagnostic graph A/B,
not as a confirmed complete fix for the 16-sample bottleneck.
