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

Probe result: pending devbox run.

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

legacy vs nearest_repair 1/4/16-sample Adam lr=1e-3 constant e1000 results:
pending devbox run.
