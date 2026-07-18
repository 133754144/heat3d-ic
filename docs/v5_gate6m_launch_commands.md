# Gate 6M launch commands

These commands are prepared only. Gate 6M closeout does not execute them.

## A — devbox

```bash
ssh devbox
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
cd ~/myCodeGitOnly/heat3d-ic
mkdir -p output/heat3d_v5_gate6m_a_logs
nohup python scripts/run_heat3d_v4_config.py \
  --config configs/heat3d_v5/generated/V4P5_35_gate6m_v32_scale_head_only_e100.yaml \
  > output/heat3d_v5_gate6m_a_logs/V4P5_35_gate6m_v32_scale_head_only_e100.log 2>&1 &
echo $!
```

Monitor:

```bash
tail -f output/heat3d_v5_gate6m_a_logs/V4P5_35_gate6m_v32_scale_head_only_e100.log
```

## B — WSL2

```bash
ssh wsl2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
cd ~/myCodeGitOnly/heat3d-ic
mkdir -p output/heat3d_v5_gate6m_b_logs
nohup python scripts/run_heat3d_v4_config.py \
  --config configs/heat3d_v5/generated/V4P5_36_gate6m_v32_epoch_regroup_e200.yaml \
  > output/heat3d_v5_gate6m_b_logs/V4P5_36_gate6m_v32_epoch_regroup_e200.log 2>&1 &
echo $!
```

Monitor:

```bash
tail -f output/heat3d_v5_gate6m_b_logs/V4P5_36_gate6m_v32_epoch_regroup_e200.log
```

Both configs are `explicit_user_instruction_only`; neither command was run
during Gate 6M preparation.
