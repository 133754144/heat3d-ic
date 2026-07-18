# Gate 6N launch commands

The formal e600 run is not started by this closeout. The revised exact
Processor-key audit retained p=0.05, so this command still points to V38 and
does not require an e3 rerun. After confirming the WSL2 checkout is clean and
no existing run would be disturbed:

```bash
ssh wsl2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
cd ~/myCodeGitOnly/heat3d-ic
git fetch origin
git switch research/v5
git pull --ff-only origin research/v5
python scripts/check_heat3d_v5_gate6n.py --preflight-only
mkdir -p output/heat3d_v5_gate6n_logs
python scripts/run_heat3d_v4_config.py \
  --config configs/heat3d_v5/generated/V4P5_38_gate6n_v36_r2r_mask_p005_e600.yaml \
  2>&1 | tee output/heat3d_v5_gate6n_logs/V4P5_38_gate6n_v36_r2r_mask_p005_e600.log
```

Monitor from another WSL2 shell:

```bash
tail -f ~/myCodeGitOnly/heat3d-ic/output/heat3d_v5_gate6n_logs/V4P5_38_gate6n_v36_r2r_mask_p005_e600.log
```
