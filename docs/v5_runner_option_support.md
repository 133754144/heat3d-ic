# V5 Runner Option Support

`configs/heat3d_v5/v5_clean_first_bypass_ablation_plan.yaml` is now connected
to a guarded V5 config adapter rather than being a documentation-only YAML.

Use a no-write command resolution such as:

```bash
python3 scripts/run_heat3d_v5_config.py \
  --config configs/heat3d_v5/v5_clean_first_bypass_ablation_plan.yaml \
  --variant local_only_bypass_control --dry-run
```

The adapter resolves the inherited V4P5_02 configuration and maps one
`ablation_matrix` entry to the real V4 controlled runner. It supports:

- `decoder_bypass_features: explicit_local_condition` with the audited,
  node-varying `k_x,k_y,k_z,q,is_top,is_bottom,is_side,is_interior` list;
- `global_context_mode: none|film`, with the fixed inference-only V5 physics
  schema, train-only standardizer, identity FiLM and batch forwarding;
- strict rejection of sample-global BC/extent broadcasts as local bypass
  inputs.

All current Task-2 variants keep `training_allowed: false`. `--execute` is
therefore deliberately rejected. This prevents the prepare-only ablations from
turning into a long run before the V5 smoke and warm-start gates pass.

## Controlled V5 warm-start

The executable short feasibility configuration is separate from the
prepare-only ablation plan. It runs only the configured 12-epoch clean-only
warm-start and is not a scratch or V4-scale long run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
cd ~/myCodeGitOnly/heat3d-ic
python3 -u scripts/run_heat3d_v5_clean_first.py \
  --config configs/heat3d_v5/v5_clean_warmstart_short.yaml --execute
```

When it is started in tmux with output redirected to
`/tmp/heat3d_v5_clean_ws_*.log`, follow its flushed JSON progress lines with:

```bash
tail -F /tmp/heat3d_v5_clean_ws_*.log
```

The runner reports preflight, frozen-baseline evaluation, each variant start
and initialization, every completed epoch, and each direction-gate outcome.
Those logs are observational only; test and hard roles remain report-only.

Verify this executable configuration without training or output writes:

```bash
python3 -B scripts/check_heat3d_v5_clean_warmstart_config.py
```

For a FiLM variant, the legacy V4 final-probe helper is disabled in the command
plan because it cannot reconstruct V5 global context; the dedicated V5 path
will own that evaluation after its smoke is complete.

## Gate-5 native scratch runner

Registry-generated N0/N1 configs use the controlled runner with these mapped
options: `native_output_mode`, `native_branch_mode`, `scale_head_mode`,
`scale_pooling`, `scale_head_hidden_size`, `decoder_bypass_output_space` and
the four native loss weights. Native mode also makes the runner attach
inference-only control volumes, physics scale, raw reference temperature and
Dirichlet data to each batch even when Global FiLM is off.

The runner selects joint best by reconstructed normalized-DeltaT
`valid_base_mse`, exports projected raw-temperature predictions, emits the four
native loss components and oracle/physics-scale diagnostics, and masks both
gradients and optimizer updates for `scale_only` or `shape_only`. See
`docs/v5_gate5_native_shape_scale.md` for the frozen semantics and N0/N1
comparison contract.
