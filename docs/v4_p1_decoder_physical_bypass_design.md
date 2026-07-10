# V4 P1 Decoder Physical Bypass

## Scope

`decoder_bypass_mode=post_decoder_residual` is an opt-in model experiment. The
default remains `decoder_bypass_mode=none`, which creates no bypass parameters
and preserves the existing RIGNO path.

This bypass does not change coordinate normalization, graph topology, solver,
loss, dataset generation, target normalization, or raw DeltaT/T recovery.

## Mechanism

- Base output remains the current decoder output in normalized DeltaT space.
- Bypass features come from `inputs.c` after existing condition normalization.
- `decoder_bypass_features=full_condition` selects condition columns by
  `feature_names`, not by hard-coded column numbers.
- Required full-condition features are `k_x`, `k_y`, `k_z`, `q`, BC flags,
  `top_h`, and relative BC temperature scalars.
- The bypass MLP produces a normalized-DeltaT residual:
  `output = base_output + decoder_bypass_residual_scale * bypass_residual`.
- `decoder_bypass_init=zero_residual` initializes the final bypass layer to
  zero, so the initial enabled model output equals the base decoder output.

## Provenance

Runs should record:

- `decoder_bypass_mode`
- `decoder_bypass_features`
- `decoder_bypass_feature_source`
- `decoder_bypass_feature_names`
- `decoder_bypass_feature_indices`
- `decoder_bypass_num_features`
- `decoder_bypass_output_space`
- `decoder_bypass_hidden_size`
- `decoder_bypass_layers`
- `decoder_bypass_init`
- `decoder_bypass_residual_scale`

The first registered experiment is
`V4P1_05_decoder_bypass_full_condition`.
