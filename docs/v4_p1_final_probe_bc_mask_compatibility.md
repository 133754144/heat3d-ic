# V4 P1 Final-Probe BC Mask Compatibility

Read this file only for final-probe BC mask compatibility, final-probe
before/after evaluation, or V4 P1 model-lab merge review.

## Scope

This is an evaluation compatibility fix, not a model improvement. It does not
change the training loader, model, solver, loss, registry CSV, training target,
or normalization path.

Remote evaluation ran on WSL2 with the existing S4 best checkpoint:

`output/heat3d_v2_runs/latent96_s6_mlp3_B88_sample_shuffle_discrete_radius_S4mlp3discretebestFT2_e400_constant_lr5e-6_wd1e-4/params_best.pkl`

No training, tmux, prediction artifacts, figures, data, checkpoints, or logs
were written. Only small metrics/metadata/report files were written under:

- `output/heat3d_v4_p1_bc_mask_compatibility/legacy/`
- `output/heat3d_v4_p1_bc_mask_compatibility/reconstructed/`

## Source Audit

- Final-probe metadata has `boundary_regions` entries with surface labels
  (`z_max`, `z_min`, `x_or_y_minmax`) but no `point_indices`.
- Medium1024 train/valid metadata has no `boundary_regions`; with
  `boundary_mask_fallback=True`, the existing loader reconstructs masks from
  coordinate extrema.
- Final-probe can be reconstructed the same way: top is `z=max`, bottom is
  `z=min`, side is `x/y` min or max, and interior is non-top/non-bottom/non-side.
- The fix belongs in the final-probe eval/adapter compatibility layer. Moving
  it into the dataset metadata loader would silently change broader loader
  semantics and is not needed for this P1.0b question.
- The compatibility layer records `mask_source=metadata` or
  `mask_source=coords_extrema_reconstructed`. Legacy unresolved metadata is
  recorded as `metadata_missing_indices_unresolved`.

## Before/After

Final-probe samples use a `16 x 16 x 4` coordinate grid. The reconstructed mask
fractions are therefore geometry-consistent but not identical to medium1024
train/valid fractions, which use a different z resolution.

| policy | mask source | top | bottom | side | interior |
| --- | --- | ---: | ---: | ---: | ---: |
| legacy | `metadata_missing_indices_unresolved` x10 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| reconstructed | `coords_extrema_reconstructed` x10 | 0.250000 | 0.250000 | 0.234375 | 0.382812 |

Mean metrics over the same 10 final-probe samples:

| policy | RMSE_K | relRMSE_DeltaT | peak_error_K | mean_bias_K | scale_ratio | range_ratio | centered_corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 0.360902 | 0.768656 | -3.147039 | -0.130546 | 0.218191 | 0.218089 | 0.858803 |
| reconstructed | 0.348347 | 0.731647 | -3.125000 | -0.095895 | 0.228005 | 0.227904 | 0.822217 |

Worst probes by RMSE remain P03, P02, and P09 in both policies. The
reconstructed masks improve RMSE and relRMSE slightly, but the peak amplitude
is still only about 23% of the label DeltaT amplitude.

## Conclusion

The old all-interior final-probe BC flags were caused by metadata compatibility,
not by final-probe geometry. Reconstructing masks from coordinate extrema fixes
the BC flag inputs for final-probe evaluation and gives a reasonable
top/bottom/side/interior distribution.

The final-probe failure remains primarily amplitude failure after the mask fix:
`scale_ratio` and `range_ratio` stay near 0.2, while `centered_corr` remains
high enough to show partial shape preservation. Therefore the V3/V4 final-probe
underprediction is not explained solely by missing BC masks.
