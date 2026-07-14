# V5 Gate 6 N3 checkpoint sync

N3 best e402 was copied from WSL2 into a new ignored input package and then
copied non-destructively to devbox. The Gate 5 run directory was not modified.

- source: `wsl2:output/heat3d_v5_runs/V4P5_07_native_pooled_latent_global_film`
- identical relative input on both hosts:
  `output/heat3d_v5_gate6_inputs/N3_best_e402`
- content files: checkpoint, run config, loss summary, and extracted
  train-only normalization/context metadata
- checkpoint SHA256:
  `3baebb9b751bf6054f36308444cdefe7a7b4f343665164b0aabdfe2610b5a228`
- both hosts: `sha256sum -c SHA256SUMS` passed for all four content files

The machine-readable hashes and copy policy are frozen in
`configs/heat3d_v5/v5_gate6_n3_checkpoint_sync.json`.
