# heat3d_v6_p1i_continuous_physics1024_v1

This directory is the immutable external archive of the frozen V6-P1i
`formal1024_v1` dataset.

## Identity

- Dataset ID: `heat3d_v6_p1i_continuous_physics1024_v1`
- Samples: `1024`
- Split counts: train `768`, valid_iid `128`, test_iid `128`
- Projected nodes per sample: `1024`
- Formal config SHA256:
  `1e15a77fe51eea7ec64614566bb6bb12bfcf05948f3b7c8c6f3c85ec759a58f8`
- Formal manifest SHA256:
  `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514`
- Formal manifest payload SHA256:
  `27d2ea3b7ec4e4ce9c6d068471cd19036ac8148b6cd57da325219d718c7e5ed5`

The external revision and tag are recorded in the source repository's
`v6_p1i_formal1024_v1_archive_manifest.json`. Existing Hugging Face subsets
were not overwritten.

## Scope

The training-preflight audit supports **continuous broad coverage** over the
frozen observed response ranges. It does not claim strict uniformity.
Package power is deliberately coupled to top-side Robin `h`; high-power /
low-top-h coverage is absent under the preregistered corner definition.

P1i uses perfect interface contact everywhere:
`R_contact=0 m²K/W`. The dataset therefore cannot identify finite
contact-resistance effects.

## Files

- `samples/<sample_id>/*.npy` and `sample_meta.json`: frozen sample payloads;
- `dataset_manifest.json`: path, split and per-file SHA256 bindings;
- `training_preflight_audit.json`: zero-modification distribution, joint
  coverage, split and applicability audit.

No thermal solver, learned-model training or model inference is part of this
archive operation.
