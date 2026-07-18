# Gate 6N P5 graph degree audit

- Baseline: `V4P5_36_gate6m_v32_epoch_regroup_e600`.
- Scope: train 672 only, 1024 physical nodes, graph seed 0.
- All 672 samples share one normalized-coordinate topology with 256 regional
  nodes.
- Full-graph r2r in-degree is min/median/P95/max = `5/13/25/41`; p2r and r2p
  full graphs also have zero zero-degree receivers.
- Upstream-style shuffle-and-prefix simulation used 128 fixed seeds per rate.
  Rates 0.05, 0.10 and 0.20 produced zero zero-degree regional nodes. Rate
  0.50 produced 39 zero-degree events and a maximum of 2 zero-degree nodes in
  one mask.
- Every tested rate reproduced the identical mask for the same seed and
  changed the mask for distinct seeds.

Gate 6N freezes `p=0.10`, r2r-only. It is deliberately more conservative than
the upstream all-edge default and retains 3302 of 3669 r2r edges, including
the dummy edge in the upstream edge-count convention.
