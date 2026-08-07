# P1i 1024-anchor cross-resolution inference protocol

Status: frozen before implementation and execution. This round runs no
large-scale inference.

## Scientific boundary

The protocol transfers the frozen P1h production idea to P1i while respecting
P1i's sample-varying source-aware supports. It is not the existing full-graph
re-discretization diagnostic, structured Direct-N compatibility test, or a
decoder-only method.

For every sample and seed:

1. Load the exact original 1024 support, pointwise k/q/CV weights and frozen
   train-only normalization.
2. Compute Global Context, regional QK inputs, physics scale and predicted
   scale from those anchors only.
3. Run the exact 1024 anchor forward and retain its scale.
4. Build a target-independent nested N-node query support containing all
   anchors, using source/layer/interface/Robin-surface coverage plus volume
   PPS on the 240825-node solver mesh.
5. Run the full-N query graph for shape, then reconstruct with the frozen
   anchor scale.
6. Reconstruct to the common 240825-node field using the frozen
   layer/interface-aware map and report the oracle sampling floor separately.

No old 1024 prediction is interpolated to form full-N pointwise k/q inputs.
Temperature and test labels cannot influence support, graph, context, cache or
resolution decisions.

## Frozen scope

- Dataset role: `formal_v6_randomblock` =
  `heat3d_v6_p1i_continuous_physics1024_v1`.
- Checkpoints: V6_06 e559, V6_07 e455 and V6_08 e587 point-global best.
- Mandatory N: 1024, 4096, 8192, 16384.
- Optional N: 32768. The prerequisite 1024-to-full-field reconstruction is
  present, but 32768 is not required and cannot enter the mandatory ranking.
- Graph backend: `sparse_kdtree_v1`; cache key includes support hash, resolved
  graph config, graph seed and graph-builder code fingerprint.
- Development role: `valid_iid` only. Test and sealed remain closed.

## Gates

R0 must first reproduce all three frozen 1024 checkpoints and full-field
metrics within the registered tolerance. Seed0 then runs a bounded 4096
development cell. Only after graph/cache, context/scale, finite-forward,
reconstruction-map and role-access checks pass may all three seeds run the
mandatory valid-only ladder. This protocol does not authorize those executions
in the current round.

The complete machine-readable contract is
`configs/heat3d_v6_p1i/v6_p1i_anchor_query_resolution_protocol.json`.
