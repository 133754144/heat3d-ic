# V6 decoder-only historical audit

## Decision

The repository does not contain a verified decoder-only high-resolution protocol. The conditional random-block experiment is therefore fail-closed and was not executed. No training, inference, test, hard, or sealed access occurred during this audit.

A qualifying decoder-only route would keep the original 1024-point `x_in`, p2r graph, encoder output, r2r graph, processor output and regional latent fixed; only `x_out=N`, r2p and the decoder would change. No current or historical implementation satisfies that definition.

## Route classification

| Route | Actual execution | Decoder-only |
|---|---|---|
| Anchored full-N | 1024 anchor full-model forward, then N-node full-model forward, followed by anchor-scale reconstruction | No |
| 1024 + reconstruction | Full-model inference at 1024, then a layer-aware map to the target field | No |
| Controlled Direct-N | `x_in=x_out=N`; complete graph and complete model are rebuilt | No |
| Generic RIGNO metadata API | Accepts separate `x_in/x_out`, but has no verified Heat3D adapter or checkpoint replay | Not established |

The anchored production configuration explicitly records `1024_anchor_forward -> anchor context/scale -> N_node_source_aware_forward -> anchor_scale_reconstruction`. The implementation calls `_predict_groups` on the N-node query group, which invokes the complete model rather than only its decoder.

## Interface and shape blockers

- `Heat3DGraphBuilder.build_metadata` accepts one coordinate array and passes it as both `x_inp` and `x_out`.
- The V6 bridge and current runner create identical `x_inp/x_out` tensors.
- The frozen checkpoint includes node-local decoder bypass features (`k_x/k_y/k_z/q` and four BC flags). Both runner and model require one-to-one input/output node count and coordinate ordering.
- The decoder receives `latent_pnodes` produced by the encoder. There is no frozen serialization contract for cached regional/physical latent state and no r2p-only graph-building API.
- There is no asymmetric-node checkpoint replay, forward-equivalence checker, configuration, or tracked result.

The lower-level generic RIGNO graph metadata method accepting distinct `x_inp/x_out` is only a possible implementation primitive. It is not evidence that the frozen Heat3D checkpoint supports decoder-only inference.

## Random-block conditional gate

The frozen random-block dataset exists on WSL2 and its hashes were verified, but the random-block repository contains zero dedicated checkpoint/run-config/loss-summary artifacts. The only available checkpoint is the P1h-trained `V6_03_V5best_P1h` checkpoint, previously admitted only through an explicit full-1024 adapter as a runtime-only OOD diagnostic.

Consequently, no random-block decoder-only accuracy, timing, FVM comparison, or speedup result is produced. Existing legal random-block resolution diagnostics retain their prior runtime-only OOD status and cannot support decoder-only or production acceleration claims.

Machine-readable evidence is recorded in `configs/heat3d_v6_p1i/v6_p1i_decoder_only_history_audit.json` and its CSV mirror.
