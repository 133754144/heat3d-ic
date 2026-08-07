# P1i high-N implementation binding

Status: **frozen_after_three_seed_r0_pass** (released only after all three 1024 R0 gates passed).

## Frozen resolution and selection

- mandatory prefixes: 1024, 4096, 8192, 16384; optional 32768 is valid-only and excluded from mandatory ranking.
- every sample keeps its exact ordered 1024 anchors; added nodes are one deterministic solver-index sequence whose prefixes define all resolutions.
- node selection uses no temperature, target, model prediction, or error.

## Field and measure binding

- coords/control-volume/layer come directly from the frozen full-field sidecar.
- this sidecar does not persist full k/q; k/q therefore fail closed to deterministic reconstruction from frozen sample metadata and the fingerprinted continuous-field implementation, with power error <=1e-12.
- selected k/q are direct values at solver indices; effective operator CV is the conservative same-layer nearest-node partition of all solver control volumes.

## Development subset and timing

- fixed valid_iid subset: 32 IDs selected by SHA256(sample_id), independent of labels and model error.
- R0 validation cost includes duplicate reference/adapter graph and forward work plus array/reconstruction audits; it is not production timing.
- no 4096/8192/16384/32768 inference was executed in this closeout.
