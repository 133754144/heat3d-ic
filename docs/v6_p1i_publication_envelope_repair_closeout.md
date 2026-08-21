# V6 Publication Envelope Repair Closeout

## Decision

- `envelope_qualification = GO`
- `ready_for_authoritative_valid32 = NO_GO_padding_prediction_equivalence_failed`
- `publication_timing_freeze = NO_GO`

Attempt 2 was not started. No formal timing cell, latency table, speedup, or
publication statistic was generated.

## Graph-only qualification

Two independent CPU Python processes built the five frozen production graph
routes for all 32 frozen valid samples plus the target-free train warmup case.
Each produced 165 records and the complete output JSON SHA was identical:
`39a866a75ba0342b7c9c77cfbb87c71763db44a0e1480f916cea134052ac3bb0`.
Per-sample edge counts, real-edge hashes, metadata hashes, support order, and
the 12 historical unpadded golden witnesses matched exactly.

The native1024 maximum was derived from observed graphs as P2R=2911 and
R2R=4201. No capacity was manually changed from 2905 to 2911. Route-specific
capacity files contain only additional masked-dummy capacity; radius, graph
seed, sparse KD-tree backend, U-v2 repair, support, model, and route semantics
were unchanged.

## Padding equivalence failure

The graph/pack-only step regenerated all 12 prepared-payload hashes under the
qualified capacity without model inference. The subsequent GPU prediction
gate compared an exact-edge-count payload with the same real graph padded to
the qualified envelope.

The first planned sample, `v6p1if1_0308` on E16384+reconstruction, exceeded
the frozen maximum prediction tolerance of `1e-6 K`. The program failed before
writing its result JSON, so the numerical delta was not persisted. Per the
fail-fast contract, the sample was not rerun and the second sample was not
started. No tolerance, envelope, model, or graph behavior was changed.

## Attempt 1 semantic correction

Attempt 1 is explicitly classified as `attempted=true`, `completed=false`, and
`publication_results_generated=false`. Its historical raw artifact remains
unchanged. The corrected meaning is recorded in the machine-readable closeout.

## Scope

No training, accuracy tuning, test/sealed access, checkpoint modification,
dataset modification, graph-policy search, or formal authoritative timing was
performed. A future retry requires an explicit new protocol; the existing seal
was not updated to authorize Attempt 2.
