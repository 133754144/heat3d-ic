# V6 publication runtime isolation and Attempt 4 contract

## Scope

This amendment changes only benchmark runtime/control-plane behavior.  The
model, checkpoint, dataset, graph policy, padding envelopes, collector and
statistics remain frozen.  Training, test and sealed roles remain closed.

Attempt 3 is retained as immutable failure evidence.  Its lifecycle was
attempted but incomplete; it generated no publication result.  The failure at
`E16384_reconstruction / seed20260815 / Q2` was obscured by a secondary
lifecycle exception before the inner service state was written.

## Runtime boundary

The production interval is exactly:

`in-memory k/q/BC -> synchronized 240825-node result ready`

Graph and prepared-payload hashes, historical-golden comparison, checkpoint
hash audit, metric work and serialization are outside this interval.  E graph
hashing is opt-in only for the untimed audit.  U formal service calls return at
the synchronized result boundary; their graph/payload audit is a separate
untimed pass and is not part of Q2 completion or refill.  Service HWM is
captured before that untimed audit.

Every Q2 failure artifact records the original exception, sample ID, order
position, completed count, failure stage, residual and limit, and all completed
rows.  The outer orchestrator embeds the inner failure JSON and its SHA rather
than replacing it with a summary exception.

FVM P2 participation is established from worker PIDs attached to real case
results, in addition to the startup barrier.  Formal successful cells use
`status=passed`, role `formal_full_valid32`, and
`real_route_smoke_only=false`.  Order, resource, warmup and lifecycle contracts
are checked immediately after each cell.

## Gate sequence

1. Static and synthetic failure-observability regression.
2. One non-authoritative full32 diagnostic:
   `E16384_reconstruction / seed20260815 / Q2`.
3. Only if step 2 passes, a new Attempt 4 directory runs the frozen 30-cell
   matrix from zero.  No cell from Attempts 1--3 may be reused or selectively
   rerun.
4. Only 30/30 plus collector/checker success may set
   `publication_timing_freeze=GO`.
