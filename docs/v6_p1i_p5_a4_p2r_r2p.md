# V6 P1i P5-A4 P2R/R2P closeout

The sparse exact backend now performs one batched, multicore
`query_ball_point` call. Candidate lists are still filtered with the frozen
input-dtype distance predicate and globally lexicographically sorted, so the
canonical edge order is unchanged. GPU-tiled construction was not used.

## Exact gates

Across B8192/E32768 frozen valid32 (64 graphs):

- reference and batched P2R arrays/hashes are exact;
- reference and batched R2P arrays/hashes are exact;
- canonical graph hashes are exact;
- when `x_in=x_out` and effective P2R/R2P radii are exactly equal, resolving
  the implicit reverse adjacency reproduces both reference arrays/hashes.

## Timing

| Route | Serial P2R+R2P (s) | Batched (s) | Exact reverse reuse (s) | Batch speedup | Reuse speedup vs batch |
|---|---:|---:|---:|---:|---:|
| B8192 | 0.042797 | 0.031753 | 0.017960 | 1.348x | 1.768x |
| E32768 | 0.073135 | 0.057817 | 0.030817 | 1.265x | 1.876x |
| pooled | 0.066559 | 0.043555 | 0.027589 | 1.528x | 1.579x |

Both changes are `GO`. Reverse reuse remains an explicit implementation option
and is enabled only after the exact equality predicate succeeds; frozen graph
policy inputs and historical artifacts are unchanged.
