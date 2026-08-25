# V6/P1i fixed-geometry supplemental runtime protocol

This supplemental study starts from `main@6922e80c392385a8ae3d09b720c5307aaee1fffd`.
It does not modify the frozen P1i dataset, checkpoint, model, graph policies, or
reconstruction algorithm.  It reads input fields from four **train-only**
geometries and never opens temperature labels, `test_iid`, or sealed IID.

The four geometries are selected before inference using only `source_count`:
for counts 3, 5, 7, and 10, choose the train sample with the lexicographically
smallest SHA256 of its sample ID.  The frozen IDs are `v6p1if1_0079`,
`v6p1if1_0971`, `v6p1if1_0393`, and `v6p1if1_0056`.

For each geometry, mesh, material/source regions, source mask, and Robin BCs are
fixed.  The K-only sweep changes conductivity through four pre-registered
in-range material quantiles while retaining the q array byte-for-byte.  The
K+Q-scale sweep uses the same conductivity points and positive multipliers
`0.8/0.95/1.05/1.2`; it preserves the q mask and normalized spatial pattern.
Every generated field is checked against the formal1024 material, q, and total
power bounds before inference.

The benchmark compares the four frozen neural inference strategies under three
reuse semantics.  `fresh_new_case` rebuilds all case-specific preparation;
`graph_only_reuse` reuses support and graph state; `full_static_reuse` also
reuses reconstruction and structural packing/JIT state while recomputing every
k/q-dependent feature.  Correctness is a hard gate: static identities must be
exact and cached predictions must agree with the standard path within the
frozen numerical tolerance.  Static setup and compilation are reported
separately from repeated-sweep latency.

No FVM labels are generated and no FVM speedup claim is made in this study.
