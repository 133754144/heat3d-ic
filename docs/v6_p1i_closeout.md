# V6/P1i scientific closeout

## Decision

**V6/P1i scientific development = CLOSED.** The frozen offline peak-tail audit
found no evaluator, data, or provenance error. No result in this closeout was
used for selection or tuning, and sealed IID remains ungenerated and unopened.

## Frozen scientific identity

- Dataset: `heat3d_v6_p1i_continuous_physics1024_v1`, the canonical
  `formal_v6_randomblock` role; manifest SHA256
  `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514`.
- Full-field sidecar: 240,825 solver nodes; archive SHA256
  `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb`.
- Reference model: `V6_06_V5best_P1i_seed0_reliable_B24`, point-global best
  epoch 559, checkpoint SHA256
  `51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e`.
- Replications: V6_07 seed1 and V6_08 seed2, retained under the same frozen
  train/valid contract.
- Production/reference inference strategy: native 1,024 source-aware anchors,
  E policy at 16,384 query nodes, then frozen layer/interface-aware
  reconstruction to 240,825 nodes.
- Confirmatory role: the frozen `model_seed0` E16384 route was opened once on
  corrected confirmatory `test_iid=128`; it was not used for model or route
  selection.

## Frozen claims

1. On frozen valid32 evidence, 16k + reconstruction is the principal
   accuracy-latency Pareto family.
2. On the WSL2 authoritative benchmark, E16384 fresh and Q2 speedups relative
   to persistent CPU FVM are approximately 2.00x under the paired workload
   definitions; devbox is a separate hardware-state replication, not extra
   seeds.
3. The corrected confirmatory test result for E16384 is full-field PG
   2.992001%, sample-first 2.948519%, raw CV RMSE 2.389097 K, source RMSE
   3.940479 K, peak RMSE 5.726285 K, and interface RMSE 0.355507 K.
4. Peak RMSE normalized by the frozen 180 K scale is 2.232229% on valid32 and
   3.181269% on test128. The test increase is primarily tail-driven: its top ten
   samples explain 95.45% of excess peak-error SSE relative to the valid32 mean
   SSE, with a smaller broad shift in the median.
5. U-v2 remains a valid-only parallel/direct inference characterization, not
   the confirmatory production route. E240825 remains an architecture control.
6. FVM is the reference physical solver; surrogate error is always measured
   against it. Neural speed claims are workload- and hardware-bounded and do
   not imply greater physical fidelity.

## Applicability boundary

The claims apply to the frozen formal P1i generator/distribution, perfect
interface contact (`R_contact=0`), its BC/material/source ranges, the frozen
checkpoint family and the registered anchor/query/reconstruction semantics.
They do not establish behavior for contact-resistance variation, unseen
geometries or sealed IID. Peak errors have a high-energy absolute-error tail,
and 32,768 nodes remain exploratory rather than a production accuracy choice.

No further valid/test analysis may change these claims. Any new dataset,
contact-resistance study, architecture, graph policy, or sealed evaluation is
a new preregistered phase rather than continuation of V6/P1i development.
