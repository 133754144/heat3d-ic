# G2-P5 final local protocol closure

This closes local G2 feature work. No SSH/devbox connection, GPU work, formal
training, P1i test/sealed read, or new baseline was used.

## GINO backend fairness

The scientific contract remains `r_in=0.15`, `r_out=0.033`, latent `32^3`, and
the frozen architecture/objective/optimizer/normalization. The Mac gate remains
the pure-PyTorch fallback. Formal execution is bound to Open3D fixed-radius
search plus torch-scatter and fails closed without CUDA or either dependency.

The remote preflight compares the exact per-query edge multiset for both input
and output GNOs and checks identical-weight outputs with `atol=rtol=1e-5` on a
fixed train sample. It also records GPU name, peak allocated/reserved memory,
one train-step wall time, and one valid-forward wall time. This Mac has no CUDA,
so optimized-backend qualification is deliberately pending rather than inferred
from fallback success.

## DeepOHeat-v1 method-native support and labels

The formal Heat3D input is now a deterministic 1024-point physics-layout-aware
support: 256 source-layout, 128 material-interface, 64 top, 64 bottom, and 512
remaining-volume points. Selection sees boolean source layout, fixed material
interface, coordinates, boundaries, and control volume only. It does not see
source amplitude, temperature, prediction/error, or official test data. All 896
train/valid cases have unique support hashes and satisfy the quota contract.

The experiment is explicitly a **method-native cross-benchmark**. Heat3D sees
1024 sparse pointwise `coords+k+q+BC` observations; DeepOHeat-v1 sees a complete
101x101 input function and physics residuals. Same training budget or same input
information budget is not claimed.

The fidelity-passed CPU solver generated all 768 train and 128 valid fields.
The cache is 2,091,740,672 bytes under
`/private/tmp/g2_p5_deepoheat_v1_labels`; nothing large is in Git. Mean solve
time was 2.063 s/case (median 1.990 s, P95 2.287 s, max 3.017 s), maximum linear
residual was `9.99e-11`, and peak process RSS was 686,669,824 bytes. Every case
records source, full-field, support, 1024 extraction, z=0.15 slice, residual,
time, and environment hashes. The official 100 test inputs/fields remained
untouched.

Train-only 768-function Heat3D normalization is frozen at payload SHA256
`3a0273bb92b8c060df8a214b1e0e7dd0e4b5df6bece86b7dea15197ca56ed0db`;
valid and test were not fitted.

## Dual-output gate and domain boundary

The single 6-train/2-valid one-step checkpoint passed native 1024 output,
validation, checkpoint/reload exactness, and frozen reconstruction to 10201
outputs. A batched-group indexing error occurred only in postprocessing after
the checkpoint was already written; the checkpoint was reused and no second
optimizer step was run.

`10201 = 101x101` is the `z=0.15` source-layer top slice. The official
DeepOHeat-v1 volumetric output is `101x101x56 = 571256` points. Therefore the U
result is a slice result, not “exactly the same full output.” Direct numerical
comparison is allowed only when both methods are evaluated on the same slice;
full-field comparison requires all 571256 physical queries.

## multi-HTC correction and independent gate

Upstream's branch value `k` is frozen as the Robin length parameter
`beta/k_Robin`, not physical HTC `h`. The contract is `beta in [0.1,0.3]` and
`h=0.2/beta`, so `h in [2/3,2]`. The old h-named manifest is explicitly
superseded and ineligible as formal evidence.

An independent six-coefficient piecewise analytical solution was compared with
a 51x51x51 sparse 3D finite-volume/difference solve for three beta pairs. The
gate passed: maximum field RMSE in nondimensional `u` was `7.05e-6`, maximum
peak error `2.25e-5`, maximum linear residual `5.19e-11`, and maximum xy
nonuniformity `1.79e-12`. Formal multi-HTC labels remain ungenerated until the
post-G1 remote sequence.

## Frozen remote order

1. GINO CUDA/Open3D/torch-scatter preflight.
2. Transolver CUDA resource preflight.
3. Only after both pass, freeze both remote environments/configs.
4. GINO seeds 0/1/2.
5. Transolver seeds 0/1/2.
6. DeepOHeat-v1 volumetric seed42 upstream-faithful reproduction.
7. Heat3D-on-DeepOHeat-v1 seeds 0/1/2.
8. multi-HTC only after frozen-label and GPU resource gates.

A later baseline problem cannot reopen a common protocol after an earlier
external baseline has completed. Test unlock remains a separate evaluation-only
operation after G1, all common-task seeds, immutable valid-selected checkpoints,
and complete config closure.

Remaining blockers are G1 completion and authorization; remote CUDA dependency
and resource preflights; hash-verified transfer of the v1 label cache; binding
the frozen method-native loader to the authorized remote V7 lifecycle without
scientific changes; multi-HTC label materialization; same-hardware runtime
measurement; and the future one-time test unlock.
