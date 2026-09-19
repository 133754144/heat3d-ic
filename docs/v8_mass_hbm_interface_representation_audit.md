# V8 MASS-HBM interface-representation audit

Date: 2026-09-19
Status: `DOUBLE_COUNT_GUARD_IMPLEMENTED`

## PAPER FACT

- MASS-HBM models material conductivity as temperature/stress dependent and interface resistance as temperature/stress dependent.
- Fig. 3 and Fig. 8 treat bonding/interfaces and 2.5D/3D stacks as explicit physical scales.
- The paper does not define the exported array indexing or state that an explicit TBR field may always be combined with an effective conductivity tensor.

## RAW FACT

All 314 cases were checked, including the full `final_interface_tbr_map.npz` arrays.

| Representation | Cases | Explicit TBR map |
|---|---:|---|
| `explicit_series_z` | 53 | nonzero in all 53 |
| `rotated_homogenized_tensor` | 261 | exactly zero in all 261 |

- Every explicit case has nonempty `interfaces.csv`; all rows carry `PASS_interface_only_bonding_representation`.
- The physical manifest identifies the explicit map as z-normal series resistance.  `final_interface_tbr_map[z_face,y,x]` lies between cells `(z,y,x)` and `(z+1,y,x)`.
- The explicit face area is the orthogonal control face `dx*dy`, and raw TBR has unit `m² K/W`; the guarded Track-A conversion is `G_face=A/R''`.
- In homogenized cases, the manifest states that interface resistance is embedded in the rotated/effective diagonal conductivity representation; the explicit map is zero.
- Exactly one case, `cases/Baseline/3d_v_hbm/full_coupling`, is marked `REVIEW_REQUIRED` by raw metadata.  It is excluded from semantic-safe adapter gates.
- No case in the handoff has both a nonzero explicit map and `rotated_homogenized_tensor`.

## INFERENCE / DECISION

Using both homogenized final `k` and a nonzero explicit `Rint` for the same interface would double-count resistance.  The adapter therefore enforces mutually exclusive modes:

- `rotated_homogenized_tensor`: accept final diag3 `k`; require explicit TBR map to be exactly zero; add no interface feature.
- `explicit_series_z`: accept an explicit interface channel only when every raw guard is `PASS_interface_only_bonding_representation` and positive faces exist.
- `REVIEW_REQUIRED`, unknown representation, missing guard, or homogenized+nonzero-TBR: raise an error.

Candidate-A node aggregation records `G/V` and signed `G*n/V` moments on the two incident cells.  It is lossy because it discards exact face pairing after aggregation.  Candidate-B should preserve the paired interface as an edge.  A future canonical edge contract may use half-cell normal resistances plus `R''/A`, but this phase does **not** freeze that formula because pre-solve `k_n`, distance conventions, and nominal interface-face masks remain incomplete.

Final `Rint` is `ORACLE`.  The nominal tables establish that reference parameters exist for the 53 explicit cases, but the handoff does not contain a complete target-independent x/y face mask from which `Rint0(x,y,z-face)` can be reconstructed.  Track B therefore remains fail-closed.

Verdict: Track-A non-LC smoke on a homogenized/zero-explicit-TBR case is `GO`; explicit Track-A cases are `CONDITIONAL_GO` under the raw no-double-count guard; Track-B interface physics is `NO_GO_PENDING_PRE_SOLVE_FACE_MAP`.
