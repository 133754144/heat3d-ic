# V8 MASS-HBM pre-solve reconstruction audit

Date: 2026-09-19
Status: `TRACK_B_FAIL_CLOSED_PARTIAL_RECONSTRUCTION`

## PAPER FACT

MASS-HBM starts from architecture, geometry, material, boundary and workload inputs, then self-consistently updates temperature-dependent conductivity, interface resistance, leakage and refresh power.  These initial design inputs—not converged fields—are the admissible source for a deployable Track-B surrogate.

## RAW FACT

The following availability was verified across the 314-case handoff:

| Desired Track-B input | Raw evidence | Reconstruction status |
|---|---|---|
| `geometry0` | solver shape, `dx/dy`, nonuniform z thickness, extents in all cases | numeric grid reconstructable; initial-vs-final-adapted status unknown |
| external `BC0` | ambient and top/bottom/side HTC in all cases | reconstructable for exported full-box faces |
| internal LC `BC0` | aggregate sink counts, conductance, sink temperature in 232 cases | blocked: no per-cell/per-face topology |
| nominal material parameters | baseline/reference material table in all cases | tables present |
| spatial `k0(x)` | no complete pre-solve material-region map | blocked |
| iteration-0 component power | iteration-0 feedback/component records in all cases | global/component values present |
| spatial `q0(x)` | no iteration-0 volumetric map or complete source-region mask | blocked |
| nominal `Rint0` | reference interface tables in 53 explicit cases | partial; complete x/y z-face mask absent |
| final `k/q/Rint` | complete converged fields | forbidden for Track B (`ORACLE`) |

Stress/strain fields and a deployable law sufficient to evaluate `k(T,σ)` or `Rint(T,σ)` are absent from the handoff.

## INFERENCE / DECISION

Track B cannot yet be constructed without inventing spatial assignments.  Unit guessing, copying final fields, silently treating final mesh as pre-solve, or using a V7 fallback is prohibited.  The future Track-B gate calls `reject_oracle_features` and raises if any input is marked `ORACLE`.

The output representation recommendation is `T - T_ref(case physics)`, where `T_ref` is computed only from pre-solve sink temperatures/conductances.  Raw absolute temperature is valid but less translation-stable; a single global reference is physically insufficient when ambient/sink temperatures vary; label-derived mean or peak temperature is forbidden.  V7 native shape-scale is not adopted unchanged because multiple internal sinks require a generic reference/BC formulation.

### Reconstructable now

- zero-origin-equivalent cell centers, control volumes, and structured connectivity;
- external full-box Robin descriptors `(A,n,G,T_inf)`;
- case-level geometry design parameters and nominal material/interface tables;
- initial component/workload power totals and iteration-0 bookkeeping.

### Still blocked

- a spatial pre-solve material/region map for `k0(x)`;
- a spatial initial source/activity map for `q0(x)`;
- a complete nominal explicit-interface face mask for `Rint0`;
- per-cell/per-face internal cooling topology for `BC0`;
- proof that the exported geometry is target independent.

### Author questions

1. Provide the exact pre-solve material/region-id field and mapping to the nominal material table.
2. Provide the iteration-0 volumetric power field, or the deterministic component-to-cell injection masks and normalization rule.
3. Provide the nominal interface face mask and orientation/index convention before temperature/stress updates.
4. Provide per-cell/per-face internal sink topology and whether HTC/sink temperature are fixed during coupling.
5. State whether the exported grid is initial or temperature-adapted and provide the initial mesh if different.

Verdict: Track-B training is `NO_GO`; non-LC Track-A engineering smoke is independently permitted and cannot support a solver-replacement claim.
