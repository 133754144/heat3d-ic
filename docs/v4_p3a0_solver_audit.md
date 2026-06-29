# V4 P3a-0 Solver Audit

Scope: pre-implementation audit for the V4 P3a sparse-equivalent solver
refactor. This document does not implement a new solver, start training, or
generate data.

## Current Reference Solver

The current label path has two solver modules:

- `rigno/heat3d_v1_reference_solver.py`: legacy dense smoke solver.
- `rigno/heat3d_v1_reference_solver_v2.py`: dense smoke solver with stronger
  metadata and diagnostics.

Both solve the same restricted steady equation family:

```text
-div(k grad T) = q
```

Supported problem assumptions:

- regular multilayer rectangular stack only;
- complete rectilinear point grid after duplicate-coordinate merge;
- `k_field` shape `(N,1)` isotropic or `(N,3)` diagonal anisotropic;
- top Robin boundary, bottom Dirichlet boundary, side adiabatic boundary;
- perfect-contact interfaces only;
- no contact resistance, no irregular footprint, no explicit TSV/BEOL/bump
  geometry, no transient coupling, and no full `(N,6)` tensor conductivity.

Dense assembly path:

- coordinates are merged with `np.unique`, then mapped to full `xs x ys x zs`;
- missing rectilinear grid points raise an error;
- a dense `n x n` matrix and dense RHS are allocated;
- neighboring face conductance uses harmonic mean of the two node
  conductivities times face area over neighbor distance;
- top Robin contributes to diagonal and RHS on the `z_max` layer;
- bottom Dirichlet is imposed by replacing the row with `T = T_bottom`;
- side adiabatic is the natural no-neighbor zero-flux boundary;
- the linear system is solved with `np.linalg.solve`.

Perfect-contact behavior:

- duplicate interface coordinates are merged before assembly;
- duplicate `k` values are arithmetic-averaged before later face harmonic
  means;
- duplicate `q` values use max pooling to preserve active source values;
- v2 records duplicate-merge metadata, but it does not compute interface heat
  flux mismatch or temperature-jump diagnostics.

Rectilinear dependency:

- `_grid_mapping` requires every Cartesian product point of `unique(x)`,
  `unique(y)`, and `unique(z)` to exist;
- control-volume widths are inferred only from 1D coordinate axes;
- source, face area, and boundary treatment assume axis-aligned structured
  control volumes.

Existing metadata:

- v2 returns `label_meta` with solver name/version, discretization type,
  supported k mode, convergence flag, dense residual proxy, bottom Dirichlet
  error, BC/interface status fields, assembly metadata, duplicate-merge
  metadata, and warnings.
- v2 generation writes `label_meta.json` beside `temperature.npy`.

Metadata gaps for V4 P3:

- no solver family field distinguishing dense legacy, sparse equivalent, and
  contact-resistance modes;
- no matrix backend, sparse format, ordering, condition estimate, solve method,
  tolerance, or iteration count;
- no global energy-balance residual;
- no top Robin flux audit;
- no side adiabatic flux audit;
- no interface flux mismatch audit;
- no contact-resistance value, temperature-jump audit, or contact heat-flux
  continuity audit;
- no grid checksum or operator checksum for dense-vs-sparse equivalence;
- no publication-validation status field.

## P3a Sparse-Equivalent Design Target

Keep the current dense solver as the legacy reference. Add a separate sparse
solver path with explicit modes:

- `legacy_equivalent`: same physics and same inputs as dense v2; intended to
  produce numerically identical labels within a strict tolerance.
- `perfect_contact`: sparse production path for current perfect-contact cases.
- `contact_resistance`: sparse path with explicit thermal boundary resistance
  across selected interfaces.

Recommended interface shape:

```text
solve_temperature(sample_dir, solver_options) -> temperature, label_meta
build_operator(problem) -> operator, rhs, operator_meta
audit_solution(problem, operator, rhs, temperature) -> diagnostics
```

The key split is operator assembly versus solve versus diagnostics. P3a should
not mix dataset generation or model training into this refactor.

Sparse data model:

- reuse current rectilinear indexing first;
- assemble row/col/data triplets, then convert to CSR/CSC;
- keep dense assembly available only for legacy comparison and tiny tests;
- expose stable node ordering and operator checksum in metadata;
- store boundary/interface faces as structured face records, not ad hoc masks.

Contact-resistance model:

- perfect contact is the `R_contact = 0` limit;
- for finite `R_contact`, interface conductance should include the added
  thermal resistance in series with the adjacent half-cell resistances;
- metadata must record `R_contact_m2K_W`, affected interface ids, contact mode,
  and sign convention for temperature jump.

## Equivalence Gates

P3a sparse-equivalent work is not allowed to change labels in
`legacy_equivalent` / `perfect_contact` mode.

Required gates before replacing the active label path:

- Same-input dense-vs-sparse temperature max absolute difference <= `1e-10` for
  current small controlled samples.
- Same-input dense-vs-sparse DeltaT max absolute difference <= `1e-10`.
- Dense and sparse operator RHS should match by row semantics, not only by final
  temperature.
- Dense and sparse bottom Dirichlet enforcement must match exactly on bottom
  nodes.
- Sparse residual proxy must be finite and within the existing v2 tolerance
  envelope.
- v2 smoke cases must still pass: regular samples, zero-q case, baseline shift
  case, isotropic and diag3 k modes.
- No generated `temperature.npy` should be overwritten in the equivalence test.

Suggested local check names for the future implementation:

- `scripts/check_heat3d_v4_p3a_sparse_legacy_equivalence.py`
- `scripts/check_heat3d_v4_p3a_contact_resistance_smoke.py`

These should use temporary samples or ignored local samples only.

## Contact-Resistance Gates

For `R_contact > 0`, exact equivalence to dense legacy is not expected. Required
smoke gates:

- `R_contact = 0` matches perfect-contact sparse mode.
- Increasing `R_contact` monotonically increases interface temperature jump for
  a fixed heat-flow direction.
- Increasing `R_contact` should not increase heat flux through the same
  interface under the same source/BC setup.
- Linear residual stays below tolerance.
- Energy-balance residual is recorded and bounded.
- Interface diagnostics report per-interface heat flux, temperature jump,
  effective contact conductance, and sign convention.
- Metadata records contact-resistance inputs and audit status.

## Publication-Oriented Solver Direction

P3a alone should not claim publication-grade validation. It should create an
auditable solver implementation that can be validated in P3b.

Publication-oriented requirements:

- analytic or manufactured-solution cases with known error behavior;
- grid-refinement protocol with fixed physical source power;
- global energy-balance audit;
- top Robin, side adiabatic, and bottom Dirichlet flux audits;
- interface heat-flux and temperature-jump audit;
- independent external reference comparison, such as FEniCS or equivalent;
- parameter provenance for geometry, `k`, `q`, BC, and contact resistance before
  dataset generation.

## File And Interface Plan

Keep legacy files unchanged unless an explicit compatibility shim is needed:

- `rigno/heat3d_v1_reference_solver.py`: legacy reference; no behavior changes.
- `rigno/heat3d_v1_reference_solver_v2.py`: dense v2 reference; no behavior
  changes unless metadata-only compatibility fields are added later.

Add future sparse implementation under a new module:

- `rigno/heat3d_v4_reference_solver.py`

Possible internal objects:

- `Heat3DProblem`: coords, k, q, BC, interfaces, units, sample metadata.
- `OperatorMeta`: grid shape, node order, sparse format, nnz, backend, checksum.
- `SolutionAudit`: residual, BC fluxes, interface fluxes, energy balance,
  monotonicity flags, warnings.

## P3a-1 Interface Skeleton

P3a-1 adds the non-solving problem extraction and operator contract skeleton in
`rigno/heat3d_v4_reference_solver.py`.

Implemented dataclass contracts:

- `Heat3DProblem`
- `GridSpec`
- `GridMapping`
- `BoundarySpec`
- `InterfaceRecord`
- `OperatorMeta`
- `SolutionAudit`
- `SolverOptions`

Implemented helpers:

- `load_problem_from_sample(sample_dir)`
- `extract_problem_from_arrays(coords, k_field, q_field, sample_meta)`
- `operator_meta_for_problem(problem, options)`

The extraction helper reuses dense-v2 semantics: `(N,1)` `k_field` expands to
diag3, duplicate coordinates are merged by arithmetic-mean `k` and max-pooled
`q`, rectilinear node ordering follows `np.unique(axis=0)`, and only
perfect-contact interface metadata is accepted.

P3a-1b fixes duplicate `q` pooling to be negative-q-safe. The max-pooling
policy is unchanged, but the accumulator is initialized from a true negative
infinity sentinel instead of `0`, so duplicate values such as `[-5, -2]` merge
to `-2` rather than being clipped to `0`.

P3a-1 checker:

- `scripts/check_heat3d_v4_p3a_problem_extraction.py`

The checker uses in-memory tiny synthetic samples only. It verifies rectilinear
grid mapping, node ordering, isotropic and diag3 `k`, top/bottom/side face
indices, duplicate merge policies, perfect-contact interface records, and the
operator metadata skeleton. It does not write artifacts, generate
`temperature.npy`, assemble a sparse operator, or solve a system.

P3a-1b extends the checker with a negative-q duplicate case:

- duplicate `q = [-5, -2]` must merge to `-2`.

Read-only devbox real-sample extraction check:

- branch pulled on devbox: `research/v4-solver`;
- checker result: `p3a_problem_extraction_ok: true`;
- samples checked:
  - `sample_000`: isotropic expanded to diag3, grid `(4, 4, 5)`, 80 merged
    nodes, 48 duplicate nodes, 3 interfaces;
  - `sample_005`: isotropic expanded to diag3, grid `(4, 4, 5)`, 80 merged
    nodes, 48 duplicate nodes, 3 interfaces;
  - `sample_008`: diag3, grid `(4, 4, 5)`, 80 merged nodes, 48 duplicate
    nodes, 3 interfaces.
- remote check was read-only: no solve, no `temperature.npy`, no data/output
  artifact writes.

Generator integration should stay deferred until solver gates pass. When it is
enabled, generators should record:

- solver family and version;
- solver mode;
- matrix backend and tolerance;
- contact model and interface ids;
- grid/operator checksum;
- energy and interface audit status;
- publication-validation status.

## P3a Execution Plan

1. Freeze dense legacy behavior as the comparison target.
2. Add problem extraction helpers without changing current solver output.
3. Add sparse assembly for current perfect-contact rectilinear cases.
4. Add dense-vs-sparse equivalence checks with `1e-10` max-difference gate.
5. Add residual, BC, and energy diagnostics on the sparse path.
6. Add contact-resistance assembly only after perfect-contact equivalence passes.
7. Add contact-resistance smoke and monotonicity checks.
8. Update solver metadata contract and label metadata fields.

No dataset generation should start until P3a passes the equivalence and contact
smoke gates. P3b validation must pass before calling the solver
publication-oriented.
