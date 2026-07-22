# V6-P1h shared-support full-field availability audit

Status: `original_full_fields_missing_rebuild_required`. Decision: `deterministic_rebuild_from_frozen_P1g`.

The immutable P1g manifest contains 1024 samples. All
1024 available sample directories contain
the projected 1024-point files, while
`samples_with_complete_solver_coords_k_q_T_count=0`.
Each sample metadata record declares a [240825]-node solver
mesh, but no sample directory contains solver-sized coordinate, conductivity, source,
and temperature arrays.

The original full fields were unavailable, so the frozen decision is to rebuild
the public mesh, conductivity, exact source field, and temperature deterministically
from the verified P1g config/manifest. The existing 1024-point P1g arrays remain
forbidden as interpolation input. Replay, generation, and final acceptance are
documented separately; this file preserves the pre-generation search evidence.

Audit source manifest SHA256: `e5329d5cd6253510d87a4432d5f2ddae67259637810c29fdfb6ddf42621875a4`.

The supplemental local search scanned
63598 NumPy files outside
the P1g projected source root. It found
`solver_sized_array_count=0`,
`archive_candidate_count=0`,
and
`solver_or_full_field_name_candidate_count=297`.

## Downstream disposition

This search result is not a stop condition. It establishes why deterministic
solver replay was required; it does not claim that the later P1h dataset is absent.
