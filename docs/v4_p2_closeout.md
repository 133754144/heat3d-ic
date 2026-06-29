# V4 P2 Closeout

Read this file only for V4 P2 closeout or merge-gate checks.

## Scope

V4 P2 covered input feature and coordinate policy experiments based on the
V4P1_12 train-consistent split path. Results are diagnostic.

## CSV Result Confirmation

All three V4P2_01-03 runs have completed result fields in
`configs/heat3d_v4/run_registry.csv`.

| Config | Main Change | Commit | Best valid_base_mse | Best RMSE | Final valid_base_mse | Final RMSE | Final-probe relRMSE | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `V4P2_01_boundary_distance_replacement` | per-axis boundary distances, legacy coords | `e2cf4fc` | 0.00879056751728 | 0.0937580264152 | 0.00879417359829 | 0.093777255229 | 0.710340709788 | completed |
| `V4P2_02_xyz_unified_coord_policy` | sample-local isotropic coords, legacy BC flags, log extent | `e2cf4fc` | 0.00870955828577 | 0.0933250142554 | 0.00874506868422 | 0.0935150719629 | 0.69151288912 | completed |
| `V4P2_03_boundary_distance_plus_xyz_unified` | sample-local isotropic coords, isotropic boundary distances, log extent | `5c8ba69` | 0.0104100611061 | 0.102029706978 | 0.0104845846072 | 0.102394260616 | 0.709447309582 | completed |

## Closeout Notes

- `V4P2_02` is the strongest P2 diagnostic result by validation MSE among the
  three controlled P2 runs.
- `V4P2_03` was rerun after the boundary-distance scaling fix; the CSV now
  records the corrected `5c8ba69` result rather than the invalidated per-axis
  result.
- No new training was launched for this closeout; `V4P2_03` was collected from
  an existing devbox output directory.
