# V6 P1i inference artifact inventory

This is a read-only inventory at parent commit `5ae734e`. No training or
inference was run, and test/sealed roles remained closed.

## Frozen runs

| run | host | primary checkpoint | 1024 support | 1024 to 240825 | reload | timing | cross-resolution |
|---|---|---|---|---|---|---|---|
| V6_06 seed0 | devbox | e559, `51567afe...b90e` | complete | complete | complete | fixed-32 seed0 available | full-graph re-discretization diagnostic only |
| V6_07 seed1 | WSL2 | e455, `71971579...e71f7` | complete | complete | complete | not run | not run |
| V6_08 seed2 | WSL2 | e587, `d67e0dac...2ab49` | complete | complete | complete | not run | not run |

The frozen three-seed manifest contains six checkpoints per run, five
prediction archives, `run_config.json`, `loss_summary.json`, environment,
training log hashes, valid support metrics, full-field reconstruction and
independent reload evidence. Remote read-only verification found all core
checkpoints, predictions, run configs, loss summaries and `valid_full_field`
artifacts with matching hashes. V6_06 remains only on devbox; V6_07/08 remain
on WSL2.

Some locally archived evidence files are not duplicated inside the current
remote run directories: `all_valid_fullfield.*`, `valid_support.*` and
`independent_replay.json`. Their sizes and hashes remain frozen in
`v6_p1i_three_seed_artifact_manifest.json`; this is an archive-location gap,
not a missing-result claim.

WSL2 was clean at `5ae734e` and matched the remote branch. Devbox was also
clean, but intentionally remained at the frozen training commit `3884de0`;
no pull or fetch was performed on that archive host.

## Existing evaluation scope

- Complete: three-seed native 1024 support and 1024-to-240825 layer-aware
  reconstruction on `valid_iid`.
- Complete: seed0 fixed-32 timing and 1024-to-full-field comparison.
- Complete but diagnostic-only: seed0 measure-conservative full-graph
  re-discretization at 512--16384. Its N=1024 cell already differs sharply
  from exact checkpoint replay and cannot stand in for anchor-derived query
  inference.
- Historical only: `heat3d_v6_randomblock_formal1024_v2` transfer/runtime OOD
  diagnostics. That dataset has no current formal OOD or independent benchmark
  role.

## Blocking gaps before cross-resolution execution

| gap | status | required closure |
|---|---|---|
| P1i sample-varying anchor/query adapter | missing | implement without changing checkpoint/model parameters |
| exact R0 replay for V6_06/07/08 | missing for new workflow | reproduce each seed's frozen 1024 support/full-field metrics |
| 4096/8192/16384 anchor-derived valid results | missing | run only after R0 and cache checks pass |
| per-support sparse graph caches and reconstruction maps | missing | bind to support/config/seed/code fingerprints |
| same-host V6_07/08 timing | missing | required before a multiseed runtime statement |
| optional 32768 | eligible, not run | may run only after mandatory valid-only cells pass |

Machine-readable details are in
`configs/heat3d_v6_p1i/v6_p1i_inference_artifact_inventory.json` and its CSV
mirror.
