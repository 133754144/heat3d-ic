# V7 G1 Close

本文件是 G1 阶段最终权威入口。它只汇总已经 science-sealed 的 P1i
`valid_iid` evidence 和 publication-only artifacts；不产生新实验、不重算
统计量、不修改 frozen receipt 或 archive。

## Stage objective

G1 回答的问题是：

> Do Heat3D internal mechanisms materially contribute under the frozen P1i contract?

结论范围严格限于 frozen P1i `valid_iid` evidence。G1 的 attribution stage 已
完成，G2 不在本阶段范围内。

## Frozen experiment contract

| Field | Frozen value |
| --- | --- |
| Dataset ID | `heat3d_v6_p1i_continuous_physics1024_v1` |
| Dataset manifest | `configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_manifest.json` |
| Dataset manifest SHA256 | `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514` |
| Full-field archive | `data/heat3d_v6_p1i_continuous_physics1024_v1_full_fields/full_fields.h5` |
| Full-field archive SHA256 | `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb` |
| Formal matrix | 7 variants × 3 seeds = 21 runs |
| Epoch budget | 200 epochs |
| Formal code SHA | `191a7a06a681556f575a1c04e2b61cb13363efe1` |
| Checkpoint rule | frozen `valid_iid` `sample_first_relative_rmse_pct`, earliest-epoch tie break |
| Statistical population | 128 frozen `valid_iid` samples |
| H1/H1b primary metric | `point_global_relative_rmse_pct` |
| H2 primary metric | `source_region_RMSE_K` |
| H2 primary route | `U16384→240825` |
| H2 robustness route | `U-direct-240825` |
| Bootstrap | 10,000 two-level replicates; seed `20260829`; percentile 95% CI |
| Test/sealed | intentionally untouched |

The formal training state is `21/21 complete`. Checkpoints, predictions,
histories/logs and provenance remain in the Git-ignored archive; large evidence
files are not tracked by Git.

## Final hypothesis table

All effects use the preregistered direction `ablation_error − Full_error`, so a
positive effect favors Full. Superiority is declared only when the paired 95% CI
excludes zero, the paired median has the same direction, and all three seed
effects have the same direction.

| Hypothesis | Frozen comparison | Primary metric | Effect | 95% CI | Claim status |
| --- | --- | --- | ---: | --- | --- |
| H1 | Full vs Vanilla | `point_global_relative_rmse_pct` | 20.43909 pp | [17.12366, 23.60488] pp | `SUPERIORITY_SUPPORTED` |
| H1b | Full vs capacity-matched Vanilla | `point_global_relative_rmse_pct` | 22.37245 pp | [17.44747, 27.71174] pp | `SUPERIORITY_SUPPORTED` |
| H2a | Full vs generic support | `source_region_RMSE_K` | 1.74615 K | [1.48199, 2.04268] K | `SUPERIORITY_SUPPORTED` |
| H2b | Full vs CV-only support | `source_region_RMSE_K` | 1.84057 K | [1.53070, 2.18014] K | `SUPERIORITY_SUPPORTED` |
| H3 | Full vs no FiLM | `sample_first_relative_rmse_pct` | 0.29417 pp | [0.21456, 0.37041] pp | `SUPERIORITY_SUPPORTED` |
| H4 | Full vs `physics-scale-only (learned residual scale correction removed)` | `raw_K_CV_RMSE_K` | 171.44018 K | [164.25621, 178.46001] K | `SUPERIORITY_SUPPORTED` |

H2 is one hypothesis group with two preregistered contrasts, not three
independent hypotheses:

- H2a: Full vs generic support.
- H2b: Full vs CV-only support.

The H2 primary per-seed effects (seeds 0/1/2) are H2a
`1.96483 / 1.56857 / 1.70998 K` and H2b
`1.66052 / 1.71218 / 2.13916 K`. The frozen robustness route gives H2a
`1.74511 K`, 95% CI `[1.48176, 2.04234] K`, with per-seed effects
`1.96409 / 1.56616 / 1.71016 K`; H2b gives `1.82170 K`, 95% CI
`[1.53548, 2.13621] K`, with per-seed effects
`1.71689 / 1.66740 / 2.07672 K`. Both routes retain the same positive
attribution direction and claim status.

The capacity-matched Vanilla seed values are retained in the publication table:
its native `point_global_relative_rmse_pct` mean ± sample SD is
`24.021 ± 5.459%`, with seed values `30.105 / 19.551 / 22.407%`.

## Scientific conclusions

Within the frozen P1i `valid_iid` contract:

1. Full vs Vanilla supports the combined benefit of the complete Heat3D
   conditioning architecture.
2. The capacity-matched Vanilla result shows that parameter capacity alone does
   not explain the Full advantage; its high seed variance is reported rather
   than hidden by a pooled mean.
3. The `physics-layout-aware sparse support` attribution is supported by the
   240825 common-domain H2 primary. It is a layout/physics-region support
   mechanism, not an amplitude-aware support claim.
4. The H2 attribution direction is robust to the two frozen U routes.
5. FiLM is a smaller but stable secondary contribution.
6. `physics-scale-only (learned residual scale correction removed)` shows that
   learned residual scale correction is critical under the frozen P1i
   formulation.

## Claim boundaries

G1 does not support any of the following claims:

- RIGNO superiority over GINO or Transolver;
- `test_iid` superiority;
- OOD or external superiority;
- SOTA, deployment, latency, or generalization claims;
- `source-amplitude-aware` support.

The support term used in all publication-facing text is
`physics-layout-aware sparse support`. Native-1024 H2 remains a supplementary
diagnostic only. The historical 21-run mixed-domain summary is labeled
“Historical mixed-domain summary — not for direct cross-variant comparison.”

## Historical audit disposition

- V6 `p2r=3074` is retained as a historical reproducibility diagnostic only;
  it is not the H2 scientific gate or the source of the native execution
  envelope.
- Early H2 Gate A/full-field fail-closed receipts remain unchanged as
  superseded audit history. They do not describe the current authoritative G1
  state.
- The padding amendment is an execution-shape-only amendment. It changed
  masked capacity after the complete geometry audit; it did not change support,
  radius, real edge sets, valid tensor prefixes, or prediction semantics.
- Historical receipts and the 719-file archive are retained unchanged.

## Evidence seal

| Evidence | Frozen reference |
| --- | --- |
| Science commit | `89e3149a1916d3441e1b9569f544ba516af4e1d8` |
| Immutable science tag | `v7-g1-science-sealed-20260903` |
| Tag object SHA | `33d8f8f52593b55353d8682e3f639d418ce629ea` |
| Final science seal receipt | [`v7_g1_final_science_seal_receipt.json`](v7_g1_final_science_seal_receipt.json) SHA `7dcf8053d421cf44fc593d5366248fe2de4a478aedfb57cfbf88b8f14be344f6` |
| Archive manifest | [`v7_g1_formal_archive_manifest.json`](v7_g1_formal_archive_manifest.json) SHA `396326724cb4f151e2e1f5f8b5c70ea1626c83fb5bebece3019e27d56590d80c` |
| Archive seal | [`v7_g1_off_worktree_archive_seal.json`](v7_g1_off_worktree_archive_seal.json) SHA `7f989624fa164b7c84b3822358673e4fc48725e505dcec32cf117ffa12f36b02` |
| Off-worktree archive | `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/` |
| Archive integrity | 719/719 files; size and SHA256 exact; 3,872,187,477 bytes |
| Publication summary | [`v7_g1_publication_summary.md`](v7_g1_publication_summary.md) |
| Publication tables | [`v7_g1_publication_tables/`](v7_g1_publication_tables/) |
| Figure manifest | [`v7_g1_figure_manifest.json`](v7_g1_figure_manifest.json) |
| Figure outputs/provenance | [`v7_g1_figures/`](v7_g1_figures/) |

The science-sealed commit and tag are immutable references to the evidence
state. The latest publication-close branch HEAD is the current `research/v7`
tip after the publication-only commit(s), and is intentionally distinct from
the science commit; its exact SHA is reported by the final Git verification.

## Final state

`G1_SCIENCE_FROZEN_AND_EVIDENCE_SEALED`

`G1 attribution stage complete; test_iid/sealed remain intentionally untouched for future final-model evaluation.`

Remaining scientific blockers: `none`.

Any future G1 scientific change requires an explicit new protocol/version and
must not silently modify the sealed evidence.
