# V7 G1 publication figure provenance

Rendered from frozen G1 evidence only. No checkpoint was loaded, no model forward/inference was run, and no new metric was defined.

- H2 primary route: `U_v2_16384_reconstruction`
- H2 primary metric: `source_region_RMSE_K`
- Selection: frozen manifest median/p90/p95 rule; no manual case selection.
- Temperature panels: common scale; error panels: common symmetric K scale.
- Population: selected `valid_iid` rows only; test/sealed untouched; G2 untouched.

## Outputs

- `docs/v7_g1_figures/G1-1_support_mechanism.pdf` — 42772 bytes — SHA256 `30b26c861e190c76f30d65bc41d985ba880037582ce8f6c5b3b44c0aa2112be9`
- `docs/v7_g1_figures/G1-1_support_mechanism.svg` — 637302 bytes — SHA256 `d6350e167020b6a56e40b91763fd6fdd7a408e60bd9e35520d781fa416d3bd1c`
- `docs/v7_g1_figures/G1-1_support_mechanism_review.png` — 587060 bytes — SHA256 `a4ae056c88e9426c6edb6c9410b994139e788a7eb5391c03ae539c9e879e6d10`
- `docs/v7_g1_figures/G1-2_h2_fullfield_comparison.pdf` — 89173 bytes — SHA256 `89e090787ce80fce2d50bff63fa98ffa4df8781b0b62bdd183f2a4b81b741365`
- `docs/v7_g1_figures/G1-2_h2_fullfield_comparison.svg` — 220303 bytes — SHA256 `bec726198c8bb673342f8b1d703db80740b667c662ff4b4d9dffb04b4ca036e4`
- `docs/v7_g1_figures/G1-2_h2_fullfield_comparison_review.png` — 332811 bytes — SHA256 `fd990cf6a2197bbaf410b3088a1774eeef5e2de2f55a780efb39974a38dda4fc`
- `docs/v7_g1_figures/G1-3_h2_paired_effect_distribution.pdf` — 21296 bytes — SHA256 `ca025ec9a1d946f54ee03c76d05f4a12d7390902869e55b9d057147351ab9fae`
- `docs/v7_g1_figures/G1-3_h2_paired_effect_distribution.svg` — 86848 bytes — SHA256 `ded8cd05b8c3d70ec4a238dca31c170e42e90e3c06878cd947b4ddd0711ca9af`
- `docs/v7_g1_figures/G1-3_h2_paired_effect_distribution_review.png` — 164247 bytes — SHA256 `feabca88bb08ec746fde6d2634f3f345d01494f61b47638e6bb0fee17b04b0ac`

## Frozen input hashes

- frozen figure manifest: `/Users/xuyihua/.codex/worktrees/9ccd/3D IC Heat/docs/v7_g1_figure_manifest.json` — SHA256 `591a76313696331f386d632af3623d9987d8f4ed13a199ecc2cd50c43d2a6b81`
- frozen archive manifest: `/Users/xuyihua/.codex/worktrees/9ccd/3D IC Heat/docs/v7_g1_formal_archive_manifest.json` — SHA256 `396326724cb4f151e2e1f5f8b5c70ea1626c83fb5bebece3019e27d56590d80c`
- frozen support provider contract: `/Users/xuyihua/.codex/worktrees/9ccd/3D IC Heat/configs/heat3d_v7/v7_g1_support_provider_contract.json` — SHA256 `7a73e16e30dd8d30e050d11820325c2ff806ddf43bc786bcd83e66f380904788`
- frozen support artifact semantics: `/Users/xuyihua/.codex/worktrees/9ccd/3D IC Heat/configs/heat3d_v7/v7_support_artifact_freeze.json` — SHA256 `2c9279f4c7f6e198178a0223e97ba676ef676fa8b249ac56b03fa03c38ff0a9f`
- frozen alternative support implementation: `/Users/xuyihua/.codex/worktrees/9ccd/3D IC Heat/rigno/heat3d_training/support.py` — SHA256 `61b02e5c2d1086aaab5ba8badf89198e618d3f46242ed5c15938e92cdf54d62f`
- valid_iid truth/shared-geometry fixture: `/private/tmp/v7_g1_publication_truth_input_20260904.npz` — SHA256 `74415add51bd92b8129d8d30d4740ed695701c3a987956470b1b56e6707f60ad`
- valid_iid native support fixture: `/private/tmp/v7_g1_publication_support_input_20260904.npz` — SHA256 `7c595f7e147cdcae678dbda4d961fc2ce23c131c7eb895083ec4d3cc6aa22944`
- valid_iid support layout metadata fixture: `/private/tmp/v7_g1_publication_support_meta_20260904.json` — SHA256 `dce35ad47963c05337effde229338c4fa0d66c20ddc8892b98778620a7242453`
- frozen H2 per-sample effects: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/h2_per_sample_effects.json` — SHA256 `fc6ffca7d9591e7bcfec32e00c306c04e1a315c0dd90a3a2418cf0d7b8df6992`
- frozen H2 hypothesis effect table: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/h2_hypothesis_effect_table.json` — SHA256 `8456df52f896bc45c572e677eb5443e8de8c5a659e7136b7922dba2e89d46095`
- frozen H2 route summaries: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/h2_variant_route_summary.json` — SHA256 `8a20f67bc9068b0f1d8a399259ea480d826abdd14d4ba2cb369c96df68cc9a84`
- frozen valid_iid full-field source archive: `devbox:/home/xyh/myCodeGitOnly/heat3d-ic/data/heat3d_v6_p1i_continuous_physics1024_v1_full_fields/full_fields.h5` — SHA256 `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb`
- H2 primary prediction Full; median: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/U_v2_16384_reconstruction/Full_seed1/predictions_best.npz` — SHA256 `be186a895a3bfa28603053bc713a06511030d72dd8c5fe9883e34a17e8f589cf`
- H2 primary prediction generic support; median: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/U_v2_16384_reconstruction/layout_agnostic_stratified_support_seed1/predictions_best.npz` — SHA256 `1f611b68f794aab8f301d89b4703aaae2e27520c16cf2f67d95c0e21ba09ff5c`
- H2 primary prediction CV-only support; median: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/U_v2_16384_reconstruction/cv_only_support_seed1/predictions_best.npz` — SHA256 `bcf0bec4c328f83f8e223672b88a18e1ceefc0d5de3d368e7d3c23df1d84cb91`
- H2 primary prediction Full; p90: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/U_v2_16384_reconstruction/Full_seed1/predictions_best.npz` — SHA256 `be186a895a3bfa28603053bc713a06511030d72dd8c5fe9883e34a17e8f589cf`
- H2 primary prediction generic support; p90: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/U_v2_16384_reconstruction/layout_agnostic_stratified_support_seed1/predictions_best.npz` — SHA256 `1f611b68f794aab8f301d89b4703aaae2e27520c16cf2f67d95c0e21ba09ff5c`
- H2 primary prediction CV-only support; p90: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/U_v2_16384_reconstruction/cv_only_support_seed1/predictions_best.npz` — SHA256 `bcf0bec4c328f83f8e223672b88a18e1ceefc0d5de3d368e7d3c23df1d84cb91`
- H2 primary prediction Full; p95: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/U_v2_16384_reconstruction/Full_seed2/predictions_best.npz` — SHA256 `53e6638ef12309d807aad523f7653d06090c7ac68547f2d23569baa6bc70d6b2`
- H2 primary prediction generic support; p95: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/U_v2_16384_reconstruction/layout_agnostic_stratified_support_seed2/predictions_best.npz` — SHA256 `351c107d635bbe4cf7de15feafe49b252517c3ba08250fa4b0e133c6aa34f951`
- H2 primary prediction CV-only support; p95: `/Users/xuyihua/Documents/heat3d-ic/v7_g1_formal_archive_sealed_20260903/h2_fullfield_240825_native/U_v2_16384_reconstruction/cv_only_support_seed2/predictions_best.npz` — SHA256 `e059046a1ec5703779f377c93bef9ce43cb660242ee1c2b4089590d53d736fba`
