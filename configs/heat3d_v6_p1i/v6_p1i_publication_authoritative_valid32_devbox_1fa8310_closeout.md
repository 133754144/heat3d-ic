# Devbox authoritative valid32 benchmark

## Result

The devbox run completed all 30 independent cells: five routes, three randomized order seeds (`20260814`, `20260815`, `20260816`), and Serial/Q2 modes. The raw matrix and collector both passed; test and sealed roles remained closed.

The committed raw directory contains the 30 cell JSON/log pairs, the raw matrix, collector JSON/CSV/Markdown, actual-data preflight, root log, and a SHA manifest. The collector reports `publication_timing_freeze=GO` for this devbox measurement.

| Route | Fresh median (seed 20260814, s) | Fresh p95 (s) | Cache-hot median (s) | Resident median (s) | Q2 throughput (samples/s) | B16→B32 marginal (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|
| E16384 + reconstruction | 0.862344 | 0.887742 | 0.015581 | 0.005962 | 1.787960 | 0.558185 | 305813760 |
| U-v2 16384 + reconstruction | 0.810669 | 0.864056 | 0.012751 | 0.004146 | 1.855977 | 0.530551 | 310633216 |
| U-v2 direct240825 | 1.195633 | 1.434518 | 0.063660 | 0.054983 | 1.316916 | 0.761542 | 3995909632 |
| E direct240825 control | 1.198740 | 1.278541 | 0.099850 | 0.087347 | 1.443580 | 0.635957 | 4052184576 |
| FVM240825 reference | 1.421254 | 1.699151 | 1.386521 | 1.399985 | 0.932219 | 1.048153 | N/A |

The CSV/Markdown collector contains the complete three-seed median/range and paired bootstrap results. These headline values are devbox-only and are not mixed with the earlier WSL2 Attempt 4 table.

## Reproduction and provenance

- Branch/HEAD: `research/v6-p1i-training` / `1fa83103fa01dff604c1f377fcc6cd61cdf2ec4d`.
- Dataset manifest SHA: `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514`.
- Full-field archive SHA: `49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb`.
- Checkpoint SHA: `51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e` (epoch 559).
- Formal runner command and per-cell commands are preserved in `authoritative_valid32_raw.json`.
- Preflight SHA: `d3e448e1616a94472eec941543eda3dcc989a314ce9eec2f813ec2f93194b760`.
- Artifact manifest SHA: `6506be8a90fa01a39876fcb94f860c581eb3f7865e05b0f3022da0adb43f0002`.

The high-N binding file still records the earlier frozen implementation fingerprints for three exact-safe implementation files. The read-only preflight therefore used a `/tmp` copy with the current file hashes; the formal runner itself used the repository binding path. This is recorded explicitly in the machine-readable metadata and is not silently presented as a fresh binding gate.

Roles: no training, no test, no sealed data, no checkpoint or dataset modification.
