# V4P4 Config Plan

Read this file only for V4P4_01-06 registry/YAML launch planning.

## Baseline

V4P4 starts from the resolved semantics of `configs/heat3d_v4/generated/V4P3_14.yaml`.
The tracked registry still generates inherited YAML from `V4_base.yaml`; V4P3_14
is the semantic baseline recorded in each config's notes.

Motivation: `V4P3_19-final` is the current formal split-level best, but
`rel_rmse_v4_pct` is dominated by a small number of P02-like /
`physical_hard_keep` high-DeltaT hotspot and strong-q samples. V4P4 therefore
tests hotspot, strong-q, and hard-sample weighting recipes first.

All V4P4_01-06 entries are `status=planned` with
`launch_policy=explicit_user_instruction_only`. Do not launch them without a
fresh explicit user instruction.
Generated inherited YAML omits default-valued keys that match `V4_base.yaml`;
the resolved registry CSV still records those values.

## Planned Configs

| config_id | short label | only changed from baseline recipe |
| --- | --- | --- |
| V4P4_01 | hotspot_x2 | `epochs=220`, `hotspot_weight=0.10`, `strong_q_weight=0.05`; paths and IDs use `V4P4_01`. |
| V4P4_02 | strongq_x2 | `epochs=220`, `hotspot_weight=0.05`, `strong_q_weight=0.10`; paths and IDs use `V4P4_02`. |
| V4P4_03 | hotspot_strongq_x2 | `epochs=220`, `hotspot_weight=0.10`, `strong_q_weight=0.10`; paths and IDs use `V4P4_03`. |
| V4P4_04 | physical_hard_weighted | V4P4_03 plus `sample_weight_policy=hard_sample_list`, tracked weight JSON, default weight `1.0`, normalize `true`; paths and IDs use `V4P4_04`. |
| V4P4_05 | fourier_hotspot_strongq | V4P4_03 plus `node_coordinate_encoding=raw_plus_fourier`, `node_coordinate_freqs=4`; paths and IDs use `V4P4_05`. |
| V4P4_06 | upstream_onecycle_hotspot_strongq | V4P4_03 plus `lr_schedule=upstream_onecycle`, `lr_init=1.0e-5`, `lr_peak=2.0e-4`, `lr_base=1.0e-5`, `lr_lowr=1.0e-6`, `pct_start=0.02`, `pct_final=0.10`; paths and IDs use `V4P4_06`. |

## Sample Weights

`configs/heat3d_v4/sample_weights/v4p4_physical_hard_keep_weight_v0.json`
selects formal train split samples with `qc_class=physical_hard_keep`.

- selected train samples: 121
- selected weights: `2.0`
- default weight for all other train samples: `1.0`
- recommended normalization: `true`
- reason counts: low-k enclosed compact hotspot 80, multi-source or high-power
  bottleneck 23, weak cooling 18

Validation and test samples are not weighted by this JSON; the runner applies
sample weighting only to train loss.

## No-Run Status

This plan only prepares registry entries, CSV mirror, generated YAML, and a
tracked sample-weight JSON. No training, tmux session, remote command, or output
artifact is part of this step.
