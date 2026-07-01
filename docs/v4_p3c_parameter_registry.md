# V4 P3c Parameter Registry

Read this file only for V4 P3c parameter ranges, source references, or registry
decisions. It is design-only and does not authorize data generation.

## Registry Rules

- The machine-readable mirror is `configs/heat3d_v4/p3c_parameter_registry.json`.
- This document describes intent; enforceable constraints live in the JSON
  registry, P3c dry-run generator, and checker.
- P3c-2 validates dry-run scenes and array-preview contracts only. It must not
  write a dataset or call the solver.
- P3c-2b validates real in-memory arrays and metadata only. It must not write a
  dataset or call the solver.
- Numeric ranges must point to source records with title, authors, year, venue,
  URL or DOI, and notes.
- k entries use `literature_anchor`, `sampling_envelope`, and `rationale`.
- q entries record `source_volume_fraction`, `integrated_power_target`, and
  `DeltaT_target_bin`.
- k, q, and top_h ranges must not be hard-coded from final_probe extremes.
- q topology is literature-driven; q magnitude is an initial Heat3D
  solver-scale design envelope that must pass P3c smoke/audit before expansion.
- V4 production contact is fixed to `R_contact=0_perfect_contact`.
- Finite `R_contact` is implemented/deferred and excluded from V4 production
  dataset/default solver.

## Source Records

| id | title | authors | year | venue | URL or DOI | notes |
| --- | --- | --- | ---: | --- | --- | --- |
| SRC-3DICE4-2025 | 3D-ICE 4.0: Accurate and efficient thermal modeling for 2.5D/3D heterogeneous chiplet systems | Kai Zhu, Darong Huang, Luis Costero, David Atienza | 2025 | arXiv | https://doi.org/10.48550/arXiv.2512.05823 | supports heterogeneous/anisotropic materials and vertical thermal paths |
| SRC-HBM-MEAS-2023 | Thermal Conductivity Measurement of High Bandwidth Memory | Darshan Chalise, David G. Cahill | 2023 | arXiv measurement preprint | https://doi.org/10.48550/arXiv.2303.06785 | anchors HBM anisotropy: in-plane about 100/140 W/m/K and through-plane about 7/2 W/m/K |
| SRC-DEEPOHEAT-2023 | DeepOHeat: Operator Learning-based Ultra-fast Thermal Simulation in 3D-IC Design | Ziyue Liu, Yixing Li, Jing Hu, Xinling Yu, Shinyu Shiau, Xin Ai, Zhiyu Zeng, Zheng Zhang | 2023 | arXiv | https://doi.org/10.48550/arXiv.2302.12949 | supports operator mapping from physical fields to temperature |
| SRC-DEEPOHEATV1-2025 | DeepOHeat-v1: Efficient Operator Learning for Fast and Trustworthy Thermal Simulation and Optimization in 3D-IC Design | Xinling Yu, Ziyue Liu, Hai Li, Yixing Li, Xin Ai, Zhiyu Zeng, Ian Young, Zheng Zhang | 2025 | IEEE TCPMT / arXiv | https://doi.org/10.48550/arXiv.2504.03955 | supports solver-audit and trustworthiness separation from training |
| SRC-BSPDN-2025 | Thermal Implications of Non-Uniform Power in BSPDN-Enabled 2.5D/3D Chiplet-based Systems-in-Package using Nanosheet Technology | Yukai Chen, Massimiliano Di Todaro, Bjorn Vermeersch, Herman Oprins, Daniele Jahier Pagliari, Julien Ryckaert, Dwaipayan Biswas, James Myers | 2025 | arXiv | https://doi.org/10.48550/arXiv.2508.02284 | supports chiplet/interposer dimensions, 200/2500 W/m2/K HTC anchors, layer k/thickness, and non-uniform power maps |
| SRC-SAUFNO-2025 | Self-Attention to Operator Learning-based 3D-IC Thermal Simulation | Zhen Huang, Hong Wang, Wenkai Yang, Muxi Tang, Depeng Xie, Ting-Jung Lin, Yu Zhang, Wei W. Xing, Lei He | 2025 | arXiv | https://doi.org/10.48550/arXiv.2510.15968 | supports high-frequency hotspot fields and same-distribution train/test splits |
| SRC-LIENHARD-2024 | A Heat Transfer Textbook, 6th edition | John H. Lienhard V, John H. Lienhard IV | 2024 | MIT open textbook | https://ahtt.mit.edu/ | supports heat-transfer coefficient and conduction boundary-condition terminology |
| SRC-HEAT3D-MEDIUM1024-2026 | V4 P1 Full Medium1024 Audit | Heat3D internal audit | 2026 | local doc | docs/v4_p1_full_medium1024_audit.md | solver-scale reference only; not a publication range source |
| SRC-HEAT3D-P3A-2026 | V4 P3a Closeout | Heat3D internal audit | 2026 | local doc | docs/v4_p3a_closeout.md | fixes V4 production contact model to R_contact=0_perfect_contact |

## k Registry Draft

| name | unit | literature_anchor | sampling_envelope | default | source_ref | source_type | rationale | production | metadata_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_k_dielectric_underfill | W/m/K | low-k package/underfill materials | 0.5 to 8.0 | 1.2 | SRC-BSPDN-2025 | literature_informed_design_envelope | cover low-k barriers without using final_probe as the range source | yes | k_class=low_k_dielectric |
| tim_effective | W/m/K | TIM/effective thermal path materials | 3.0 to 30.0 | 8.0 | SRC-BSPDN-2025 | literature_informed_design_envelope | represent intermediate package/TIM conduction blocks | yes | k_class=tim_effective |
| effective_stack_medium_k | W/m/K | effective chiplet/interposer stack layers | 5.0 to 80.0 | 30.0 | SRC-BSPDN-2025, SRC-3DICE4-2025 | literature_informed_design_envelope | provide non-extreme background and block materials | yes | k_class=effective_stack |
| silicon_like | W/m/K | silicon-like layer conductivity | 80.0 to 180.0 | 130.0 | SRC-BSPDN-2025, SRC-3DICE4-2025 | literature_informed_design_envelope | cover die-like material fields and bridges | yes | k_class=silicon_like |
| high_k_tsv_or_spreader | W/m/K | TSV/spreader/high-k path material | 150.0 to 430.0 | 230.0 | SRC-BSPDN-2025, SRC-3DICE4-2025 | literature_informed_design_envelope | cover vertical escape and lateral spreading paths | yes | k_class=high_k_tsv_spreader |
| hbm_like_anisotropic_k | W/m/K | in-plane about 100/140; through-plane about 7/2 | diag3 components from 2.0 to 140.0 | [100.0, 100.0, 7.0] | SRC-HBM-MEAS-2023, SRC-3DICE4-2025 | measurement_anchor_plus_sampling_envelope | force anisotropic thermal paths into the random-block family | yes | k_class=hbm_like_diag3 |
| diag3_anisotropy_ratio | ratio | anisotropic material support | 0.5 to 2.0 around scalar base k | [1.0, 1.0, 1.0] | SRC-3DICE4-2025, SRC-HBM-MEAS-2023 | literature_informed_design_envelope | production v0 targets 20% diag3 samples; checker must support it | yes | k_mode=diag3 |

## q Registry Draft

All q ranges are volumetric Heat3D generator densities in W/m3. The shape
families are literature-driven; the absolute ranges are initial Heat3D scale
envelopes that must be validated by P3c smoke DeltaT and energy audits.

| name | unit | range/default | source_volume_fraction | integrated_power_target | DeltaT_target_bin | sampling | source_ref | source_type | production | metadata_tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact_hotspot_q_density | W/m3 | 5.0e7 to 2.0e8 / 1.0e8 | 0.002 to 0.03 | 0.02 to 1.20 W | nominal_to_hard | log_uniform, 1 to 3 compact blocks | SRC-BSPDN-2025, SRC-SAUFNO-2025, SRC-HEAT3D-MEDIUM1024-2026 | literature_motif_plus_heat3d_scale_reference | yes | q_family=compact_hotspot |
| multi_block_q_density | W/m3 | 2.0e7 to 1.5e8 / 8.0e7 | 0.005 to 0.08 | 0.05 to 2.00 W | nominal_to_hard | log_uniform, 2 to 6 blocks | SRC-BSPDN-2025, SRC-SAUFNO-2025, SRC-HEAT3D-MEDIUM1024-2026 | literature_motif_plus_heat3d_scale_reference | yes | q_family=multi_block |
| elongated_q_density | W/m3 | 1.0e7 to 1.2e8 / 5.0e7 | 0.01 to 0.08 | 0.05 to 1.50 W | low_to_nominal | log_uniform elongated strip | SRC-BSPDN-2025, SRC-SAUFNO-2025 | literature_motif_plus_heat3d_scale_reference | yes | q_family=elongated |
| weak_background_q_density | W/m3 | 1.0e6 to 5.0e6 / 3.0e6 | 0.40 to 1.00 | 0.20 to 1.00 W | low | log_uniform broad background | SRC-BSPDN-2025, SRC-HEAT3D-MEDIUM1024-2026 | literature_motif_plus_heat3d_scale_reference | yes | q_family=weak_background |
| weak_background_hotspot_q_density | W/m3 | 5.0e7 to 2.0e8 / 1.0e8 | 0.002 to 0.03 hotspot over background | 0.25 to 2.00 W | nominal_to_hard | log_uniform hotspot over weak background | SRC-BSPDN-2025, SRC-HEAT3D-MEDIUM1024-2026 | literature_motif_plus_heat3d_scale_reference | yes | q_family=background_plus_hotspot |
| dual_z_q_density | W/m3 | 2.0e7 to 1.5e8 / 8.0e7 | 0.004 to 0.06 | 0.05 to 2.00 W | nominal_to_hard | log_uniform sources in two z bands | SRC-DEEPOHEAT-2023, SRC-SAUFNO-2025 | literature_motif_plus_heat3d_scale_reference | yes | q_family=dual_z |
| tsv_adjacent_q_density | W/m3 | 3.0e7 to 1.8e8 / 9.0e7 | 0.002 to 0.03 | 0.05 to 1.50 W | nominal_to_hard | log_uniform near high-k vertical path | SRC-BSPDN-2025, SRC-3DICE4-2025 | literature_motif_plus_heat3d_scale_reference | yes | q_family=tsv_adjacent |

## Cooling Regimes

| name | unit | range/default | literature_anchor | source_ref | source_type | rationale | metadata_tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weak_effective_air | W/m2/K | 200 to 500 / 300 | BSPDN uses 200 W/m2/K low-HTC anchor | SRC-BSPDN-2025, SRC-LIENHARD-2024 | literature_anchor_plus_design_envelope | preserve weak-cooling samples without relying on final_probe | cooling=weak_effective_air |
| nominal_package | W/m2/K | 500 to 1500 / 1000 | package-level effective cooling between anchors | SRC-BSPDN-2025, SRC-LIENHARD-2024, SRC-HEAT3D-MEDIUM1024-2026 | literature_anchor_plus_heat3d_reference | center production distribution near current stable solver scale | cooling=nominal_package |
| strong_forced_or_effective_heatsink | W/m2/K | 1500 to 3000 / 2500 | BSPDN uses 2500 W/m2/K high-HTC anchor | SRC-BSPDN-2025, SRC-LIENHARD-2024 | literature_anchor_plus_design_envelope | include strong cooling without turning final_probe into a hard source | cooling=strong_forced_or_effective_heatsink |

## BC Registry Draft

| name | unit | range/default | sampling | source_ref | source_type | production | metadata_tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top_robin_h | W/m2/K | 200 to 3000 / by cooling regime | regime mixture from cooling registry | SRC-BSPDN-2025, SRC-LIENHARD-2024 | cooling_regime_mixture | yes | bc=top_robin_h |
| bottom_dirichlet_temperature | K | 300.0 to 300.0 / 300.0 | fixed in P3c production | SRC-LIENHARD-2024, SRC-HEAT3D-MEDIUM1024-2026 | physics_bc_contract | yes | bc=bottom_dirichlet_T |
| top_ambient_temperature | K | 300.0 to 300.0 / 300.0 | fixed in P3c production | SRC-LIENHARD-2024, SRC-HEAT3D-MEDIUM1024-2026 | physics_bc_contract | yes | bc=top_ambient_T |
| bc_flag_channels | binary | 0 to 1 / geometry-derived | deterministic from coordinates | SRC-DEEPOHEAT-2023, SRC-HEAT3D-MEDIUM1024-2026 | model_input_contract | yes | bc_flags=top_bottom_side_interior |
| side_boundary_model | enum | adiabatic / adiabatic | fixed in P3c production | SRC-LIENHARD-2024, SRC-HEAT3D-P3A-2026 | physics_bc_contract | yes | bc=side_adiabatic |

## Geometry And Block Registry

| name | unit | range/default | sampling | source_ref | source_type | rationale | metadata_tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| domain_xy_mm | mm | 5.0 to 20.0 / 10.0 | fixed v0 or bounded uniform later | SRC-BSPDN-2025, SRC-HEAT3D-MEDIUM1024-2026 | literature_anchor_plus_heat3d_reference | keep smoke comparable to current 10 mm domain while allowing chiplet-scale planning | geometry=domain_xy_mm |
| domain_z_mm | mm | 0.5 to 3.0 / 2.0 | fixed v0 or bounded uniform later | SRC-BSPDN-2025, SRC-HEAT3D-MEDIUM1024-2026 | literature_anchor_plus_heat3d_reference | preserve thin 3D stack aspect ratio and current solver comparability | geometry=domain_z_mm |
| grid_shape_candidates | nodes | [16,16,4], [32,32,4], [32,32,8] / [16,16,4] | discrete choice by stage | SRC-3DICE4-2025, SRC-HEAT3D-MEDIUM1024-2026 | solver_stage_envelope | start at existing 1024-node compatibility before scaling | geometry=grid_shape |
| material_block_count | count | 1 to 12 / 4 | discrete uniform | SRC-3DICE4-2025, SRC-SAUFNO-2025 | design_envelope_literature_motif | enough heterogeneity for random-block learning without smoke instability | geometry=material_block_count |
| material_block_xy_fraction | fraction | 0.05 to 0.60 / 0.20 | log_uniform or beta | SRC-BSPDN-2025, SRC-SAUFNO-2025 | design_envelope_literature_motif | cover local barriers, bridges, and broad spreaders | geometry=material_block_xy_fraction |
| material_block_z_fraction | fraction | 0.25 to 1.00 / 0.50 | discrete/bounded uniform | SRC-3DICE4-2025, SRC-HBM-MEAS-2023 | design_envelope_literature_motif | allow through-stack and partial-stack material paths | geometry=material_block_z_fraction |
| source_block_count | count | 1 to 8 / 3 | discrete by q family | SRC-BSPDN-2025, SRC-SAUFNO-2025 | design_envelope_literature_motif | support compact, multi-block, and dual-z power maps | geometry=source_block_count |
| source_block_xy_fraction | fraction | 0.02 to 0.40 / 0.08 | log_uniform or beta | SRC-BSPDN-2025, SRC-SAUFNO-2025 | design_envelope_literature_motif | represent localized hotspots and elongated sources | geometry=source_block_xy_fraction |
| source_block_z_fraction | fraction | 0.25 to 1.00 / 0.50 | discrete/bounded uniform | SRC-DEEPOHEAT-2023, SRC-SAUFNO-2025 | design_envelope_literature_motif | support layer-local and multi-layer heat sources | geometry=source_block_z_fraction |
| placement_policy | enum | random_block_ic_motif_mixture / random_block_ic_motif_mixture | weighted categorical | SRC-BSPDN-2025, SRC-3DICE4-2025 | design_policy | mix IC motifs into random blocks instead of stack-template interpolation | geometry=placement_policy |
| overlap_policy | enum | k_blocks_may_overlap_q_blocks / k_blocks_may_overlap_q_blocks | explicit metadata | SRC-DEEPOHEAT-2023, SRC-SAUFNO-2025 | design_policy | allow heat sources to sit inside, beside, or across material blocks | geometry=overlap_policy |

## Expected DeltaT Distribution

| bin | rule | source_ref | source_type | rationale |
| --- | --- | --- | --- | --- |
| reject_low | deltaT_peak_K < 0.02 K | SRC-HEAT3D-MEDIUM1024-2026 | generator_qc_initial_bin | labels this small are likely numerically valid but uninformative for early training |
| low | 0.02 to 0.2 K | SRC-HEAT3D-MEDIUM1024-2026 | generator_qc_initial_bin | preserve low-amplitude cases without letting them dominate |
| nominal | 0.2 to 2.0 K | SRC-HEAT3D-MEDIUM1024-2026 | generator_qc_initial_bin | preferred production mass before P3c-3/4 calibration |
| hard | 2.0 to 8.0 K | SRC-HEAT3D-MEDIUM1024-2026 | generator_qc_initial_bin | retain difficult but plausible cases if solver audit passes |
| review_high | 8.0 to 15.0 K | SRC-HEAT3D-MEDIUM1024-2026 | generator_qc_initial_bin | manual review band until smoke/pilot calibration |
| reject_high | deltaT_peak_K > 15.0 K | SRC-HEAT3D-MEDIUM1024-2026 | generator_qc_initial_bin | reject or downscale to avoid uncalibrated amplitude outliers |

## Production Mix

| name | unit | range/default | source_ref | source_type | rationale | metadata_tag |
| --- | --- | --- | --- | --- | --- | --- |
| diag3_target_fraction | fraction | 0.20 to 0.20 / 0.20 | SRC-3DICE4-2025, SRC-HBM-MEAS-2023 | v4_design_requirement | production v0 must include anisotropic samples; generator/checker must support diag3 | production_mix=diag3_target_fraction |
| q_family_uniformity | fraction | each active family at least 0.10 / balanced | SRC-BSPDN-2025, SRC-SAUFNO-2025 | design_policy | avoid a dataset dominated by one hotspot topology | production_mix=q_family_uniformity |
| cooling_regime_min_fraction | fraction | each regime at least 0.15 / balanced | SRC-BSPDN-2025, SRC-LIENHARD-2024 | design_policy | guarantee weak, nominal, and strong cooling coverage | production_mix=cooling_regime_min_fraction |

## Array Synthesis Policies

| policy | required behavior |
| --- | --- |
| `background_k_policy` | default background is `effective_stack_medium_k`; allowed background families are `effective_stack_medium_k`, `silicon_like`, and `hbm_like_anisotropic_k`; `low_k_dielectric_underfill` is not the default background |
| `k_overlap_policy` | `deterministic_priority_override`: initialize background k for every node, apply blocks in deterministic order, keep final k only, and record `covered_by_blocks` plus `winning_block_id` |
| `q_overlap_policy` | `sum_volumetric_sources`: overlapping q blocks sum per cell; max pooling is forbidden for generator q merge |
| `power_calibration_policy` | calibrate q density from solver control-volume weighted realized block volume and integrated-power target; q audit uses the same weights as the solver |

## Background k Reference Values

These are semiconductor substrate/material anchors for background selection,
not final_probe-derived hard ranges.

| background family | reference values | sources | use |
| --- | --- | --- | --- |
| `effective_stack_medium_k` | 10/30/60 W/m/K suggested composite anchors | SRC-BSPDN-2025, SRC-3DICE4-2025, SRC-SAUFNO-2025 | default background for equivalent stack/substrate/interposer composites |
| `silicon_like` | 100 to 150 W/m/K | SRC-SAUFNO-2025, SRC-BSPDN-2025 | allowed background for silicon-like die/substrate scenes |
| `hbm_like_anisotropic_k` | in-plane 100/140 W/m/K, through-plane 7/2 W/m/K | SRC-HBM-MEAS-2023, SRC-3DICE4-2025 | allowed diag3 background when anisotropic arrays are active |
| `low_k_dielectric_underfill` | 0.5 to 8 W/m/K | SRC-BSPDN-2025, SRC-SAUFNO-2025 | minority background or block-only low-k barrier; never default background |

## Contact Registry Draft

| name | unit | range/default | sampling | source_ref | source_type | production | metadata_tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| production_contact_resistance | m2*K/W | 0.0 to 0.0 / 0.0 | fixed | SRC-HEAT3D-P3A-2026 | production_gate | yes | contact_model=R_contact_0_perfect_contact |
| finite_contact_resistance_deferred | m2*K/W | deferred / deferred | no production sampling | SRC-3DICE4-2025, SRC-HEAT3D-P3A-2026 | implemented_deferred_smoke_only | no | contact_model=finite_R_deferred |

## Split And Audit Policy

P3c generated samples should be assigned to train/test using a fixed seed random
split from the unified distribution. The generation pipeline must audit
train/test coverage after generation and reject the dataset only for generator,
solver, metadata, distribution-balance, or DeltaT-QC failures. It must not
create a stress split or treat final_probe as a pass/fail gate.
