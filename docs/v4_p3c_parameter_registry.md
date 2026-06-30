# V4 P3c Parameter Registry

Read this file only for V4 P3c parameter ranges, source references, or registry
decisions. It is design-only and does not authorize data generation.

## Registry Rules

- The machine-readable mirror is `configs/heat3d_v4/p3c_parameter_registry.json`.
- Numeric ranges must point to source records with title, authors, year, venue,
  URL or DOI, and notes.
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
| SRC-DEEPOHEAT-2023 | DeepOHeat: Operator Learning-based Ultra-fast Thermal Simulation in 3D-IC Design | Ziyue Liu, Yixing Li, Jing Hu, Xinling Yu, Shinyu Shiau, Xin Ai, Zhiyu Zeng, Zheng Zhang | 2023 | arXiv | https://doi.org/10.48550/arXiv.2302.12949 | supports operator mapping from physical fields to temperature |
| SRC-DEEPOHEATV1-2025 | DeepOHeat-v1: Efficient Operator Learning for Fast and Trustworthy Thermal Simulation and Optimization in 3D-IC Design | Xinling Yu, Ziyue Liu, Hai Li, Yixing Li, Xin Ai, Zhiyu Zeng, Ian Young, Zheng Zhang | 2025 | IEEE TCPMT / arXiv | https://doi.org/10.48550/arXiv.2504.03955 | supports solver-audit and trustworthiness separation from training |
| SRC-BSPDN-2025 | Thermal Implications of Non-Uniform Power in BSPDN-Enabled 2.5D/3D Chiplet-based Systems-in-Package using Nanosheet Technology | Yukai Chen, Massimiliano Di Todaro, Bjorn Vermeersch, Herman Oprins, Daniele Jahier Pagliari, Julien Ryckaert, Dwaipayan Biswas, James Myers | 2025 | arXiv | https://doi.org/10.48550/arXiv.2508.02284 | supports localized synthetic power maps, 5 um power-map granularity, HTC examples, and layer k examples |
| SRC-SAUFNO-2025 | Self-Attention to Operator Learning-based 3D-IC Thermal Simulation | Zhen Huang, Hong Wang, Wenkai Yang, Muxi Tang, Depeng Xie, Ting-Jung Lin, Yu Zhang, Wei W. Xing, Lei He | 2025 | arXiv | https://doi.org/10.48550/arXiv.2510.15968 | supports high-frequency hotspot fields and same-distribution train/test splits |
| SRC-MULTISCALE-REVIEW-2026 | A Review of Multiscale Thermal Modeling in Heterogeneous 3D ICs | Baibhari Priya Barua, Md Rahatul Islam Udoy, Ahmedullah Aziz | 2026 | arXiv | https://doi.org/10.48550/arXiv.2604.03290 | supports TIM/TBR/anisotropy risk tracking and validation requirements |
| SRC-LIENHARD-2024 | A Heat Transfer Textbook, 6th edition | John H. Lienhard V, John H. Lienhard IV | 2024 | MIT open textbook | https://ahtt.mit.edu/ | supports heat-transfer coefficient and conduction boundary-condition terminology |
| SRC-HEAT3D-MEDIUM1024-2026 | V4 P1 Full Medium1024 Audit | Heat3D internal audit | 2026 | local doc | docs/v4_p1_full_medium1024_audit.md | solver-scale reference only; not a publication range source |
| SRC-HEAT3D-P3A-2026 | V4 P3a Closeout | Heat3D internal audit | 2026 | local doc | docs/v4_p3a_closeout.md | fixes V4 production contact model to R_contact=0_perfect_contact |

## k Registry Draft

| name | unit | range/default | sampling | source_ref | source_type | production | metadata_tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low_k_dielectric_underfill | W/m/K | 0.5 to 8.0 / 1.2 | log_uniform per block | SRC-BSPDN-2025, SRC-MULTISCALE-REVIEW-2026 | literature_informed_design_envelope | yes | k_class=low_k_dielectric |
| tim_effective | W/m/K | 3.0 to 30.0 / 8.0 | log_uniform per TIM-like block | SRC-BSPDN-2025, SRC-MULTISCALE-REVIEW-2026 | literature_informed_design_envelope | yes | k_class=tim_effective |
| effective_stack_medium_k | W/m/K | 5.0 to 80.0 / 30.0 | log_uniform background or block | SRC-BSPDN-2025, SRC-3DICE4-2025 | literature_informed_design_envelope | yes | k_class=effective_stack |
| silicon_like | W/m/K | 80.0 to 180.0 / 130.0 | log_uniform block or background | SRC-BSPDN-2025, SRC-3DICE4-2025 | literature_informed_design_envelope | yes | k_class=silicon_like |
| high_k_tsv_or_spreader | W/m/K | 150.0 to 430.0 / 230.0 | log_uniform bridge/block | SRC-BSPDN-2025, SRC-3DICE4-2025 | literature_informed_design_envelope | yes | k_class=high_k_tsv_spreader |
| diag3_anisotropy_ratio | ratio | 0.5 to 2.0 / [1.0, 1.0, 1.0] | bounded ratio around scalar base k | SRC-3DICE4-2025, SRC-MULTISCALE-REVIEW-2026 | literature_informed_design_envelope | yes, if checker-ready | k_mode=diag3 |

## q Registry Draft

All q ranges are volumetric Heat3D generator densities in W/m^3. The shape
families are literature-driven; the absolute ranges are initial Heat3D scale
envelopes that must be validated by P3c smoke DeltaT and energy audits.

| name | unit | range/default | sampling | source_ref | source_type | production | metadata_tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact_hotspot_q_density | W/m^3 | 5.0e7 to 2.0e8 / 1.0e8 | log_uniform, 1 to 3 compact blocks | SRC-BSPDN-2025, SRC-SAUFNO-2025, SRC-HEAT3D-MEDIUM1024-2026 | literature_motif_plus_heat3d_scale_reference | yes | q_family=compact_hotspot |
| multi_block_q_density | W/m^3 | 2.0e7 to 1.5e8 / 8.0e7 | log_uniform, 2 to 6 blocks | SRC-BSPDN-2025, SRC-SAUFNO-2025, SRC-HEAT3D-MEDIUM1024-2026 | literature_motif_plus_heat3d_scale_reference | yes | q_family=multi_block |
| elongated_q_density | W/m^3 | 1.0e7 to 1.2e8 / 5.0e7 | log_uniform elongated strip | SRC-BSPDN-2025, SRC-SAUFNO-2025 | literature_motif_plus_heat3d_scale_reference | yes | q_family=elongated |
| weak_background_q_density | W/m^3 | 1.0e6 to 5.0e6 / 3.0e6 | log_uniform full-volume or broad block | SRC-BSPDN-2025, SRC-HEAT3D-MEDIUM1024-2026 | literature_motif_plus_heat3d_scale_reference | yes | q_family=weak_background |
| weak_background_hotspot_q_density | W/m^3 | 5.0e7 to 2.0e8 / 1.0e8 | log_uniform hotspot over weak background | SRC-BSPDN-2025, SRC-HEAT3D-MEDIUM1024-2026 | literature_motif_plus_heat3d_scale_reference | yes | q_family=background_plus_hotspot |
| dual_z_q_density | W/m^3 | 2.0e7 to 1.5e8 / 8.0e7 | log_uniform sources in two z bands | SRC-DEEPOHEAT-2023, SRC-SAUFNO-2025 | literature_motif_plus_heat3d_scale_reference | yes | q_family=dual_z |
| tsv_adjacent_q_density | W/m^3 | 3.0e7 to 1.8e8 / 9.0e7 | log_uniform near high-k vertical path | SRC-BSPDN-2025, SRC-3DICE4-2025 | literature_motif_plus_heat3d_scale_reference | yes | q_family=tsv_adjacent |

## BC Registry Draft

| name | unit | range/default | sampling | source_ref | source_type | production | metadata_tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top_robin_h | W/m^2/K | 300.0 to 3000.0 / 1200.0 | log_uniform per sample | SRC-BSPDN-2025, SRC-LIENHARD-2024, SRC-HEAT3D-MEDIUM1024-2026 | literature_informed_design_envelope | yes | bc=top_robin_h |
| bottom_dirichlet_temperature | K | 300.0 to 300.0 / 300.0 | fixed in P3c production | SRC-LIENHARD-2024, SRC-HEAT3D-MEDIUM1024-2026 | physics_bc_contract | yes | bc=bottom_dirichlet_T |
| top_ambient_temperature | K | 300.0 to 300.0 / 300.0 | fixed in P3c production | SRC-LIENHARD-2024, SRC-HEAT3D-MEDIUM1024-2026 | physics_bc_contract | yes | bc=top_ambient_T |
| bc_flag_channels | binary | 0 to 1 / geometry-derived | deterministic from coordinates | SRC-DEEPOHEAT-2023, SRC-HEAT3D-MEDIUM1024-2026 | model_input_contract | yes | bc_flags=top_bottom_side_interior |
| side_boundary_model | enum | adiabatic / adiabatic | fixed in P3c production | SRC-LIENHARD-2024, SRC-HEAT3D-P3A-2026 | physics_bc_contract | yes | bc=side_adiabatic |

## Contact Registry Draft

| name | unit | range/default | sampling | source_ref | source_type | production | metadata_tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| production_contact_resistance | m^2*K/W | 0.0 to 0.0 / 0.0 | fixed | SRC-HEAT3D-P3A-2026 | production_gate | yes | contact_model=R_contact_0_perfect_contact |
| finite_contact_resistance_deferred | m^2*K/W | deferred / deferred | no production sampling | SRC-MULTISCALE-REVIEW-2026, SRC-HEAT3D-P3A-2026 | implemented_deferred_smoke_only | no | contact_model=finite_R_deferred |

## Split And Audit Policy

P3c generated samples should be assigned to train/test using a fixed seed random
split from the unified distribution. The generation pipeline must audit
train/test coverage after generation and reject the dataset only for generator,
solver, metadata, or distribution-balance failures. It must not create a stress
split or treat final_probe as a pass/fail gate.
