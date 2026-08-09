# P1i GPU-only High-N protocol amendment

This amendment applies only to new GPU-only High-N development runs. The prior
4096 fail-closed result and its artifacts remain unchanged.

- Formal High-N backend: GPU.
- CPU/GPU real-edge topology equality: report-only diagnostic, not a hard gate.
- Hard gates: frozen valid32 and checkpoint; frozen 1024 anchor context/scale;
  same-GPU graph-cache hash; same-GPU cached/fresh prediction replay; fixed-input
  GPU replay; finite prediction; support/reconstruction hashes; physical support
  coverage and pre-resolution capacity/memory feasibility.
- Same-GPU replay tolerances: maximum absolute error `0.1 K`, RMSE `0.01 K`,
  scale drift `1e-6`. These gates apply after formal frozen-anchor-scale
  reconstruction. The discarded query-scale-head value remains report-only.
- Mandatory order: 4096, 8192, 16384. Optional order after all mandatory gates
  pass: 32768, then 65536.
- Accuracy is reported but never used to change the sampler, model, checkpoint,
  representation, or execution order.
- Access remains frozen valid32 only. Test, sealed IID, training, checkpoint
  modification, manifest modification, and three-seed valid128 are forbidden.

Machine-readable authority:
`configs/heat3d_v6_p1i/v6_p1i_gpu_only_high_n_protocol_amendment.json`.
