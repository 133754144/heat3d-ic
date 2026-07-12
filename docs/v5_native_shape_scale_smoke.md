# V5 Native Shape--Scale Smoke

## Result

- Frozen source: `V4P5_02_clean_baseline_raw_B28_e600` epoch `405`; only encoder/processor parameters were loaded.
- New shape decoder emits unnormalized `psi`; `phi_hat` is normalized independently per sample by control-volume RMS.
- Scale head uses inference-only global context and starts with zero residual around `log(s_phys)`.
- Target decomposition/reconstruction max error: `8.88178e-16 K`.
- Native reconstruction max error before projection: `0 K`; Dirichlet projection max error: `0 K`.
- B4 output shape: `[4, 1, 1024, 1]`; B1 output shape: `[1, 1, 1024, 1]`; finite gradient: `True`.

No target-derived value is accepted by the native prediction API. Targets are used only after inference for the four configured loss terms: shape CV, log scale, relative field, and raw absolute field.
