# Heat3D v3 S5 Targeted-Loss Ablation Results

Scope: read-only closeout over completed S5/P4/A/B/C checkpoint outputs. No training was started and no final-probe labels were regenerated.

| run | epoch | selection | valid_base | iid_err% | stress_base | stress_err% | raw RMSE | zRMSE | top-k | peak_rel | peak_abs | probe RMSE | probe rel | P02 | P03 | P09 | Tmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S5 base best | 1527 | valid_loss | 0.0210238 | 22.4119 | 0.0291828 | 23.8142 | 0.00282356 | 0.0662843 | 0.948828 | 0.0320597 | n/a | 0.372716 | 0.805463 | 0.631598 | 0.980921 | 0.547795 | n/a |
| S5final FT no-mask final | 100 | valid_loss | 0.0213502 | 22.5854 | 0.0289516 | 23.7225 | 0.0027353 | 0.0646257 | 0.951758 | 0.0312176 | n/a | 0.372366 | 0.804419 | 0.631203 | 0.980769 | 0.54755 | n/a |
| P4 mse-selected best | 83 | valid_base_mse | 0.0211334 | 22.4717 | 0.0288591 | 23.6816 | 0.00273058 | 0.0644722 | 0.952539 | 0.0309734 | n/a | 0.372598 | 0.805305 | 0.631104 | 0.980627 | 0.547066 | n/a |
| P4 mse-selected final | 100 | valid_base_mse | 0.0212424 | 22.5264 | 0.0289112 | 23.7029 | 0.00272915 | 0.0644472 | 0.952148 | 0.0306853 | n/a | 0.372563 | 0.805188 | 0.63109 | 0.980628 | 0.547071 | n/a |
| A hotspot-only best | 86 | valid_base_mse | 0.021128 | 22.4687 | 0.0288827 | 23.6906 | 0.00273322 | 0.0644192 | 0.952539 | 0.031241 | 0.0115521 | 0.372499 | 0.804856 | 0.631292 | 0.980796 | 0.547617 | -3.23824 |
| A hotspot-only final | 100 | valid_base_mse | 0.0212803 | 22.5477 | 0.0288583 | 23.679 | 0.0027257 | 0.0643361 | 0.952344 | 0.0310684 | 0.0115232 | 0.372405 | 0.804563 | 0.63119 | 0.980766 | 0.547577 | -3.238 |
| B hotspot0p05 strongq0p025 best | 71 | valid_base_mse | 0.0211461 | 22.4757 | 0.0289132 | 23.7045 | 0.00273068 | 0.0644745 | 0.95332 | 0.0309613 | 0.011507 | 0.37247 | 0.804778 | 0.631269 | 0.980775 | 0.547588 | -3.23837 |
| B hotspot0p05 strongq0p025 final | 100 | valid_base_mse | 0.0212645 | 22.539 | 0.0288992 | 23.6986 | 0.00272838 | 0.0644056 | 0.951758 | 0.0308295 | 0.0114983 | 0.372395 | 0.804554 | 0.631195 | 0.980731 | 0.547543 | -3.23817 |
| C hotspot0p025 strongq0p025 best | 1 | valid_base_mse | 0.0211266 | 22.4693 | 0.029078 | 23.7705 | 0.00282906 | 0.0649502 | 0.949805 | 0.0349656 | 0.0127942 | 0.372694 | 0.805403 | 0.631422 | 0.980943 | 0.547876 | -3.23951 |
| C hotspot0p025 strongq0p025 final | 100 | valid_base_mse | 0.0212646 | 22.5409 | 0.0288912 | 23.6941 | 0.00272786 | 0.0643914 | 0.951758 | 0.0307917 | 0.0114879 | 0.372391 | 0.804534 | 0.631195 | 0.980743 | 0.547537 | -3.23817 |

## Conclusions

- A/B/C did not beat S5 base best on valid_base_mse; S5 base best remains the scalar checkpoint reference in this table.
- A hotspot-only is the cleanest targeted-loss diagnostic because it isolates hotspot emphasis without the strong-q auxiliary term.
- B and C do not provide evidence that strong-q is the main gain source; strong-q is paused as a main ablation axis.
- Stop further hotspot/strong-q weight sweeps for now; move to LR escape, checkpoint origin, condition weighting, and model-path diagnostics.
- Final probe remains a side diagnostic only: it is not trained, not used to directly tune weights, and not treated as formal benchmark evidence.
- Every main candidate checkpoint should continue to run both best/final final-probe inference for comparability.
- P10 keeps the unsupported schema gap note: localized top contact / side asymmetry unsupported.

Ignored machine-readable mirror: `output/heat3d_v3_targeted_loss_audit/s5_targeted_loss_ablation_results.json`.
