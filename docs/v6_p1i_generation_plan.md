# V6-P1i staged generation plan

1. Freeze and commit the literature, background-k, sampling, and pilot
   acceptance contracts.
2. Generate exactly 128 Sobol cases without training or model inference.
3. Preserve every attempted config, seed, manifest, and rejection reason.
4. Audit solver residual, energy conservation, source/support coverage,
   DeltaT max/mean/CV-RMS histograms and KDEs, quantiles, gaps, modes, and
   parameter correlations.
5. Report the pilot result before deciding whether a 1024-sample protocol may
   be frozen. A passing pilot does not automatically generate 1024 cases.

The preregistered primary DeltaT-peak interval is 30--150 K, with a wider
20--180 K safety interval. The distribution gate checks continuous coverage
rather than forcing equal counts in four temperature bins. No model errors,
test labels, or learned predictions are used anywhere in this workflow.

