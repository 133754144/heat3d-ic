# V7-G2 P23 evidence hierarchy

## Tier 1 — V6 canonical historical evidence

Heat3D V6_06/V6_07/V6_08 remains the canonical P1i three-seed result. The seed1/seed2 binary checkpoints are permanently unavailable after the original WSL2 retirement, but the original checkpoint SHA, reload PASS, training receipts, and historical three-seed aggregate remain valid. This tier is not replayable and is not paired-bootstrap eligible.

## Tier 2 — V7 e200 replayable reference

V7 e200 is an independent 200-epoch dynamically scheduled cohort, not the first 200 epochs of V7 e600. All three best/final/latest checkpoints are present and SHA-verified, with exact reload evidence. Valid-only replay is complete on the DeepOHeat-v1 571,256-node cohort. It is a detailed Heat3D reference, not a replacement for V6 canonical P1i evidence.

The e200 cohort cannot be inserted into the P23 Heat3D-vs-Therm-FM paired comparison: its valid cases and output domain differ from the 240,825-node V6 P1i/Therm-FM domain. That comparison is therefore explicitly fail-closed rather than silently mixed.

## Tier 3 — fixed seed0 metric bridge

V6 seed0 was replayed previously under the P23 evaluator. Point-global relative RMSE, sample-first CV-relative RMSE, and peak-temperature RMSE are formula-equivalent to the historical evaluator within numerical roundoff. The historical CV-weighted raw RMSE is not equivalent to the current unweighted RMSE and is excluded from the bridge table.

## P1i valid-only sidecar closure (separate from V6 canonical replay)

The V7 G1 Full e200 x3 direct240825 comparison with Therm-FM is now closed on
the same 128 valid cases and 240,825-node domain. The common evaluator
reproduction gate passed, and the corrected case bootstrap recomputes the same
pooled sufficient-statistics estimators used by the aggregate table. This is a
valid-only sidecar/common-evaluator identity result, not a new checkpoint
inference reproduction: the earlier `d9d961a` checkpoint-to-P1i `FAIL_CLOSED`
receipt remains authoritative. The retained direct sidecars are bound by a
post-hoc receipt; the original G1 archive manifest is not rewritten.

Field-shape Corr/Amp/top-k and mechanism strata are secondary diagnostics only;
they do not alter the primary metric, checkpoint selection, or claim set. See
`docs/v7_g2_p1i_precloseout_status.md` and `docs/results/v7_g2_p1i_*`.

## P24 readiness options

No sealed evaluation is opened by P23 salvage.

* **Option A — fixed seed0 sealed confirmation:** Heat3D seed0, GINO seed0, Transolver seed0, and Therm-FM seed0, with a single evaluation-only unlock and frozen evaluator/configuration SHAs.
* **Option B — V7 e200 three-seed Heat3D sealed reference:** only after defining a symmetric cross-model cohort/domain contract. The current e200/DeepOHeat-v1 domain is not symmetric with the P1i Therm-FM domain, so this option needs an amendment before execution.

Neither option is selected or executed in this turn. The P1i valid-only
comparison is ready for review, but sealed evaluation remains locked.
