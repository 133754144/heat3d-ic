# V6 main merge instructions

These commands are for a later, explicitly approved merge after final audit.
They are not executed by the V6 governance closeout.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git merge --no-ff origin/research/v6-p1h-shared-support
python3 scripts/check_heat3d_v6_total_governance.py
python3 scripts/check_heat3d_v6_hard_ood_closeout.py
python3 scripts/check_heat3d_v6_merge_readiness.py
git diff --check HEAD^
```

Before pushing `main`, verify the merge commit contains the same P1h manifest,
full-field archive, canonical seed0 checkpoint, source-aware ladder, and
governance hashes recorded in
`configs/heat3d_v6/v6_total_governance_manifest.json`. Do not resolve conflicts
by regenerating data, changing checkpoint selection, or rewriting historical
results.
