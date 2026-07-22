# V6-P1f unified layered dataset closeout

## Outcome

`heat3d_v6_p1f_unified_layered1024_v0` is the qualified formal V6-layer
dataset.  It is a new version; P1d and P1e remain immutable provenance.

The global contract was frozen from a 128-case pilot using top h
`1000/1400 W/(m2 K)`, bottom h `20/120 W/(m2 K)`, and package power `4/6 W`.
The pilot passed on its first preregistered whole batch, so no second contract
attempt or sample-level adjustment was made.  None of the 16 pilot geometry
groups or 128 pilot samples appears in the final dataset.

## Final structure

- 128 independent geometry groups and eight complete 2x2x2 BC/power cases per
  group;
- train: 96 groups / 768 cases;
- valid: 16 groups / 128 cases;
- test: 16 groups / 128 cases;
- no OOD roles;
- every split has the same BC, power, source-count, layout, and fixed-material
  distributions;
- source count 3--10, each at probability 0.125 in every split;
- four layout families, each at probability 0.25 in every split;
- randomized source area, aspect ratio, clustering, inter-layer alignment,
  upper/lower power fraction, and within-layer area-weighted power allocation.

## Gate and integrity

- below 30 K: `0`;
- 30--80 K: `1024/1024` (`100%`);
- above 100 K: `0`;
- above 120 K: `0`;
- peak DeltaT min/median/max: `30.055/46.353/77.816 K`;
- BC--power Pearson and Spearman correlations: `0` in train, valid, and test;
- minimum source resolution: `270` control volumes and `7` in-plane intervals;
- maximum q: `5.688889e9 W/m3`;
- aggregate package power / total source area: `10.035--38.562 W/cm2`
  (median `19.721 W/cm2`);
- maximum single-source power: `3 W`;
- maximum surface power density: `60.749 W/cm2`;
- maximum absolute energy-balance relative error: `1.203e-10`;
- every sample covers all package layers and interfaces in the group-frozen
  1024-point projection.

The gate was applied to the complete 1024-case version.  No Rth power
inversion, temperature filtering, resampling, replacement, local repair,
model training, or model inference occurred.

## Provenance SHA256

- P1d config: `58cff515dc6af27b2b262535101318c01069ff84788a9c45c17efd6339502fcc`;
- P1e config: `8d1448005a2afb3267c891dfb5660cf5d6e2ea3e9ca6bce6abee755b3f1ae1e3`;
- P1f pilot config: `aee804b94e25164297d55c0f2b8f59415ca77aa51ff91249ba094300bfbb649f`;
- P1f pilot manifest: `6fcd300b2ebcd1ce3051fb07d7dbd88bc22dcb408ddc1047379af960ccedc00e`;
- P1f final config: `6b05c889760675954300428066f3ff6a12109725073bf0edb336fc9eb04e0fda`;
- P1f final manifest: `fd311b9b8c19b1f578f2cbc7c8322826766d22bfc75b5067820799abd34c2e03`.

Raw generated arrays remain under ignored `data/` directories; tracked
configs, manifests, split maps, audits, reports, and checkers reproduce and
verify the dataset contract.
