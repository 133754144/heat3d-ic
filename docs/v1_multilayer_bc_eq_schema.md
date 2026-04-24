# v1 Multilayer BC Equivalent Demo Schema

## Subset

Subset path:

`subsets/v1_multilayer_bc_eq_demo/`

Recommended local sample path:

`data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_demo/samples/`

## Sample directory

Each sample directory is named `sample_xxx` and contains:

```text
coords.npy
layer_id.npy
region_id.npy
material_id.npy
k_field.npy
q_field.npy
sample_meta.json
```

For solver samples, the directory must also contain:

```text
temperature.npy
```

The current metadata-only smoke stage intentionally does not generate `temperature.npy`.

## Array schema

| File | Shape | Meaning | Unit |
| --- | --- | --- | --- |
| `coords.npy` | `(N, 3)` | sampled point coordinates | `m` |
| `layer_id.npy` | `(N,)` | layer identifier per point | unitless |
| `region_id.npy` | `(N,)` | region identifier per point | unitless |
| `material_id.npy` | `(N,)` | material / equivalent material identifier per point | unitless |
| `k_field.npy` | `(N, 1)`, `(N, 3)`, or `(N, 6)` | thermal conductivity field | `W/(m*K)` |
| `q_field.npy` | `(N, 1)` | volumetric heat generation | `W/m^3` |
| `temperature.npy` | `(N, 1)` | steady temperature field for solver samples | `K` |

`k_field` supports:

- `(N, 1)`: isotropic thermal conductivity
- `(N, 3)`: diagonal anisotropic thermal conductivity, ordered as `kx, ky, kz`
- `(N, 6)`: symmetric conductivity tensor, ordered as `kxx, kyy, kzz, kxy, kxz, kyz`

The first metadata-only smoke data only uses `(N, 1)`, but the schema supports all three forms.

`q_field` is always volumetric heat generation in `W/m^3`. If the source was derived from area power density or another representation, the conversion must be recorded in `generation_config`.

## Loader encoding policy

The current metadata-first loader keeps the default input mode as pure-physics:

- `coords`
- encoded `k_field`
- `q`
- BC encoding

Feature naming rules:

- native `(N, 1)` conductivity: `k_iso`
- diagonal conductivity: `k_x`, `k_y`, `k_z`
- volumetric heat generation: `q`
- BC encoding:
  - `is_top`
  - `is_bottom`
  - `is_side`
  - `is_interior`
  - `top_h`
  - `top_T_inf`
  - `bottom_T_fixed`

Current loader support:

- `k_encoding_mode="native"`
  - `(N,1)` stays `(N,1)`
  - `(N,3)` stays `(N,3)`
  - `(N,6)` stays `(N,6)`
- `k_encoding_mode="diag3"`
  - `(N,1)` expands to `(N,3)` as `[k, k, k]`
  - `(N,3)` stays `(N,3)`
  - `(N,6)` is not implemented yet in this mode

The current metadata-only samples only generate `(N,1)`. Future solver/train stages should at least support true `(N,3)` diagonal anisotropy.

The current main smoke set still uses `(N,1)` for `sample_000` through `sample_004`. `sample_005` is an additional diagnostic metadata-only sample that stores a real diagonal anisotropic `k_field` with shape `(N,3)`. No `(N,6)` sample is generated in the current batch.

## Metadata schema

`sample_meta.json` must contain:

```json
{
  "schema_version": "v1.0",
  "subset_name": "v1_multilayer_bc_eq_demo",
  "sample_id": "sample_000",
  "stage": "metadata_only",
  "split": "train",
  "domain": {},
  "layers": [],
  "regions": [],
  "materials": [],
  "boundary_regions": [],
  "boundary_types": {},
  "boundary_params": {},
  "interfaces": [],
  "generation_config": {},
  "units": {},
  "validation": {},
  "parameter_sources": {
    "literature_backed": [],
    "provisional_engineering_assumption": [],
    "requires_user_confirmation": []
  }
}
```

Allowed `stage` values:

- `metadata_only`
- `solver_smoke`
- `supervised_smoke`

Allowed `split` values:

- `train`
- `valid`
- `test_id`
- `test_ood_stack`
- `test_ood_bc`
- `test_ood_material`

The current first batch generates only:

- `train`
- `valid`
- `test_id`
- `test_ood_stack`

`test_ood_bc` is reserved for later held-out top Robin HTC experiments.

For the tiny parallel supervised smoke subset, `temperature.npy` should be treated as the supervised target / label for steady operator learning. It is not the intended mainline inference input.

## Boundary conditions

Boundary conditions must be described with separate fields:

- `boundary_regions`
- `boundary_types`
- `boundary_params`

The first-stage main BC setup is:

- `top`: Robin
- `bottom`: Dirichlet
- `sides`: adiabatic

Example:

```json
{
  "boundary_regions": [
    {"name": "top", "surface": "z_max"},
    {"name": "bottom", "surface": "z_min"},
    {"name": "sides", "surface": "x_or_y_minmax"}
  ],
  "boundary_types": {
    "top": "Robin",
    "bottom": "Dirichlet",
    "sides": "adiabatic"
  },
  "boundary_params": {
    "top": {"h_W_m2K": 2000.0, "ambient_temperature_K": 300.0},
    "bottom": {"fixed_temperature_K": 300.0},
    "sides": {"heat_flux_W_m2": 0.0}
  }
}
```

The loader must later convert these metadata fields into numerical model features, for example:

- point flags: `is_top`, `is_bottom`, `is_side`, `is_interior`
- broadcast parameters: `top_h`, `top_T_inf`, `bottom_T_fixed`

The current loader skeleton should default to a pure-physics input:

- `coords`
- `k_field`
- `q_field`
- BC encoding

`layer_id`, `region_id`, and `material_id` should remain available for metadata bookkeeping and optional auxiliary features, but they should not be required by the default input path.

## Interfaces

The first-stage interface type is:

- `perfect_contact`

Future schema versions may reserve:

- `contact_resistance`

Each interface must reference valid adjacent layer ids.

## Parameter-source tags

Important parameters must be classified in `parameter_sources`:

- `literature_backed`
- `provisional_engineering_assumption`
- `requires_user_confirmation`

Metadata-only smoke samples may use provisional values for shape, thermal conductivity, heat-generation amplitude, and Robin parameters, but they must not be presented as literature-verified values.
