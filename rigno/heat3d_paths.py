"""Shared local path helpers for Heat3D data."""

from __future__ import annotations

from pathlib import Path


CANONICAL_DATA_SUBDIR = Path(
  "data/heat3d-thermal-simulation/subsets/v0_unitcube_demo/samples"
)
LEGACY_DATA_SUBDIR = Path("dataset_3d_heat")


def has_heat3d_samples(path: Path) -> bool:
  """Returns True when a directory contains Heat3D sample_xxx folders."""

  return path.is_dir() and any(
    child.is_dir() and child.name.startswith("sample_")
    for child in path.iterdir()
  )


def _normalize_path(path: Path, repo_dir: Path) -> Path:
  path = path.expanduser()
  return path if path.is_absolute() else (repo_dir / path)


def resolve_heat3d_data_dir(
  data_dir: str | Path | None = None,
  repo_dir: str | Path | None = None,
) -> Path:
  """Resolves the Heat3D sample directory.

  The preferred local layout mirrors the Hugging Face dataset repo:

    data/heat3d-thermal-simulation/subsets/v0_unitcube_demo/samples/

  The legacy root-level dataset_3d_heat/ directory is still accepted as a
  fallback for existing local checkouts.
  """

  root = Path(repo_dir).resolve() if repo_dir is not None else Path(__file__).resolve().parents[1]

  if data_dir is not None:
    candidate = _normalize_path(Path(data_dir), root)
    if has_heat3d_samples(candidate):
      return candidate

    samples_dir = candidate / "samples"
    if has_heat3d_samples(samples_dir):
      return samples_dir

    return candidate

  for relative_path in (CANONICAL_DATA_SUBDIR, LEGACY_DATA_SUBDIR):
    candidate = root / relative_path
    if has_heat3d_samples(candidate):
      return candidate

  return root / CANONICAL_DATA_SUBDIR
