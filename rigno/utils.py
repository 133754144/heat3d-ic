"""General utility functions."""

from typing import Union, Sequence

import flax.typing
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np


Array = Union[jnp.ndarray, np.ndarray]

def shuffle_arrays(key: flax.typing.PRNGKey, arrays: Sequence[Array], axis: int = 0) -> Sequence[Array]:
  """Shuffles a set of arrays with the same random permutation along the given axis."""

  # Move the desired axis to the leading axis
  arrays = tree.tree_map(lambda v: jnp.moveaxis(v, axis, 0), arrays)

  # Get permutation
  length = arrays[0].shape[0]
  assert all(tree.tree_map(lambda v: v.shape[0] == length, arrays))
  permutation = jax.random.permutation(key, length)

  # Permute along the leading axis
  arrays = tree.tree_map(lambda v: v[permutation], arrays)
  # Move back the leading axis to its place
  arrays = tree.tree_map(lambda v: jnp.moveaxis(v, 0, axis), arrays)

  return arrays

def normalize(arr: Array, shift: Array, scale: Array):
  """Normalizes a given array by shifting and scaling it."""

  scale = jnp.where(scale == 0., 1., scale)
  arr = (arr - shift) / scale
  return arr

def unnormalize(arr: Array, mean: Array, std: Array):
  """Reverts the shift-scale normalization."""

  arr = std * arr + mean
  return arr
