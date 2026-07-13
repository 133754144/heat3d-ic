from typing import Tuple, Union, NamedTuple

from flax import linen as nn
import flax.typing
import jax.numpy as jnp
import jax.random
import jraph
import numpy as np
from scipy.spatial import Delaunay

from rigno.graph.entities import (TypedGraph, EdgeSet, EdgeSetKey,
  EdgesIndices, NodeSet, Context)
from rigno.models.graphnet import DeepTypedGraphNet
from rigno.models.operator import AbstractOperator, Inputs
from rigno.utils import Array, shuffle_arrays


class RegionInteractionGraphSet(NamedTuple):
  """The set of the graphs that are used in RIGNO."""

  #: Graph connecting the physical nodes to the regional nodes
  p2r: TypedGraph
  #: Graph containing bi-directional edges in the regional mesh
  r2r: TypedGraph
  #: Graph connecting the regional nodes to the physical nodes
  r2p: TypedGraph

  def __len__(self) -> int:
    return self.p2r.nodes['pnodes'].n_node.shape[0]

class RegionInteractionGraphMetadata(NamedTuple):
  """Light-weight class for storing graph metadata."""

  x_pnodes_inp: Array
  x_pnodes_out: Array
  x_rnodes: Array
  r_rnodes: Array
  p2r_edge_indices: Array
  r2r_edge_indices: Array
  r2r_edge_domains: Array
  r2p_edge_indices: Array

  def __len__(self) -> int:
    return self.x_pnodes_inp.shape[0]

class RegionInteractionGraphBuilder:
  """Class for building the graphs that are used in RIGNO."""

  def __init__(self,
    periodic: bool,
    rmesh_levels: int,
    subsample_factor: float,
    overlap_factor_p2r: float,
    overlap_factor_r2p: float,
    node_coordinate_freqs: int,
    node_coordinate_encoding: str = "raw",
    coverage_repair_policy: str = "none",
    radius_policy: str = "discrete_physical_coverage",
    repair_p2r: bool = True,
    repair_r2p: bool = True,
    min_physical_coverage: int = 1,
  ):
    """
    Class for building the graphs that are used in RIGNO.

    Args:
        periodic: If True, periodic boundary conditions are considered
          in defining the edges.
        rmesh_levels: Number of times that the physical nodes are
          downsampled for defining the edges in the r2r graph.
        subsample_factor: Factor for spatial downsampling of the nodes.
        overlap_factor_p2r: Factor by which the minimum support-regions
          in the p2r graph get multiplied to.
        overlap_factor_r2p: Factor by which the minimum support-regions
          in the r2p graph get multiplied to.
        node_coordinate_encoding: Node coordinate feature encoding. "raw"
          preserves the normalized coordinates; "raw_plus_fourier" appends
          non-periodic Fourier features to the raw coordinates.
        node_coordinate_freqs: Number of frequencies for encoding the
          normalized spatial coordinates when node_coordinate_encoding is
          "raw_plus_fourier".
        coverage_repair_policy: Optional post-radius graph repair. The legacy
          default is "none"; "nearest_rnode" adds nearest regional edges only
          for physical nodes below min_physical_coverage.
        radius_policy: Support-radius policy. The v4 default is
          "discrete_physical_coverage"; "legacy_kdtree_mean4" preserves the
          earlier KDTree mean-4 radius policy for explicit ablation/control.
        repair_p2r: Apply nearest-rnode repair to p2r when repair is enabled.
        repair_r2p: Apply nearest-rnode repair to r2p when repair is enabled.
        min_physical_coverage: Minimum physical-node degree targeted by repair.
    """

    if coverage_repair_policy not in {"none", "nearest_rnode"}:
      raise ValueError(
        "coverage_repair_policy must be one of {'none', 'nearest_rnode'}, "
        f"found {coverage_repair_policy!r}"
      )
    if radius_policy not in {"legacy_kdtree_mean4", "discrete_physical_coverage"}:
      raise ValueError(
        "radius_policy must be one of "
        "{'legacy_kdtree_mean4', 'discrete_physical_coverage'}, "
        f"found {radius_policy!r}"
      )
    if min_physical_coverage < 1:
      raise ValueError("min_physical_coverage must be at least 1")
    if node_coordinate_encoding not in {"raw", "raw_plus_fourier"}:
      raise ValueError(
        "node_coordinate_encoding must be one of {'raw', 'raw_plus_fourier'}, "
        f"found {node_coordinate_encoding!r}"
      )
    if int(node_coordinate_freqs) < 1:
      raise ValueError("node_coordinate_freqs must be at least 1")

    # Set attributes
    self.periodic = periodic
    self.overlap_factor_p2r = overlap_factor_p2r
    self.overlap_factor_r2p = overlap_factor_r2p
    self.node_coordinate_encoding = node_coordinate_encoding
    self.node_coordinate_freqs = int(node_coordinate_freqs)
    self.rmesh_levels = rmesh_levels
    self.subsample_factor = subsample_factor
    self.coverage_repair_policy = coverage_repair_policy
    self.radius_policy = radius_policy
    self.repair_p2r = bool(repair_p2r)
    self.repair_r2p = bool(repair_r2p)
    self.min_physical_coverage = int(min_physical_coverage)

    # Domain shifts for periodic BC
    # self._domain_shifts = jnp.concatenate([
    #   jnp.array([[0., 0.]]),  # C
    #   jnp.array([[-2, 0.]]),  # W
    #   jnp.array([[-2, +2]]),  # NW
    #   jnp.array([[0., +2]]),  # N
    #   jnp.array([[+2, +2]]),  # NE
    #   jnp.array([[+2, 0.]]),  # E
    #   jnp.array([[+2, -2]]),  # SE
    #   jnp.array([[0., -2]]),  # S
    #   jnp.array([[-2, -2]]),  # SW
    # ], axis=0)
    self._domain_shifts = jnp.array([
      [dx, dy, dz] for dx in [-2, 0, 2]
                for dy in [-2, 0, 2]
                for dz in [-2, 0, 2]
    ])

  def _coordinate_node_features(self, x: Array) -> Array:
    """Return node coordinate features independent from boundary periodicity."""

    if self.node_coordinate_encoding == "raw":
      return x
    if self.node_coordinate_encoding != "raw_plus_fourier":
      raise ValueError(
        "node_coordinate_encoding must be one of {'raw', 'raw_plus_fourier'}, "
        f"found {self.node_coordinate_encoding!r}"
      )
    phi = jnp.pi * (x + 1)  # train_minmax_to_unit_box coords [-1, 1] -> [0, 2pi]
    freqs = jnp.arange(
      1,
      self.node_coordinate_freqs + 1,
      dtype=phi.dtype,
    )
    angles = jnp.expand_dims(phi, axis=-1) * freqs
    sin_feats = jnp.sin(angles).reshape(*x.shape[:-1], -1)
    cos_feats = jnp.cos(angles).reshape(*x.shape[:-1], -1)
    return jnp.concatenate([x, sin_feats, cos_feats], axis=-1)

  def _compute_minimum_support_radius(self, x: Array) -> Array:
      """
      Returns the minimum radius of the support sub-region of each regional node.
      By considering the neighnor nodes, it ensures that the union of all support
      sub-regions covers the whole domain.
      """
      # NOTE: This function is not jittable because of the Delaunay triangulation

      # if self.periodic:
      #   # Repeat the domain in all directions before constructing a triangulation
      #   x_extended = (x[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
      #   tri = Delaunay(points=x_extended)
      # else:
      #   tri = Delaunay(points=x)

      # medians = _compute_triangulation_medians(tri)
      # radii = np.zeros(shape=(x.shape[0],))
      # mask = tri.simplices < x.shape[0] # [N, 3]
      # values = medians[mask]
      # indices = tri.simplices[mask]
      # sorted_idx = np.argsort(indices)
      # sorted_indices = indices[sorted_idx]
      # sorted_values = values[sorted_idx]
      # unique_indices, idx_start = np.unique(sorted_indices, return_index=True)
      # radii[unique_indices] = np.maximum.reduceat(sorted_values, idx_start)

      # return radii
       
      """3D 版本（您的 UnitCube 是非周期性，用这个最快最稳）"""
      x = np.asarray(x)
      from scipy.spatial import KDTree
      tree = KDTree(x)
      radii = np.zeros(len(x))
      for i in range(len(x)):
        dist, _ = tree.query(x[i], k=5)   # self + 4 近邻
        radii[i] = np.mean(dist[1:]) * 0.8
      return jnp.array(radii)

  def _get_physical_to_regional_distance(self,
    centers: Array,
    points: Array,
    ord_distance: int = 2,
  ) -> np.ndarray:
    """Returns physical-to-regional distances in the graph coordinate metric."""

    rel = np.asarray(points)[:, None] - np.asarray(centers)
    if self.periodic:
      rel = np.where(rel >= 1., (rel - 2.), rel)
      rel = np.where(rel < -1., (rel + 2.), rel)
    return np.linalg.norm(rel, ord=ord_distance, axis=-1)

  def _compute_discrete_physical_coverage_radius(self,
    centers: Array,
    points: Array,
  ) -> Array:
    """Returns radii covering each regional node's assigned physical nodes."""

    distance = self._get_physical_to_regional_distance(centers=centers, points=points)
    nearest_rnodes = np.argmin(distance, axis=1)
    nearest_distance = distance[np.arange(distance.shape[0]), nearest_rnodes]
    radii = np.zeros(shape=(np.asarray(centers).shape[0],), dtype=distance.dtype)
    np.maximum.at(radii, nearest_rnodes, nearest_distance)
    radii = np.nextafter(radii, np.asarray(np.inf, dtype=radii.dtype))
    return jnp.asarray(radii)

  def _get_effective_support_radii(self,
    r_rnodes: Array,
    overlap_factor: float,
  ) -> Array:
    """Returns final radii before physical-to-regional edge construction."""

    if self.radius_policy == "discrete_physical_coverage":
      # This monotone policy deliberately bypasses the legacy global clip and
      # non-monotone hard reset. It uses the minimal 1x coverage radius to
      # retain the guarantee without inheriting the legacy overlap edge cost.
      return r_rnodes
    return jnp.clip(overlap_factor * r_rnodes, a_min=0, a_max=r_rnodes.max())

  def _get_supported_pnodes_by_rnodes(self,
    centers: Array,
    points: Array,
    radii: Array,
    ord_distance: int = 2,
    apply_legacy_hard_reset: bool = True,
  ) -> Array:
    """
    Get the indices of the physical nodes that lie in the support sub-region of
    each regional node.

    Arguments:
      centers: The coordinates of the regional nodes.
      points: The coordinates of the physical nodes.
      radii: The support radius of each regional node.
      ord_distance: The order of the norm for defining the
        support sub-region of a regional node. Typical values
        are 1, 2, and np.inf

    Returns:
      The indices of the physical nodes for each regional node.
    """

    if apply_legacy_hard_reset:
      # Replace large radii
      # NOTE: Makeshift solution for peculiar geometries
      # TODO: Instead, remove out-of-domain mesh edges in order to avoid large radiuses
      radii = np.where(radii < .5, radii, .2)

    # Get relative coordinates
    rel = points[:, None] - centers
    # Mirror relative positions because of periodic boudnary conditions
    if self.periodic:
      rel = jnp.where(rel >= 1., (rel - 2.), rel)
      rel = jnp.where(rel < -1., (rel + 2.), rel)

    # Compute distance
    # NOTE: Order of the norm determines the shape of the sub-regions
    distance = jnp.linalg.norm(rel, ord=ord_distance, axis=-1)

    # Get indices
    # -> [idx_point, idx_center]
    idx_nodes = jnp.stack(jnp.where(distance <= radii), axis=-1)

    return idx_nodes

  def _repair_physical_node_coverage(self,
    edge_indices: Array,
    centers: Array,
    points: Array,
  ) -> Array:
    """Adds nearest regional edges for physical nodes below the repair target."""

    n_rnodes = int(np.asarray(centers).shape[0])
    if self.min_physical_coverage > n_rnodes:
      raise ValueError(
        "min_physical_coverage cannot exceed the number of regional nodes: "
        f"{self.min_physical_coverage} > {n_rnodes}"
      )

    edges = np.asarray(edge_indices)
    degree = np.bincount(edges[:, 0], minlength=np.asarray(points).shape[0])
    nodes_to_repair = np.flatnonzero(degree < self.min_physical_coverage)
    if nodes_to_repair.size == 0:
      return edge_indices

    distance = self._get_physical_to_regional_distance(centers=centers, points=points)
    nearest_order = np.argsort(distance, axis=1)
    existing = {(int(pnode), int(rnode)) for pnode, rnode in edges}
    additions = []
    for pnode in nodes_to_repair:
      needed = self.min_physical_coverage - int(degree[pnode])
      for rnode in nearest_order[pnode]:
        edge = (int(pnode), int(rnode))
        if edge in existing:
          continue
        additions.append(edge)
        existing.add(edge)
        needed -= 1
        if needed == 0:
          break

    if not additions:
      return edge_indices
    return jnp.concatenate(
      [edge_indices, jnp.asarray(additions, dtype=edge_indices.dtype)],
      axis=0,
    )

  def _get_r2r_edges(self, x_rmesh: Array) -> Tuple[Array, Array]:
    """
    Defines the edges of the r2r graph (processor graph).

    Arguments:
      x_rmesh: Coordinates of the regional nodes.

    Returns:
      The edges (pair of node indices) and the index of the corresponding
      (extended) domain of the source and destination nodes.
    """

    # Define edges and their corresponding -extended- domain
    edges = []
    domains = []
    for level in range(self.rmesh_levels):
      # Sub-sample the rmesh
      _rmesh_size = int(x_rmesh.shape[0] / (self.subsample_factor ** level))
      if _rmesh_size < 4:
        continue
      _x_rmesh = x_rmesh[:_rmesh_size]
      # Construct a triangulation
      if self.periodic:
        # Repeat the rmesh in periodic directions
        _x_rmesh_extended = (_x_rmesh[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
        tri = Delaunay(points=_x_rmesh_extended)
      else:
        tri = Delaunay(points=_x_rmesh)
      # Get the relevant edges
      _extended_edges = _get_edges_from_triangulation(tri)
      domains_level = _extended_edges // _rmesh_size
      edges_level = _extended_edges % _rmesh_size
      idx_relevant_edges = np.any(domains_level == 0, axis=1) if self.periodic else np.all(domains_level == 0, axis=1)
      edges_level = edges_level[idx_relevant_edges]
      domains_level = domains_level[idx_relevant_edges]
      edges.append(edges_level)
      domains.append(domains_level)

    # Remove repeated edges
    edges = jnp.concatenate(edges)
    domains = jnp.concatenate(domains)
    _, unique_idx = jnp.unique(edges, axis=0, return_index=True)
    edges = edges[unique_idx]
    domains = domains[unique_idx]

    return edges, domains

  def build_metadata(self, x_inp: Array, x_out: Array, domain: Array, rmesh_correction_dsf: int = 1, key: Union[flax.typing.PRNGKey, None] = None) -> RegionInteractionGraphMetadata:
    """Returns the metadata that is needed for building all RIGNO graphs."""

    # Normalize coordinates in [-1, +1) —— 归一化
    x_inp = 2 * (x_inp - domain[0]) / (domain[1] - domain[0]) - 1
    x_out = 2 * (x_out - domain[0]) / (domain[1] - domain[0]) - 1

    # Randomly sub-sample pmesh to get rmesh —— 随机采样
    if key is None: key = jax.random.PRNGKey(0)
    x_rnodes = _subsample_pointset(key=key, x=x_inp, factor=self.subsample_factor)

    # Downsample or upsample the rmesh —— 参数rmesh_correction_dsf不为1时微调网格密度
    if rmesh_correction_dsf > 1:
      x_rnodes = _subsample_pointset(key=key, x=x_rnodes, factor=rmesh_correction_dsf)
    elif rmesh_correction_dsf < 1:
      x_rnodes = _upsample_pointset(key=key, x=x_rnodes, factor=(1 / rmesh_correction_dsf))

    # Compute minimum support radius of each rmesh node —— 计算每个区域节点的“最小支持半径”
    if self.radius_policy == "discrete_physical_coverage":
      coverage_points = jnp.concatenate([x_inp, x_out], axis=0)
      r_rnodes = self._compute_discrete_physical_coverage_radius(
        centers=x_rnodes,
        points=coverage_points,
      )
    else:
      r_rnodes = self._compute_minimum_support_radius(x_rnodes)

    # Get edge indices
    p2r_edge_indices = self._get_supported_pnodes_by_rnodes(
      centers=x_rnodes, # 区域节点坐标
      points=x_inp,     # 物理节点坐标
      radii=self._get_effective_support_radii(r_rnodes, self.overlap_factor_p2r),
      apply_legacy_hard_reset=(self.radius_policy == "legacy_kdtree_mean4"),
    ) # 返回哪些物理点连到哪些区域点
    r2r_edge_indices, r2r_edge_domains = self._get_r2r_edges(x_rnodes)
    r2p_points = (
      x_inp
      if (
        self.radius_policy == "legacy_kdtree_mean4"
        and self.coverage_repair_policy == "none"
      )
      else x_out
    )
    r2p_edge_indices = self._get_supported_pnodes_by_rnodes(
      centers=x_rnodes,
      points=r2p_points,
      radii=self._get_effective_support_radii(r_rnodes, self.overlap_factor_r2p),
      apply_legacy_hard_reset=(self.radius_policy == "legacy_kdtree_mean4"),
    )

    if self.coverage_repair_policy == "nearest_rnode":
      if self.repair_p2r:
        p2r_edge_indices = self._repair_physical_node_coverage(
          edge_indices=p2r_edge_indices,
          centers=x_rnodes,
          points=x_inp,
        )
      if self.repair_r2p:
        r2p_edge_indices = self._repair_physical_node_coverage(
          edge_indices=r2p_edge_indices,
          centers=x_rnodes,
          points=x_out,
        )
    r2p_edge_indices = jnp.flip(r2p_edge_indices, axis=-1)

    # Add dummy nodes and edges
    p2r_edge_indices = jnp.concatenate([p2r_edge_indices, jnp.array([[x_inp.shape[0], x_rnodes.shape[0]]])], axis=0)
    r2r_edge_indices = jnp.concatenate([r2r_edge_indices, jnp.array([[x_rnodes.shape[0], x_rnodes.shape[0]]])], axis=0)
    r2r_edge_domains = jnp.concatenate([r2r_edge_domains, jnp.array([[0, 0]])], axis=0)
    r2p_edge_indices = jnp.concatenate([r2p_edge_indices, jnp.array([[x_rnodes.shape[0], x_out.shape[0]]])], axis=0)
    x_inp = jnp.concatenate([x_inp, jnp.zeros(shape=(1, x_inp.shape[-1]))], axis=0)
    x_out = jnp.concatenate([x_out, jnp.zeros(shape=(1, x_out.shape[-1]))], axis=0)
    x_rnodes = jnp.concatenate([x_rnodes, jnp.zeros(shape=(1, x_rnodes.shape[-1]))], axis=0)
    r_rnodes = jnp.concatenate([r_rnodes, jnp.zeros(shape=(1,))], axis=0)

    # Convert dtypes to save memory
    r2r_edge_domains = r2r_edge_domains.astype(jnp.uint8)
    if (max(x_inp.shape[0], x_out.shape[0]) < jnp.iinfo(jnp.uint16).max):
      p2r_edge_indices=p2r_edge_indices.astype(jnp.uint16)
      r2r_edge_indices=r2r_edge_indices.astype(jnp.uint16)
      r2p_edge_indices=r2p_edge_indices.astype(jnp.uint16)
    # Ommit storing duplicated edge indices
    if (
      self.overlap_factor_p2r == self.overlap_factor_r2p
      and np.array_equal(
        np.asarray(r2p_edge_indices),
        np.flip(np.asarray(p2r_edge_indices), axis=-1),
      )
    ):
      # NOTE: it will be the inverse of p2r edges
      r2p_edge_indices = None

    # Store the graph data
    graph_metadata = RegionInteractionGraphMetadata(
      x_pnodes_inp=jnp.expand_dims(x_inp, axis=0),
      x_pnodes_out=jnp.expand_dims(x_out, axis=0),
      x_rnodes=jnp.expand_dims(x_rnodes, axis=0),
      r_rnodes=jnp.expand_dims(r_rnodes, axis=0),
      p2r_edge_indices=jnp.expand_dims(p2r_edge_indices, axis=0),
      r2r_edge_indices=jnp.expand_dims(r2r_edge_indices, axis=0),
      r2r_edge_domains=jnp.expand_dims(r2r_edge_domains, axis=0),
      r2p_edge_indices=(jnp.expand_dims(r2p_edge_indices, axis=0) if (r2p_edge_indices is not None) else None),
    )

    return graph_metadata

  def _init_structural_features(self,
    x_sen: Array,
    x_rec: Array,
    idx_sen: Array,
    idx_rec: Array,
    max_edge_length: float,
    feats_sen: Array = None,
    feats_rec: Array = None,
    shift: bool = False,
    domain_sen: Array = None,
    domain_rec: Array = None,
  ) -> Tuple[EdgeSet, NodeSet, NodeSet]:
    """
    Creates the edge set and the node sets of a graph. The edge and node feature vectors
    are initialized with the structural features that are computed based on the coordinates
    of the nodes.

    Args:
        x_sen: The coordiantes of the sender nodes.
        x_rec: The coordiantes of the receiver nodes.
        idx_sen: The indices of the sender nodes in edges.
        idx_rec: The indices of the receiver nodes in edges.
        max_edge_length: Maximum possible edge length that is used for normalization.
        feats_sen: Forced (structural) features of the sender nodes. Defaults to None.
        feats_rec: Forced (structural) features of the receiver nodes. Defaults to None.
        shift: If True, the long cross-boundary edge lengths are replaced with equivalent
          short lengths. This operation only makes sense for periodic boundary conditions.
          Defaults to False.
        domain_sen: Index of the (extended) domain of the sender nodes. Defaults to None.
        domain_rec: Index of the (extended) domain of the receiver nodes. Defaults to None.

    Returns:
        Edge set, sender node set, and the receiver node set.
    """

    # Get number of nodes and the edges
    batch_size = x_sen.shape[0]
    num_sen = x_sen.shape[1]
    num_rec = x_rec.shape[1]
    assert idx_sen.shape[1] == idx_rec.shape[1]
    num_edg = idx_sen.shape[1]

    # Define node features. Coordinate encoding is independent from periodic
    # boundary handling; periodic only affects edge distance/wrap logic below.
    sender_node_feats = self._coordinate_node_features(x_sen)
    receiver_node_feats = self._coordinate_node_features(x_rec)
    # Concatenate with forced features
    if feats_sen is not None:
      sender_node_feats = jnp.concatenate([sender_node_feats, feats_sen], axis=-1)
    if feats_rec is not None:
      receiver_node_feats = jnp.concatenate([receiver_node_feats, feats_rec], axis=-1)

    # Build node sets
    sender_node_set = NodeSet(
      n_node=jnp.tile(jnp.array([num_sen]), reps=(batch_size, 1)),
      features=sender_node_feats,
    )
    receiver_node_set = NodeSet(
      n_node=jnp.tile(jnp.array([num_rec]), reps=(batch_size, 1)),
      features=receiver_node_feats,
    )

    # Define edge features
    batched_index = jax.vmap(lambda f, idx: f[idx])
    batched_index_single = jax.vmap(lambda f, idx: f[idx], in_axes=(None, 0))
    z_ij = batched_index(x_sen, idx_sen) - batched_index(x_rec, idx_rec)
    if self.periodic:
      if not shift:
        # NOTE: For p2r and r2p, mirror the large relative coordinates
        # MODIFY: Unify the mirroring with the below method in r2r
        z_ij = jnp.where(z_ij < -1.0, z_ij + 2, z_ij)
        z_ij = jnp.where(z_ij >= 1.0, z_ij - 2, z_ij)
      else:
        # NOTE: For the r2r multi-mesh, use extended domain indices and shifts
        z_ij = (
          (batched_index(x_sen, idx_sen) + batched_index_single(self._domain_shifts, domain_sen))
          - (batched_index(x_rec, idx_rec) + batched_index_single(self._domain_shifts, domain_rec))
        )
    d_ij = jnp.linalg.norm(z_ij, axis=-1, keepdims=True)
    # Normalize and concatenate edge features
    z_ij = z_ij / max_edge_length
    d_ij = d_ij / max_edge_length
    edge_feats = jnp.concatenate([z_ij, d_ij], axis=-1)

    # Build edge set
    edge_set = EdgeSet(
      n_edge=jnp.tile(jnp.array([num_edg]), reps=(batch_size, 1)),
      indices=EdgesIndices(
        senders=idx_sen,
        receivers=idx_rec,
      ),
      features=edge_feats,
    )

    return edge_set, sender_node_set, receiver_node_set

  def _build_p2r_graph(self, x_pnodes: Array, x_rnodes: Array, idx_edges: Array, r_rmesh: Array) -> TypedGraph:
    """Constructs the encoder graph (pmesh to rmesh)"""

    # Get the initial features
    edge_set, pmesh_node_set, rmesh_node_set = self._init_structural_features(
      x_sen=x_pnodes,
      x_rec=x_rnodes,
      idx_sen=idx_edges[..., 0],
      idx_rec=idx_edges[..., 1],
      max_edge_length=(2. * jnp.sqrt(x_rnodes.shape[-1])),
      feats_rec=jnp.expand_dims(self.overlap_factor_p2r * r_rmesh, axis=-1),
    )

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.tile(jnp.array([1]), reps=(x_rnodes.shape[0], 1)), features=()),
      nodes={'pnodes': pmesh_node_set, 'rnodes': rmesh_node_set},
      edges={EdgeSetKey('p2r', ('pnodes', 'rnodes')): edge_set},
    )

    return graph

  def _build_r2r_graph(self, x_rnodes: Array, idx_edges: Array, idx_domains: Array, r_rmesh: Array) -> TypedGraph:
    """Constructs the processor graph (rmesh to rmesh)"""

    # Set the initial features
    edge_set, rmesh_node_set, _ = self._init_structural_features(
      x_sen=x_rnodes,
      x_rec=x_rnodes,
      idx_sen=idx_edges[..., 0],
      idx_rec=idx_edges[..., 1],
      max_edge_length=(2. * jnp.sqrt(x_rnodes.shape[-1])),
      feats_sen=jnp.expand_dims(self.overlap_factor_p2r * r_rmesh, axis=-1),
      feats_rec=jnp.expand_dims(self.overlap_factor_r2p * r_rmesh, axis=-1),
      shift=True,
      domain_sen=idx_domains[..., 0],
      domain_rec=idx_domains[..., 1],
    )

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.tile(jnp.array([1]), reps=(x_rnodes.shape[0], 1)), features=()),
      nodes={'rnodes': rmesh_node_set},
      edges={EdgeSetKey('r2r', ('rnodes', 'rnodes')): edge_set},
    )

    return graph

  def _build_r2p_graph(self, x_pnodes: Array, x_rnodes: Array, idx_edges: Array, r_rmesh: Array) -> TypedGraph:
    """Constructs the decoder graph (rmesh to pmesh)"""

    # Get the initial features
    edge_set, rmesh_node_set, pmesh_node_set = self._init_structural_features(
      x_sen=x_rnodes,
      x_rec=x_pnodes,
      idx_sen=idx_edges[..., 0],
      idx_rec=idx_edges[..., 1],
      max_edge_length=(2. * jnp.sqrt(x_rnodes.shape[-1])),
      feats_sen=jnp.expand_dims(self.overlap_factor_r2p * r_rmesh, axis=-1),
    )

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.tile(jnp.array([1]), reps=(x_rnodes.shape[0], 1)), features=()),
      nodes={'pnodes': pmesh_node_set, 'rnodes': rmesh_node_set},
      edges={EdgeSetKey('r2p', ('rnodes', 'pnodes')): edge_set},
    )

    return graph

  def build_graphs(self, metadata: RegionInteractionGraphMetadata) -> RegionInteractionGraphSet:
    """Constructs all the graphs that are used by RIGNO by using the necessary pre-computed metadata."""

    # Unwrap the attributes
    x_pnodes_inp = metadata.x_pnodes_inp
    x_pnodes_out = metadata.x_pnodes_out
    x_rnodes = metadata.x_rnodes
    r_rnodes = metadata.r_rnodes
    p2r_edge_indices = metadata.p2r_edge_indices
    r2r_edge_indices = metadata.r2r_edge_indices
    r2r_edge_domains = metadata.r2r_edge_domains
    r2p_edge_indices = metadata.r2p_edge_indices
    # Flip p2r indices if r2p is None
    if r2p_edge_indices is None:
      r2p_edge_indices = jnp.flip(metadata.p2r_edge_indices, axis=-1)

    # Build the graphs
    graphs = RegionInteractionGraphSet(
      p2r=self._build_p2r_graph(x_pnodes_inp, x_rnodes, p2r_edge_indices, r_rnodes),
      r2r=self._build_r2r_graph(x_rnodes, r2r_edge_indices, r2r_edge_domains, r_rnodes),
      r2p=self._build_r2p_graph(x_pnodes_out, x_rnodes, r2p_edge_indices, r_rnodes),
    )

    return graphs

class Encoder(nn.Module):
  """Encoder block of RIGNO.

  Args:
    node_latent_size: Dimension of the latent node features.
    edge_latent_size: Dimension of the latent edge features.
    mlp_hidden_layers: Number of hidden layers in the MLPs.
    use_layer_norm: Whether to use LayerNorm layers.
    conditioned_normalization: Whether to use conditioned normalization layers.
    cond_norm_hidden_size: Hidden size for the shallow MLP used for
      computing shift and scales in the conditioned normalization layers.
    p_edge_masking: Probability of masking an edge.
  """

  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    self.gnn = DeepTypedGraphNet(
      embed_nodes=True,  # Embed raw features of all nodes
      embed_edges=True,  # Embed raw features of the edges
      edge_latent_size=dict(p2r=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size, pnodes=self.node_latent_size),
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=1,
      use_layer_norm=self.use_layer_norm,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=True,
      aggregate_edges_for_nodes_fn=jraph.segment_mean,
    )

  def __call__(self,
    graph: TypedGraph,
    pnode_features: Array,
    tau: Union[None, float],
    key: Union[flax.typing.PRNGKey, None] = None,
  ) -> tuple[Array, Array]:
    """Runs the p2r GNN, extracting latent physical and regional nodes."""

    # Get batch size
    batch_size = pnode_features.shape[0]

    # Concatenate node structural features with input features
    pnodes = graph.nodes['pnodes']
    rnodes = graph.nodes['rnodes']
    new_pnodes = pnodes._replace(
      features=jnp.concatenate([pnode_features, pnodes.features], axis=-1)
    )
    # To make sure capacity of the embedded is identical for the physical nodes and
    # the regional nodes, we also append some dummy zero input features for the
    # regional nodes.
    dummy_rnode_features = jnp.zeros(
        rnodes.features.shape[:2] + (pnode_features.shape[-1],),
        dtype=pnode_features.dtype)
    new_rnodes = rnodes._replace(
      features=jnp.concatenate([dummy_rnode_features, rnodes.features], axis=-1)
    )

    # Get edges
    p2r_edges_key = graph.edge_key_by_name('p2r')
    edges = graph.edges[p2r_edges_key]
    # Drop out edges randomly with the given probability
    if key is not None:
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[1])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        key=key, arrays=[edges.features, edges.indices.senders, edges.indices.receivers], axis=1)
      new_edge_features = new_edge_features[:, :n_edges_after]
      new_edge_senders = new_edge_senders[:, :n_edges_after]
      new_edge_receivers = new_edge_receivers[:, :n_edges_after]
    else:
      n_edges_after = edges.features.shape[1]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    # Change edge feature dtype
    new_edge_features = new_edge_features.astype(dummy_rnode_features.dtype)
    # Build new edge set
    new_edges = EdgeSet(
      n_edge=jnp.tile(jnp.array([n_edges_after]), reps=(batch_size, 1)),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    input_graph = graph._replace(
      edges={p2r_edges_key: new_edges},
      nodes={
        'pnodes': new_pnodes,
        'rnodes': new_rnodes
      })

    # Run the GNN
    p2r_out = self.gnn(input_graph, condition=tau)
    latent_rnodes = p2r_out.nodes['rnodes'].features
    latent_pnodes = p2r_out.nodes['pnodes'].features

    return latent_rnodes, latent_pnodes

class Processor(nn.Module):
  """Processor block of RIGNO.

  Args:
    steps: Number of message passing blocks in the processor.
    node_latent_size: Dimension of the latent node features.
    edge_latent_size: Dimension of the latent edge features.
    mlp_hidden_layers: Number of hidden layers in the MLPs.
    use_layer_norm: Whether to use LayerNorm layers.
    conditioned_normalization: Whether to use conditioned normalization layers.
    cond_norm_hidden_size: Hidden size for the shallow MLP used for
      computing shift and scales in the conditioned normalization layers.
    p_edge_masking: Probability of masking an edge.
  """

  steps: int
  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    self.gnn = DeepTypedGraphNet(
      embed_nodes=False,  # Node features already embdded by previous layers
      embed_edges=True,  # Embed raw features of the edges
      edge_latent_size=dict(r2r=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size),
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=self.steps,
      use_layer_norm=True,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      # NOTE: segment_mean because number of edges is not balanced
      aggregate_edges_for_nodes_fn=jraph.segment_mean,
    )

  def __call__(self,
    graph: TypedGraph,
    rnode_features: Array,
    tau: Union[None, float],
    key: Union[flax.typing.PRNGKey, None] = None,
  ) -> Array:
    """Runs the r2r GNN, extracting updated latent regional nodes."""

    # Get batch size
    batch_size = rnode_features.shape[0]

    # Replace the node features
    # NOTE: We don't need to add the structural node features, because these are
    # already part of  the latent state, via the original p2r gnn.
    rnodes = graph.nodes['rnodes']
    new_rnodes = rnodes._replace(features=rnode_features)

    # Get edges
    r2r_edges_key = graph.edge_key_by_name('r2r')
    # NOTE: We are assuming here that the r2r gnn uses a single set of edge keys
    # named 'r2r' for the edges and that it uses a single set of nodes named 'rnodes'
    msg = ('The setup currently requires to only have one kind of edge in the mesh GNN.')
    assert len(graph.edges) == 1, msg
    edges = graph.edges[r2r_edges_key]
    # Drop out edges randomly with the given probability
    # NOTE: We need the structural edge features, because it is the first
    # time we are seeing this particular set of edges.
    if key is not None:
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[1])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        key=key, arrays=[edges.features, edges.indices.senders, edges.indices.receivers], axis=1)
      new_edge_features = new_edge_features[:, :n_edges_after]
      new_edge_senders = new_edge_senders[:, :n_edges_after]
      new_edge_receivers = new_edge_receivers[:, :n_edges_after]
    else:
      n_edges_after = edges.features.shape[1]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    # Change edge feature dtype
    new_edge_features = new_edge_features.astype(rnode_features.dtype)
    # Build new edge set
    new_edges = EdgeSet(
      n_edge=jnp.tile(jnp.array([n_edges_after]), reps=(batch_size, 1)),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    # Build the graph
    input_graph = graph._replace(
      edges={r2r_edges_key: new_edges},
      nodes={'rnodes': new_rnodes},
    )

    # Run the GNN
    output_graph = self.gnn(input_graph, condition=tau)
    output_rnodes = output_graph.nodes['rnodes'].features

    return output_rnodes

class Decoder(nn.Module):
  """Decoder block of RIGNO.

  Args:
    num_outputs: Number of output variables.
    node_latent_size: Dimension of the latent node features.
    edge_latent_size: Dimension of the latent edge features.
    mlp_hidden_layers: Number of hidden layers in the MLPs.
    use_layer_norm: Whether to use LayerNorm layers.
    conditioned_normalization: Whether to use conditioned normalization layers.
    cond_norm_hidden_size: Hidden size for the shallow MLP used for
      computing shift and scales in the conditioned normalization layers.
    p_edge_masking: Probability of masking an edge.
  """

  num_outputs: int
  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    self.gnn = DeepTypedGraphNet(
    # NOTE: with variable mesh, the output pnode features must be embedded
    embed_nodes=False,
    embed_edges=True,  # Embed raw features of the edges
    # Require a specific node dimensionaly for the physical node outputs
    # NOTE: This triggers the independent mapping for pnodes
    node_output_size=dict(pnodes=self.num_outputs),
    edge_latent_size=dict(r2p=self.edge_latent_size),
    node_latent_size=dict(rnodes=self.node_latent_size, pnodes=self.node_latent_size),
    mlp_num_hidden_layers=self.mlp_hidden_layers,
    num_message_passing_steps=1,
    use_layer_norm=True,
    conditioned_normalization=self.conditioned_normalization,
    cond_norm_hidden_size=self.cond_norm_hidden_size,
    include_sent_messages_in_node_update=False,
    activation='swish',
    f32_aggregation=False,
    # NOTE: segment_mean because number of edges is not balanced
    aggregate_edges_for_nodes_fn=jraph.segment_mean,
  )

  def __call__(self,
    graph: TypedGraph,
    rnode_features: Array,
    pnode_features: Array,
    tau: Union[None, float],
    key: Union[flax.typing.PRNGKey, None] = None,
  ) -> Array:
    """Runs the r2p GNN, extracting the output physical nodes."""

    # Get batch size
    batch_size = rnode_features.shape[0]

    # NOTE: We don't need to add the structural node features, because these are
    # already part of the latent state, via the original p2r gnn.
    rnodes = graph.nodes['rnodes']
    pnodes = graph.nodes['pnodes']
    new_rnodes = rnodes._replace(features=rnode_features)
    new_pnodes = pnodes._replace(features=pnode_features)

    # Get edges
    r2p_edges_key = graph.edge_key_by_name('r2p')
    edges = graph.edges[r2p_edges_key]
    # Drop out edges randomly with the given probability
    if key is not None:
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[1])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        key=key, arrays=[edges.features, edges.indices.senders, edges.indices.receivers], axis=1)
      new_edge_features = new_edge_features[:, :n_edges_after]
      new_edge_senders = new_edge_senders[:, :n_edges_after]
      new_edge_receivers = new_edge_receivers[:, :n_edges_after]
    else:
      n_edges_after = edges.features.shape[1]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    # Change edge feature dtype
    new_edge_features = new_edge_features.astype(pnode_features.dtype)
    # Build new edge set
    new_edges = EdgeSet(
      n_edge=jnp.tile(jnp.array([n_edges_after]), reps=(batch_size, 1)),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    # Build the new graph
    input_graph = graph._replace(
      edges={r2p_edges_key: new_edges},
      nodes={
        'rnodes': new_rnodes,
        'pnodes': new_pnodes
      })

    # Run the GNN
    output_graph = self.gnn(input_graph, condition=tau)
    output_pnodes = output_graph.nodes['pnodes'].features

    return output_pnodes

class RIGNO(AbstractOperator):
  """RIGNO: Region Interaction Graph Neural Operator.
  The default values correspond to the RIGNO-18 model.

  Args:
    num_outputs: Number of output variables.
    processor_steps: Number of message passing blocks in the processor.
    node_latent_size: Dimension of the latent node features.
    edge_latent_size: Dimension of the latent edge features.
    mlp_hidden_layers: Number of hidden layers in the MLPs.
    concatenate_t: Wether to concatenate the input time to the features of all nodes.
    concatenate_tau: Wether to concatenate the lead time to the features of all nodes.
    conditioned_normalization: Whether to use conditioned normalization layers.
    cond_norm_hidden_size: Hidden size for the shallow MLP used for
      computing shift and scales in the conditioned normalization layers.
    p_edge_masking: Probability of masking an edge.
  """

  num_outputs: int
  processor_steps: int = 18
  node_latent_size: int = 128
  edge_latent_size: int = 128
  mlp_hidden_layers: int = 1
  concatenate_t: bool = True
  concatenate_tau: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: int = 16
  p_edge_masking: int = 0.5
  decoder_bypass_mode: str = 'none'
  decoder_bypass_features: str = 'none'
  decoder_bypass_feature_source: str = 'normalized_c'
  decoder_bypass_feature_indices: Tuple[int, ...] = ()
  decoder_bypass_feature_names: Tuple[str, ...] = ()
  # Audited source names are configuration provenance.  The decoder consumes
  # only the corresponding resolved indices above, but retaining the names in
  # the model config lets the V5 runner reconstruct and validate a local-only
  # bypass without passing an unsupported keyword to RIGNO.
  decoder_bypass_local_feature_names: Tuple[str, ...] = ()
  decoder_bypass_num_features: int = 0
  decoder_bypass_output_space: str = 'normalized_deltaT'
  decoder_bypass_hidden_size: int = 64
  decoder_bypass_layers: int = 2
  decoder_bypass_init: str = 'zero_residual'
  decoder_bypass_residual_scale: float = 1.0
  global_context_mode: str = 'none'
  global_context_feature_dim: int = 0
  global_context_feature_names: Tuple[str, ...] = ()
  film_target: str = 'rnodes_processed'
  film_init: str = 'identity'
  film_hidden_size: int = 64
  native_output_mode: str = 'legacy_normalized_deltaT'
  shape_scale_epsilon: float = 1.0e-12
  scale_head_hidden_size: int = 64
  scale_head_init: str = 'identity'
  scale_head_mode: str = 'physics_only'
  scale_pooling: str = 'mean'
  native_branch_mode: str = 'joint'

  def _check_coordinates(self, x: Array) -> None:
    assert x is not None
    assert x.ndim == 2
    assert x.shape[1] <= 3
    assert x.min() >= -1
    assert x.max() <= +1

  def _check_function(self, u: Array, x: Array) -> None:
    assert u is not None
    assert u.ndim == 4
    assert u.shape[1] == 1
    assert u.shape[2] == x.shape[2], f'u: {u.shape}, x: {x.shape}'

  def setup(self):
    self._validate_decoder_bypass_config()
    self._validate_global_context_config()
    self._validate_native_shape_scale_config()
    self.encoder = Encoder(
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      p_edge_masking=self.p_edge_masking,
      name='encoder',
    )

    self.processor = Processor(
      steps=self.processor_steps,
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      p_edge_masking=self.p_edge_masking,
      name='processor',
    )

    self.decoder = Decoder(
      num_outputs=self.num_outputs,
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      p_edge_masking=self.p_edge_masking,
      name='decoder',
    )
    if self._decoder_bypass_enabled():
      self.decoder_bypass_hidden = [
        nn.Dense(
          self.decoder_bypass_hidden_size,
          name=f'decoder_bypass_hidden_{index}',
        )
        for index in range(self.decoder_bypass_layers)
      ]
      output_kernel_init = (
        nn.initializers.zeros
        if self.decoder_bypass_init == 'zero_residual'
        else nn.initializers.lecun_normal()
      )
      output_bias_init = (
        nn.initializers.zeros
        if self.decoder_bypass_init == 'zero_residual'
        else nn.initializers.zeros
      )
      self.decoder_bypass_output = nn.Dense(
        self.num_outputs,
        kernel_init=output_kernel_init,
        bias_init=output_bias_init,
        name='decoder_bypass_output',
      )
    else:
      self.decoder_bypass_hidden = ()
      self.decoder_bypass_output = None
    if self._global_film_enabled():
      self.global_film_hidden = nn.Dense(
        self.film_hidden_size,
        name='global_film_hidden',
      )
      # Identity FiLM requires both gamma and beta to start identically zero.
      # This leaves all pre-existing encoder/processor/decoder parameter paths
      # unchanged for partial loading from a V4 checkpoint.
      self.global_film_output = nn.Dense(
        2 * self.node_latent_size,
        kernel_init=nn.initializers.zeros,
        bias_init=nn.initializers.zeros,
        name='global_film_output',
      )
    else:
      self.global_film_hidden = None
      self.global_film_output = None
    if self._native_shape_scale_enabled():
      self.global_scale_hidden = nn.Dense(
        self.scale_head_hidden_size,
        name='global_scale_hidden',
      )
      # residual_scale(g)=0 at initialization, so s_hat=s_phys exactly.
      self.global_scale_output = nn.Dense(
        1,
        kernel_init=nn.initializers.zeros,
        bias_init=nn.initializers.zeros,
        name='global_scale_output',
      )
    else:
      self.global_scale_hidden = None
      self.global_scale_output = None

  def _decoder_bypass_enabled(self) -> bool:
    return self.decoder_bypass_mode != 'none'

  def _global_film_enabled(self) -> bool:
    return self.global_context_mode == 'film'

  def _native_shape_scale_enabled(self) -> bool:
    return self.native_output_mode == 'native_shape_scale'

  def _validate_decoder_bypass_config(self) -> None:
    if self.decoder_bypass_mode not in {'none', 'post_decoder_residual'}:
      raise ValueError(
        "decoder_bypass_mode must be one of {'none', 'post_decoder_residual'}, "
        f"found {self.decoder_bypass_mode!r}"
      )
    if self.decoder_bypass_features not in {'none', 'full_condition', 'explicit_local_condition'}:
      raise ValueError(
        "decoder_bypass_features must be one of {'none', 'full_condition', "
        "'explicit_local_condition'}, "
        f"found {self.decoder_bypass_features!r}"
      )
    if self.decoder_bypass_feature_source != 'normalized_c':
      raise ValueError(
        "decoder_bypass_feature_source must be 'normalized_c', "
        f"found {self.decoder_bypass_feature_source!r}"
      )
    if self.decoder_bypass_init != 'zero_residual':
      raise ValueError(
        "decoder_bypass_init must be 'zero_residual', "
        f"found {self.decoder_bypass_init!r}"
      )
    allowed_output_spaces = {'normalized_deltaT', 'native_psi'}
    if self.decoder_bypass_output_space not in allowed_output_spaces:
      raise ValueError(
        "decoder_bypass_output_space must be one of "
        f"{allowed_output_spaces}, "
        f"found {self.decoder_bypass_output_space!r}"
      )
    if self.decoder_bypass_hidden_size < 1:
      raise ValueError("decoder_bypass_hidden_size must be >= 1")
    if self.decoder_bypass_layers < 1:
      raise ValueError("decoder_bypass_layers must be >= 1")
    if self.decoder_bypass_mode == 'none':
      if self.decoder_bypass_features != 'none':
        raise ValueError("decoder_bypass_mode='none' requires decoder_bypass_features='none'")
      return
    if self.decoder_bypass_features not in {'full_condition', 'explicit_local_condition'}:
      raise ValueError(
        "post_decoder_residual requires decoder_bypass_features='full_condition' "
        "or 'explicit_local_condition'"
      )
    if not self.decoder_bypass_feature_indices:
      raise ValueError("decoder bypass requires resolved feature indices")
    if self.decoder_bypass_num_features != len(self.decoder_bypass_feature_indices):
      raise ValueError(
        "decoder_bypass_num_features must match decoder_bypass_feature_indices"
      )

  def _validate_global_context_config(self) -> None:
    if self.global_context_mode not in {'none', 'film'}:
      raise ValueError(
        "global_context_mode must be one of {'none', 'film'}, "
        f"found {self.global_context_mode!r}"
      )
    if self.film_target != 'rnodes_processed':
      raise ValueError(
        "film_target must be 'rnodes_processed', "
        f"found {self.film_target!r}"
      )
    if self.film_init != 'identity':
      raise ValueError(
        "film_init must be 'identity', "
        f"found {self.film_init!r}"
      )
    if self.film_hidden_size < 1:
      raise ValueError("film_hidden_size must be >= 1")
    if self.global_context_mode == 'none':
      if self.global_context_feature_dim < 0:
        raise ValueError("global_context_feature_dim must be >= 0")
      if self.global_context_feature_names and (
        len(self.global_context_feature_names) != self.global_context_feature_dim
      ):
        raise ValueError(
          "global_context_feature_names must match global_context_feature_dim"
        )
      return
    if self.global_context_feature_dim < 1:
      raise ValueError("film global_context_feature_dim must be >= 1")
    if self.global_context_feature_names and (
      len(self.global_context_feature_names) != self.global_context_feature_dim
    ):
      raise ValueError(
        "global_context_feature_names must match global_context_feature_dim"
      )

  def _validate_native_shape_scale_config(self) -> None:
    if self.native_output_mode not in {'legacy_normalized_deltaT', 'native_shape_scale'}:
      raise ValueError(
        "native_output_mode must be one of "
        "{'legacy_normalized_deltaT', 'native_shape_scale'}; "
        f"found {self.native_output_mode!r}"
      )
    if self.shape_scale_epsilon <= 0.0:
      raise ValueError("shape_scale_epsilon must be > 0")
    if self.scale_head_hidden_size < 1:
      raise ValueError("scale_head_hidden_size must be >= 1")
    if self.scale_head_init != 'identity':
      raise ValueError("scale_head_init must be 'identity'")
    if self.scale_head_mode not in {'physics_only', 'physics_plus_pooled_latent'}:
      raise ValueError(
        "scale_head_mode must be one of {'physics_only', "
        "'physics_plus_pooled_latent'}"
      )
    if self.scale_pooling not in {'mean'}:
      raise ValueError("scale_pooling must currently be 'mean'")
    if self.native_branch_mode not in {'scale_only', 'shape_only', 'joint'}:
      raise ValueError(
        "native_branch_mode must be one of {'scale_only', 'shape_only', 'joint'}"
      )
    if not self._native_shape_scale_enabled():
      if self.decoder_bypass_output_space != 'normalized_deltaT':
        raise ValueError(
          "legacy_normalized_deltaT requires decoder_bypass_output_space="
          "'normalized_deltaT'"
        )
      return
    if self.num_outputs != 1:
      raise ValueError("native_shape_scale requires num_outputs=1")
    if self.global_context_feature_dim < 1:
      raise ValueError("native_shape_scale requires a nonempty global context")
    if self.global_context_feature_names and (
      len(self.global_context_feature_names) != self.global_context_feature_dim
    ):
      raise ValueError(
        "native_shape_scale global_context_feature_names must match feature dimension"
      )
    if self._decoder_bypass_enabled() and self.decoder_bypass_output_space != 'native_psi':
      raise ValueError(
        "native_shape_scale decoder bypass must use decoder_bypass_output_space='native_psi'"
      )

  @staticmethod
  def _prepare_features(feats: Array) -> Array:
    # Expand time axis
    feats = jnp.expand_dims(feats, axis=1)
    return feats

  def _encode_process_decode(self,
    graphs: RegionInteractionGraphSet,
    pnode_features: Array,
    tau: Union[None, float],
    global_context: Union[None, Array] = None,
    key: flax.typing.PRNGKey = None,
  ) -> Tuple[Array, Array]:

    # Add dummy node features
    dummy_pnode_features = jnp.zeros(shape=(pnode_features.shape[0], 1, pnode_features.shape[2]))
    pnode_features = jnp.concatenate([pnode_features, dummy_pnode_features], axis=1)

    # Transfer data for the physical mesh to the regional mesh
    # -> [batch_size, num_nodes, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    (latent_rnodes, latent_pnodes) = self.encoder(graphs.p2r, pnode_features, tau, key=subkey)
    self.sow(
      col='intermediates', name='pnodes_encoded',
      value=self._prepare_features(latent_pnodes[:, :-1])
    )
    self.sow(
      col='intermediates', name='rnodes_encoded',
      value=self._prepare_features(latent_rnodes[:, :-1])
    )

    # Run message-passing in the regional mesh
    # -> [batch_size, num_rnodes, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    updated_latent_rnodes = self.processor(graphs.r2r, latent_rnodes, tau, key=subkey)
    if self._global_film_enabled():
      self.sow(
        col='intermediates', name='rnodes_processed_pre_film',
        value=self._prepare_features(updated_latent_rnodes[:, :-1])
      )
    updated_latent_rnodes = self._apply_global_film(updated_latent_rnodes, global_context)
    self.sow(
      col='intermediates', name='rnodes_processed',
      value=self._prepare_features(updated_latent_rnodes[:, :-1])
    )

    # Transfer data from the regional mesh to the physical mesh
    # -> [batch_size, num_pnodes_out, latent_size]
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    output_pnodes = self.decoder(graphs.r2p, updated_latent_rnodes, latent_pnodes, tau, key=subkey)
    self.sow(
      col='intermediates', name='pnodes_decoded',
      value=self._prepare_features(output_pnodes[:, :-1])
    )

    # Remove dummy node features
    output_pnodes = output_pnodes[:, :-1, :]

    return output_pnodes, updated_latent_rnodes[:, :-1]

  def _apply_global_film(
    self,
    rnode_latents: Array,
    global_context: Union[None, Array],
  ) -> Array:
    """Apply sample-global identity-initialized FiLM at processed rnodes."""

    if not self._global_film_enabled():
      # Preserve the V4 path exactly when the feature is disabled.
      return rnode_latents
    context = self._global_context_array(
      global_context, batch_size=rnode_latents.shape[0], dtype=rnode_latents.dtype)
    film_hidden = nn.gelu(self.global_film_hidden(context))
    gamma_beta = self.global_film_output(film_hidden)
    gamma, beta = jnp.split(gamma_beta, 2, axis=-1)
    self.sow(col='intermediates', name='global_film_gamma', value=gamma)
    self.sow(col='intermediates', name='global_film_beta', value=beta)
    return (1.0 + gamma[:, None, :]) * rnode_latents + beta[:, None, :]

  def _global_context_array(
    self,
    global_context: Union[None, Array],
    *,
    batch_size: int,
    dtype,
  ) -> Array:
    if global_context is None:
      raise ValueError("native shape-scale or Global FiLM requires global_context")
    context = jnp.asarray(global_context, dtype=dtype)
    if context.ndim != 2:
      raise ValueError(
        "global_context must have shape [batch_size, feature_dim], "
        f"found {context.shape}"
      )
    if context.shape[0] != batch_size:
      raise ValueError(
        "global_context batch size must match model batch: "
        f"context={context.shape[0]} batch={batch_size}"
      )
    if context.shape[1] != self.global_context_feature_dim:
      raise ValueError(
        "global_context feature dimension mismatch: "
        f"context={context.shape[1]} configured={self.global_context_feature_dim}"
      )
    return context

  @staticmethod
  def _prediction_field(value: Array, prediction: Array, name: str) -> Array:
    """Coerce a per-node raw field to ``[B,1,N,1]`` prediction layout."""

    array = jnp.asarray(value, dtype=prediction.dtype)
    if array.shape == prediction.shape:
      return array
    if array.ndim == 2 and array.shape == (prediction.shape[0], prediction.shape[2]):
      return array[:, None, :, None]
    if array.ndim == 1 and array.shape[0] == prediction.shape[2]:
      return jnp.broadcast_to(array[None, None, :, None], prediction.shape)
    raise ValueError(
      f"{name} must have shape [B,1,N,1], [B,N], or [N]; "
      f"found {array.shape} for prediction {prediction.shape}"
    )

  @staticmethod
  def _sample_scalar(value: Array, prediction: Array, name: str) -> Array:
    """Coerce one scalar per sample to ``[B,1,1,1]``."""

    array = jnp.asarray(value, dtype=prediction.dtype)
    if array.ndim == 1 and array.shape[0] == prediction.shape[0]:
      return array[:, None, None, None]
    if array.ndim == 2 and array.shape == (prediction.shape[0], 1):
      return array[:, :, None, None]
    if array.ndim == 4 and array.shape == (prediction.shape[0], 1, 1, 1):
      return array
    raise ValueError(
      f"{name} must have one scalar per batch item; found {array.shape} "
      f"for prediction {prediction.shape}"
    )

  def predict_native_shape_scale(
    self,
    inputs: Inputs,
    graphs: RegionInteractionGraphSet,
    *,
    control_volumes: Array,
    log_s_phys: Array,
    reference_temperature: Array,
    dirichlet_mask: Array,
    prescribed_temperature: Array,
    global_context: Union[None, Array] = None,
    key: flax.typing.PRNGKey = None,
  ) -> dict:
    """Predict native ``DeltaT=s*phi`` and project raw Dirichlet nodes.

    ``psi`` is the decoder's unnormalized field. ``phi_hat`` is normalized per
    sample by physical control-volume RMS; the scale head predicts a residual
    around inference-only ``log_s_phys``. Targets appear nowhere in this API.
    """

    if not self._native_shape_scale_enabled():
      raise ValueError("predict_native_shape_scale requires native_output_mode='native_shape_scale'")
    psi, processed_rnodes = self._call_with_processed_rnodes(
      inputs, graphs, key=key, global_context=global_context)
    volumes = self._prediction_field(control_volumes, psi, 'control_volumes')
    volume_sum = jnp.sum(volumes, axis=2, keepdims=True)
    if volumes.shape != psi.shape:
      raise ValueError("control_volumes must align with native psi")
    dirichlet = self._prediction_field(dirichlet_mask, psi, 'dirichlet_mask') > 0.5
    psi_free = jnp.where(dirichlet, jnp.zeros_like(psi), psi)
    psi_rms = jnp.sqrt(
      jnp.sum(jnp.square(psi_free) * volumes, axis=2, keepdims=True)
      / jnp.maximum(volume_sum, self.shape_scale_epsilon)
    )
    phi_hat = psi_free / jnp.maximum(psi_rms, self.shape_scale_epsilon)
    context = self._global_context_array(
      global_context, batch_size=psi.shape[0], dtype=psi.dtype)
    if self.scale_head_mode == 'physics_plus_pooled_latent':
      if self.scale_pooling != 'mean':
        raise ValueError(f"unsupported scale pooling {self.scale_pooling!r}")
      pooled_rnodes = jnp.mean(processed_rnodes, axis=1)
      scale_features = jnp.concatenate([context, pooled_rnodes], axis=-1)
    else:
      pooled_rnodes = jnp.zeros((psi.shape[0], 0), dtype=psi.dtype)
      scale_features = context
    scale_hidden = nn.gelu(self.global_scale_hidden(scale_features))
    residual_scale = self.global_scale_output(scale_hidden)[:, :, None, None]
    log_s_hat = self._sample_scalar(log_s_phys, psi, 'log_s_phys') + residual_scale
    s_hat = jnp.exp(log_s_hat)
    reconstruction_shape = (
      jax.lax.stop_gradient(phi_hat)
      if self.native_branch_mode == 'scale_only'
      else phi_hat
    )
    reconstruction_scale = (
      jax.lax.stop_gradient(s_hat)
      if self.native_branch_mode == 'shape_only'
      else s_hat
    )
    delta_unprojected = reconstruction_scale * reconstruction_shape
    reference = self._prediction_field(reference_temperature, psi, 'reference_temperature')
    prescribed = self._prediction_field(prescribed_temperature, psi, 'prescribed_temperature')
    raw_temperature_unprojected = reference + delta_unprojected
    raw_temperature = jnp.where(dirichlet, prescribed, raw_temperature_unprojected)
    delta_hat = raw_temperature - reference
    self.sow(col='intermediates', name='native_shape_psi_rms', value=psi_rms)
    self.sow(col='intermediates', name='native_scale_residual', value=residual_scale)
    self.sow(col='intermediates', name='native_scale_log_s_hat', value=log_s_hat)
    return {
      'psi': psi,
      'psi_free': psi_free,
      'phi_hat': phi_hat,
      'psi_cv_rms': psi_rms,
      'residual_scale': residual_scale,
      'log_s_hat': log_s_hat,
      's_hat': s_hat,
      'pooled_rnodes': pooled_rnodes,
      'deltaT_hat_unprojected': delta_unprojected,
      'raw_temperature_unprojected': raw_temperature_unprojected,
      'raw_temperature': raw_temperature,
      'deltaT_hat': delta_hat,
    }

  def _call_with_processed_rnodes(self,
    inputs: Inputs,
    graphs: RegionInteractionGraphSet,
    key: flax.typing.PRNGKey = None,
    global_context: Union[None, Array] = None,
  ) -> Tuple[Array, Array]:
    """Inputs must be of shape [batch_size, 1, num_physical_nodes, num_inputs]"""

    # Check input functions
    self._check_function(inputs.u, x=inputs.x_inp)
    if inputs.c is not None:
      self._check_function(inputs.c, x=inputs.x_inp)
    assert inputs.u.shape[3] == self.num_outputs

    # Read dimensions
    batch_size = inputs.u.shape[0]
    num_pnodes_inp = inputs.x_inp.shape[2]
    num_pnodes_out = inputs.x_out.shape[2]

    # Prepare the time channel
    if self.concatenate_t:
      assert inputs.t is not None
      t_inp = jnp.array(inputs.t, dtype=jnp.float32)
      if t_inp.ndim == 4:
        t_inp = t_inp[:, :, 0, 0]
      if t_inp.size == 1:
        t_inp = jnp.tile(t_inp.reshape(1, 1), reps=(batch_size, 1))
    # Prepare the time difference channel
    if self.concatenate_tau:
      assert inputs.tau is not None
      tau = jnp.array(inputs.tau, dtype=jnp.float32)
      if tau.ndim == 4:
        tau = tau[:, :, 0, 0]
      if tau.size == 1:
        tau = jnp.tile(tau.reshape(1, 1), reps=(batch_size, 1))
    else:
      tau = None

    # Concatenate the known coefficients to the channels of the input function
    if inputs.c is None:
      u_inp = inputs.u
    else:
      u_inp = jnp.concatenate([inputs.u, inputs.c], axis=-1)

    # Prepare the physical node features
    # u -> [batch_size, num_pnodes_inp, num_inputs]
    pnode_features = jnp.moveaxis(u_inp,
      source=(0, 1, 2, 3), destination=(0, 3, 1, 2)
    ).squeeze(axis=3)

    # Concatente with forced features
    pnode_features_forced = []
    if self.concatenate_t:
      pnode_features_forced.append(jnp.tile(jnp.expand_dims(t_inp, axis=1), reps=(1, num_pnodes_inp, 1)))
    if self.concatenate_tau:
      pnode_features_forced.append(jnp.tile(jnp.expand_dims(tau, axis=1), reps=(1, num_pnodes_inp, 1)))
    pnode_features = jnp.concatenate([pnode_features, *pnode_features_forced], axis=-1)

    # Run the GNNs
    subkey, key = jax.random.split(key) if (key is not None) else (None, None)
    output_pnodes, processed_rnodes = self._encode_process_decode(
      graphs=graphs, pnode_features=pnode_features, tau=tau,
      global_context=global_context, key=subkey)

    # Reshape the output to u
    # [batch_size, num_pnodes_out, num_outputs] -> [batch_size, 1, num_pnodes_out, num_outputs]
    output = self._prepare_features(output_pnodes)
    output = self._apply_decoder_bypass(output, inputs)
    self._check_function(output, x=inputs.x_out)

    return output, processed_rnodes

  def call(self,
    inputs: Inputs,
    graphs: RegionInteractionGraphSet,
    key: flax.typing.PRNGKey = None,
    global_context: Union[None, Array] = None,
  ) -> Array:
    output, _ = self._call_with_processed_rnodes(
      inputs, graphs, key=key, global_context=global_context)
    return output

  def _apply_decoder_bypass(self, base_output: Array, inputs: Inputs) -> Array:
    if not self._decoder_bypass_enabled():
      return base_output
    if inputs.c is None:
      raise ValueError("decoder bypass requires inputs.c")
    if inputs.x_inp.shape != inputs.x_out.shape:
      raise ValueError(
        "decoder bypass requires one-to-one input/output node alignment; "
        f"x_inp={inputs.x_inp.shape} x_out={inputs.x_out.shape}"
      )
    if inputs.c.shape[2] != base_output.shape[2]:
      raise ValueError(
        "decoder bypass requires inputs.c node count to match decoder output; "
        f"c={inputs.c.shape} output={base_output.shape}"
      )
    indices = jnp.asarray(self.decoder_bypass_feature_indices, dtype=jnp.int32)
    residual = jnp.take(inputs.c, indices, axis=-1)
    for layer in self.decoder_bypass_hidden:
      residual = nn.gelu(layer(residual))
    residual = self.decoder_bypass_output(residual)
    self.sow(col='intermediates', name='decoder_bypass_residual', value=residual)
    return base_output + float(self.decoder_bypass_residual_scale) * residual

def _subsample_pointset(key, x: Array, factor: float) -> Array:
  """Downsamples a point cloud by randomly subsampling them."""

  x = jnp.array(x)
  x_shuffled, = shuffle_arrays(key, [x])

  return x_shuffled[:int(x.shape[0] / factor)]

def _upsample_pointset(key, x: Array, factor: float) -> Array:
  """Upsamples a point cloud by adding the middle point of randomly selected simplices."""

  factor = factor ** x.shape[-1]
  num_new_points = int(x.shape[0] * (factor - 1))
  tri = Delaunay(points=x)
  simplices = jax.random.permutation(key=key, x=tri.simplices)[jnp.arange(num_new_points)]
  x_ext = np.mean(x[simplices], axis=1)

  return np.concatenate([x, x_ext], axis=0)

def _get_edges_from_triangulation(tri: Delaunay, bidirectional: bool = True):

  indptr, cols = tri.vertex_neighbor_vertices
  rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
  edges = np.stack([rows, cols], -1)
  if bidirectional:
    edges = np.concatenate([edges, np.flip(edges, axis=-1)], axis=0)

  return edges

def _compute_triangulation_medians(tri: Delaunay) -> Array:
  edges = np.zeros(shape=tri.simplices.shape)
  medians = np.zeros(shape=tri.simplices.shape)
  for i in range(tri.simplices.shape[1]):
    points = tri.points[np.delete(tri.simplices, i, axis=1)]
    points = [p.squeeze(1) for p in np.split(points, axis=1, indices_or_sections=2)]
    edges[:, i] = np.linalg.norm(np.subtract(*points), axis=1)
  for i in range(tri.simplices.shape[1]):
    medians[:, i] = .67 * np.sqrt((2 * np.sum(np.power(np.delete(edges, i, axis=1), 2), axis=1) - np.power(edges[:, i], 2)) / 4)

  return medians
