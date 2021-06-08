import numpy as np
import logging
from deepchem.feat import MaterialStructureFeaturizer
from collections import defaultdict
from typing import List, Dict, Tuple, DefaultDict, Any
from deepchem.utils.typing import PymatgenStructure
from deepchem.feat.graph_data import GraphData
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial.transform import Rotation


class LCNNFeaturizer(MaterialStructureFeaturizer):
  """
  Calculates the 2-D Surface graph features in 6 different permutations-

  Based on the implementation of Lattice Graph Convolution Neural
  Network (LCNN). This method produces the Atom wise features ( One Hot Encoding)
  and Adjacent neighbour in the specified order of permutations. Neighbors are
  determined by first extracting a site local environment from the primitive cell,
  and perform graph matching and distance matching to find neighbors.
  First, the template of the Primitive cell needs to be defined along with periodic
  boundary conditions and active and spectator site details. structure(Data Point
  i.e different configuration of adsorbate atoms) is passed for featurization.

  This particular featurization produces a regular-graph (equal number of Neighbors)
  along with its permutation in 6 symmetric axis. This transformation can be
  applied when orderering of neighboring of nodes around a site play an important role
  in the propert predictions. Due to consideration of local neighbor environment,
  this current implementation would be fruitful in finding neighbors for calculating
  formation energy of adbsorption tasks where the local. Adsorption turns out to be important
  in many applications such as catalyst and semiconductor design.

  The permuted neighbors are calculated using the Primitive cells i.e periodic cells
  in all the data points are built via lattice transformation of the primitive cell.

  `Primitive cell Format:`

  1. Pymatgen structure object with site_properties key value
   - "SiteTypes" mentioning if it is a active site "A1" or spectator
     site "S1".
  2. ns , the number of spectator types elements. For "S1" its 1.
  3. na , the number of active types elements. For "A1" its 1.
  4. aos, the different species of active elements "A1".
  5. pbc, the periodic boundary conditions.

  `Data point Structure Format(Configuration of Atoms):`

  1. Pymatgen structure object with site_properties with following key value.
   - "SiteTypes", mentioning if it is a active site "A1" or spectator
     site "S1".
   - "oss", different occupational sites. For spectator sites make it -1.

  It is highly recommended that cells of data are directly redefined from
  the primitive cell, specifically, the relative coordinates between sites
  are consistent so that the lattice is non-deviated.

  References
  ----------
  .. [1] Jonathan Lym and Geun Ho Gu, J. Phys. Chem. C 2019, 123, 18951−18959

  Examples
  --------
  >>> import deepchem as dc
  >>> from pymatgen.core import Structure
  >>> import numpy as np
  >>> PRIMITIVE_CELL = {
  ...   "lattice": [[2.818528, 0.0, 0.0],
  ...               [-1.409264, 2.440917, 0.0],
  ...               [0.0, 0.0, 25.508255]],
  ...   "coords": [[0.66667, 0.33333, 0.090221],
  ...              [0.33333, 0.66667, 0.18043936],
  ...              [0.0, 0.0, 0.27065772],
  ...              [0.66667, 0.33333, 0.36087608],
  ...              [0.33333, 0.66667, 0.45109444],
  ...              [0.0, 0.0, 0.49656991]],
  ...   "species": ['H', 'H', 'H', 'H', 'H', 'He'],
  ...   "site_properties": {'SiteTypes': ['S1', 'S1', 'S1', 'S1', 'S1', 'A1']}
  ... }
  >>> PRIMITIVE_CELL_INF0 = {
  ...    "cutoff": np.around(6.00),
  ...    "structure": Structure(**PRIMITIVE_CELL),
  ...    "aos": ['1', '0', '2'],
  ...    "pbc": [True, True, False],
  ...    "ns": 1,
  ...    "na": 1
  ... }
  >>> DATA_POINT = {
  ...   "lattice": [[1.409264, -2.440917, 0.0],
  ...               [4.227792, 2.440917, 0.0],
  ...               [0.0, 0.0, 23.17559]],
  ...   "coords": [[0.0, 0.0, 0.099299],
  ...              [0.0, 0.33333, 0.198598],
  ...              [0.5, 0.16667, 0.297897],
  ...              [0.0, 0.0, 0.397196],
  ...              [0.0, 0.33333, 0.496495],
  ...              [0.5, 0.5, 0.099299],
  ...              [0.5, 0.83333, 0.198598],
  ...              [0.0, 0.66667, 0.297897],
  ...              [0.5, 0.5, 0.397196],
  ...              [0.5, 0.83333, 0.496495],
  ...              [0.0, 0.66667, 0.54654766],
  ...              [0.5, 0.16667, 0.54654766]],
  ...   "species": ['H', 'H', 'H', 'H', 'H', 'H',
  ...               'H', 'H', 'H', 'H', 'He', 'He'],
  ...   "site_properties": {
  ...     "SiteTypes": ['S1', 'S1', 'S1', 'S1', 'S1',
  ...                   'S1', 'S1', 'S1', 'S1', 'S1',
  ...                   'A1', 'A1'],
  ...     "oss": ['-1', '-1', '-1', '-1', '-1', '-1',
  ...             '-1', '-1', '-1', '-1', '0', '2']
  ...                   }
  ... }
  >>> featuriser = dc.feat.LCNNFeaturizer(**PRIMITIVE_CELL_INF0)
  >>> print(type(featuriser._featurize(Structure(**DATA_POINT))))
  <class 'deepchem.feat.graph_data.GraphData'>

  Notes
  -----
  This Class requires pymatgen , networkx , scipy installed.
  """

  def __init__(self,
               structure: PymatgenStructure,
               aos: List[str],
               pbc: List[bool],
               ns: int = 1,
               na: int = 1,
               cutoff: float = 6.00):
    """
    Parameters
    ----------
    structure: : PymatgenStructure
      Pymatgen Structure object of the primitive cell used for calculating
      neighbors from lattice transformations.It also requires site_properties
      attribute with "Sitetypes"(Active or spectator site).
    aos: List[str]
      A list of all the active site species. For the Pt, N, NO configuration
      set it as ['0', '1', '2']
    pbc: List[bool]
      Periodic Boundary Condition
    ns: int (default 1)
      The number of spectator types elements. For "S1" its 1.
    na: int (default 1)
      the number of active types elements. For "A1" its 1.
    cutoff: float (default 6.00)
      Cutoff of radius for getting local environment.Only
      used down to 2 digits.
    """
    try:
      from pymatgen.core import Structure
    except:
      raise ImportError("This class requires pymatgen to be installed.")

    if type(structure) is not Structure:
      structure = Structure(**structure)
    self.aos = aos
    self.cutoff = np.around(cutoff, 2)
    self.setup_env = _load_primitive_cell(structure, aos, pbc, ns, na, cutoff)

  def _featurize(self, structure: PymatgenStructure) -> GraphData:
    """
    Parameters
    ----------
    structure: : PymatgenStructure
      Pymatgen Structure object of the surface configuration. It also requires
      site_properties attribute with "Sitetypes"(Active or spectator site) and
      "oss"(Species of Active site from the list of self.aos and "-1" for
      spectator sites).

    Returns
    -------
    graph: GraphData
      Node features, All edges for each node in diffrent permutations
    """
    xSites, xNSs = self.setup_env.read_datum(structure)
    config_size = xNSs.shape
    v = np.arange(0, len(xSites)).repeat(config_size[2] * config_size[3])
    u = xNSs.flatten()
    graph = GraphData(node_features=xSites, edge_index=np.array([u, v]))
    return graph


class _SiteEnvironment(object):

  def __init__(self,
               pos: np.ndarray,
               sitetypes: List[str],
               env2config: List[int],
               permutations: List[List[int]],
               cutoff: float = 6.00,
               Grtol: float = 0.0,
               Gatol: float = 0.01,
               rtol: float = 0.01,
               atol: float = 0.0,
               tol: float = 0.01,
               grtol: float = 1e-3):
    """
    Initialize site environment

    This class contains local site environment information. This is used
    to find neighbor list in the datum.

    Parameters
    ----------
    pos : np.ndarray
      n x 3 list or numpy array of (non-scaled) positions. n is the
      number of atom.
    sitetypes : List[str]
      n list of string. String must be S or A followed by a
      number. S indicates a spectator sites and A indicates a active
      sites.
    env2config: List[int]
      A particular permutation of the neighbors around an active
      site. These indexes will be used for lattice transformation.
    permutations : List[List[int]]
      p x n list of list of integer. p is the permutation
      index and n is the number of sites.
    cutoff : float
      cutoff used for pooling neighbors.
    Grtol : float (default 0.0)
      relative tolerance in distance for forming an edge in graph
    Gatol : float (default 0.01)
      absolute tolerance in distance for forming an edge in graph
    rtol : float (default 0.01)
      relative tolerance in rmsd in distance for graph matching
    atol : float (default 0.0)
      absolute tolerance in rmsd in distance for graph matching
    tol : float (default 0.01)
      maximum tolerance of position RMSD to decide whether two
      environment are the same
    grtol : float (default 0.01)
      tolerance for deciding symmetric nodes
    """
    try:
      import networkx.algorithms.isomorphism as iso
    except:
      raise ImportError("This class requires networkx to be installed.")
    self.pos = pos
    self.sitetypes = sitetypes
    self.activesiteidx = [i for i, s in enumerate(self.sitetypes) if 'A' in s]
    self.formula: DefaultDict[str, int] = defaultdict(int)
    for s in sitetypes:
      self.formula[s] += 1
    self.permutations = permutations
    self.env2config = env2config
    self.cutoff = cutoff
    # Set up site environment matcher
    self.tol = tol
    # Graphical option
    self.Grtol = Grtol
    self.Gatol = Gatol
    # tolerance for grouping nodes
    self.grtol = grtol
    # determine minimum distance between sitetypes.
    # This is used to determine the existence of an edge
    dists = squareform(pdist(pos))
    mindists = defaultdict(list)
    for i, row in enumerate(dists):
      row_dists = defaultdict(list)
      for j in range(0, len(sitetypes)):
        if i == j:
          continue
        # Sort by bond
        row_dists[frozenset((sitetypes[i], sitetypes[j]))].append(dists[i, j])
      for pair in row_dists:
        mindists[pair].append(np.min(row_dists[pair]))
    # You want to maximize this in order to make sure every node gets an edge
    self.mindists = {}
    for pair in mindists:
      self.mindists[pair] = np.max(mindists[pair])
    # construct graph
    self.G = self._construct_graph(pos, sitetypes)
    # matcher options
    self._nm = iso.categorical_node_match('n', '')
    self._em = iso.numerical_edge_match('d', 0, rtol, 0)

  def _construct_graph(self, pos: np.ndarray, sitetypes: List[str]):
    """
    Returns local environment graph using networkx and
    tolerance specified.

    Parameters
    ----------
    pos: np.ndarray
      ns x 3. coordinates of positions. ns is the number of sites.
      sitetypes: ns. sitetype for each site
    sitetypes: List[str]
      List of site properties mentioning if it is a active site "Ai"
      or spectator site "Si".

    Returns
    ------
    G: networkx.classes.graph.Graph
      networkx graph used for matching site positions in
      datum.
    """
    try:
      import networkx as nx
    except:
      raise ImportError("This class requires networkx to be installed.")

    # construct graph
    G = nx.Graph()
    dists = cdist([[0, 0, 0]], pos - np.mean(pos, 0))[0]
    sdists = np.sort(dists)
    uniquedists = sdists[
        ~(np.triu(np.abs(sdists[:, None] - sdists) <= self.grtol, 1)).any(0)]
    orderfromcenter = np.digitize(dists, uniquedists)
    # Add nodes
    for i, o in enumerate(orderfromcenter):
      G.add_node(i, n=str(o) + sitetypes[i])
    # Add edge. distance is edge attribute
    dists = pdist(pos)
    n = 0
    for i in range(len(sitetypes)):
      for j in range(i + 1, len(sitetypes)):
        if dists[n] < self.mindists[frozenset((sitetypes[i], sitetypes[j]))] or\
            (abs(self.mindists[frozenset((sitetypes[i], sitetypes[j]))] - dists[n]) <= self.Gatol + self.Grtol * abs(dists[n])):
          G.add_edge(i, j, d=dists[n])
        n += 1
    return G

  def get_mapping(self, env: Dict[str, Any]) -> Dict[int, int]:
    """
    Returns mapping of sites from input to this object

    Pymatgen molecule_matcher does not work unfortunately as it needs to be
    a reasonably physical molecule.
    Here, the graph is constructed by connecting the nearest neighbor, and
    isomorphism is performed to find matches, then kabsch algorithm is
    performed to make sure it is a match. NetworkX is used for portability.

    Parameters
    ----------

    env : Dict[str, Any]
      dictionary that contains information of local environment of a
      site in datum. See _get_SiteEnvironments definition in the class
      _SiteEnvironments for what this variable should be.

    Returns
    -------
    dict : Dict[int, int]
      Atom mapping from Primitive cell to data point.
    """
    try:
      import networkx.algorithms.isomorphism as iso
    except:
      raise ImportError("This class requires networkx to be installed.")
    # construct graph

    G = self._construct_graph(env['pos'], env['sitetypes'])
    if len(self.G.nodes) != len(G.nodes):
      s = 'Number of nodes is not equal.\n'
      raise ValueError(s)
    elif len(self.G.edges) != len(G.edges):
      logging.warning("Expected the number of edges to be equal",
                      len(self.G.edges), len(G.edges))
      s = 'Number of edges is not equal.\n'
      s += "- Is the data point's cell a redefined lattice of primitive cell?\n"
      s += '- If relaxed structure is used, you may want to check structure or increase Gatol\n'
      raise ValueError(s)
    GM = iso.GraphMatcher(self.G, G, self._nm, self._em)
    # Gets the isomorphic mapping. Also the most time consuming part of the code
    ams = list(GM.isomorphisms_iter())

    if not ams:
      s = 'No isomorphism found.\n'
      s += "- Is the data point's cell a redefined lattice of primitive cell?\n"
      s += '- If relaxed structure is used, you may want to check structure or increase rtol\n'
      raise ValueError(s)

    rmsd = []
    for am in ams:  # Loop over isomorphism
      # reconstruct graph after alinging point order
      xyz = np.zeros((len(self.pos), 3))
      for i in am:
        xyz[i, :] = env['pos'][am[i], :]
      rotation, _ = Rotation.align_vectors(self.pos, xyz)
      R = rotation.as_matrix()
      # RMSD
      rmsd.append(
          np.sqrt(
              np.mean(np.linalg.norm(np.dot(self.pos, R) - xyz, axis=1)**2)))
    mini = np.argmin(rmsd)
    minrmsd = rmsd[mini]
    if minrmsd < self.tol:
      return ams[mini]
    else:
      s = 'No isomorphism found.\n'
      s += '-Consider increasing neighbor finding tolerance'
      raise ValueError(s)


class _SiteEnvironments(object):

  def __init__(self, site_envs: List[_SiteEnvironment], ns: int, na: int,
               aos: List[str], eigen_tol: float, pbc: np.ndarray,
               cutoff: float):
    """
    Initialize
    Use Load to initialize this class.

    Parameters
    ----------
    site_envs : List[_SiteEnvironment]
      list of _SiteEnvironment object
    ns : int
      number of spectator sites types
    na : int
      number of active sites types
    aos : List[str]
      Available occupational states for active sites
      string should be the name of the occupancy. (consistent with the input data)
    eigen_tol : float
      tolerance for eigenanalysis of point group analysis in pymatgen.
    pbc : List[str]
      Boolean array, periodic boundary condition.
    cutoff : float
      Cutoff radius in angstrom for pooling sites to construct local environment
    """
    self.site_envs = site_envs
    self.unique_site_types: List[str] = [
        env.sitetypes[0] for env in self.site_envs
    ]
    self.ns = ns
    self.na = na
    self.aos = aos
    self.eigen_tol = eigen_tol
    self.pbc = pbc
    self.cutoff = cutoff

  def read_datum(self, struct,
                 cutoff_factor: float = 1.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load structure data and return neighbor information

    Parameters
    ----------
    struct: : PymatgenStructure
      Pymatgen Structure object of the surface configuration. It also requires
      site_properties attribute with "Sitetypes"(Active or spectator site) and
      "oss"(Species of Active site from the list of self.aos and "-1" for
      spectator sites).
    cutoff_factor : float
      this is extra buffer factor multiplied to cutoff to
      ensure pooling all relevant sites.

    Return
    ------
    XSites : List[float]
      One hot encoding features of the site.
    XNSs : List[List[int]]
      Neighbors calculated in different permutations
    """
    oss = [
        species for species in struct.site_properties["oss"] if species != '-1'
    ]
    # Construct one hot encoding
    XSites = np.zeros((len(oss), len(self.aos)))
    for i, o in enumerate(oss):
      XSites[i, self.aos.index(o)] = 1
    # get mapping between all site index to active site index
    alltoactive = {}
    n = 0
    for i, s in enumerate(struct.site_properties["SiteTypes"]):
      if 'A' in s:
        alltoactive[i] = n
        n += 1
    # Get Neighbors
    # Read Data
    site_envs = _get_SiteEnvironments(
        struct,
        self.cutoff * cutoff_factor,
        self.pbc,
        get_permutations=False,
        eigen_tol=self.eigen_tol)
    XNSs: List[list] = [[] for _ in range(len(self.site_envs))]
    for env in site_envs:
      i = self.unique_site_types.index(env['sitetypes'][0])
      new_env = self._truncate(self.site_envs[i], env)

      # get map between two environment
      mapping = self.site_envs[i].get_mapping(new_env)
      # align input to the primitive cell (reference)
      aligned_idx = [
          new_env['env2config'][mapping[i]]
          for i in range(len(new_env['env2config']))
      ]
      # apply permutations
      nni_perm = np.take(aligned_idx, self.site_envs[i].permutations)
      # remove spectators
      nni_perm = nni_perm[:, self.site_envs[i].activesiteidx]
      # map it to active sites
      nni_perm = np.vectorize(alltoactive.__getitem__)(nni_perm)
      XNSs[i].append(nni_perm.tolist())
    return np.array(XSites), np.array(XNSs)

  @classmethod
  def _truncate(cls, env_ref: _SiteEnvironment,
                env: Dict[str, Any]) -> Dict[str, Any]:
    """
    When cutoff_factor is used, it will pool more site than cutoff
    factor specifies. This will rule out non-relevant sites by distance.

    Parameters
    ----------
    env_ref: _SiteEnvironment
      Site information of the primitive cell
    env: Dict[str, Any]
      Site information of the data point

    Returns
    -------
    env: Dict[str, Union[list, np.ndarray]]
    """
    # Extract the right number of sites by distance
    dists = defaultdict(list)
    for i, s in enumerate(env['sitetypes']):
      dists[s].append([i, env['dist'][i]])
    for s in dists:
      dists[s] = sorted(dists[s], key=lambda x: x[1])
    siteidx = []
    for s in dists:
      siteidx += [i[0] for i in dists[s][:env_ref.formula[s]]]
    siteidx = sorted(siteidx)
    env['pos'] = [env['pos'][i] for i in range(len(env['pos'])) if i in siteidx]

    env['pos'] = np.subtract(env['pos'], np.mean(env['pos'], 0))
    env['sitetypes'] = [
        env['sitetypes'][i]
        for i in range(len(env['sitetypes']))
        if i in siteidx
    ]
    env['env2config'] = [env['env2config'][i] for i in siteidx]
    del env['dist']
    return env


def _load_primitive_cell(struct: PymatgenStructure,
                         aos: List[str],
                         pbc: List[bool],
                         ns: int,
                         na: int,
                         cutoff: float,
                         eigen_tol: float = 1e-5) -> _SiteEnvironments:
  """
  This loads the primitive cell, along with all the permutations
  required for creating a neighbor. This produces the site environments of
  the primitive cell.

  Parameters
  ----------
  struct: PymatgenStructure
    Pymatgen Structure object of the primitive cell used for calculating
    neighbors from lattice transformations.It also requires site_properties
    attribute with "Sitetypes"(Active or spectator site).
  aos: List[str]
    A list of all the active site species. For the Pt, N, NO configuration
    set it as ['0', '1', '2'].
  pbc: List[bool]
    Periodic Boundary Condition
  ns: int (default 1)
    The number of spectator types elements. For "S1" its 1.
  na: int (default 1)
    The number of active types elements. For "A1" its 1.
  cutoff: float (default 6.00)
    Cutoff of radius for getting local environment.Only
    used down to 2 digits.
  eigen_tol : float (default)
    tolerance for eigenanalysis of point group analysis in
    pymatgen.

  Returns
  -------
  SiteEnvironments: _SiteEnvironments
    Instance of the _SiteEnvironments object
  """
  site_envs = _get_SiteEnvironments(
      struct, cutoff, pbc, True, eigen_tol=eigen_tol)
  site_envs_format = [
      _SiteEnvironment(e['pos'], e['sitetypes'], e['env2config'],
                       e['permutations'], cutoff) for e in site_envs
  ]

  ust = [env.sitetypes[0] for env in site_envs_format]
  usi = np.unique(ust, return_index=True)[1]
  site_envs_format = [site_envs_format[i] for i in usi]
  return _SiteEnvironments(site_envs_format, ns, na, aos, eigen_tol, pbc,
                           cutoff)


def _get_SiteEnvironments(struct: PymatgenStructure,
                          cutoff: float,
                          PBC: List[bool],
                          get_permutations: bool = True,
                          eigen_tol: float = 1e-5) -> List[Dict[str, Any]]:
  """
  Used to extract information about both primitive cells and data points.
  Extract local environments from Structure object by calculating neighbors
  based on gaussian distance. For primitive cell, Different permutations of the
  neighbors are calculated and will be later will mapped for data point in the
  _SiteEnvironment.get_mapping() function.
  site types ,

  Parameters
  ----------
  struct: PymatgenStructure
    Pymatgen Structure object of the primitive cell used for calculating
    neighbors from lattice transformations.It also requires site_properties
    attribute with "Sitetypes"(Active or spectator site).
  cutoff : float
    cutoff distance in angstrom for collecting local
    environment.
  pbc : np.ndarray
    Periodic boundary condition
  get_permutations : bool (default True)
    Whether to find permuted neighbor list or not.
  eigen_tol : float (default 1e-5)
    Tolerance for eigenanalysis of point group analysis in
    pymatgen.

  Returns
  ------
  site_envs : List[Dict[str, Any]]
    list of local_env class
  """
  try:
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
  except:
    raise ImportError("This class requires pymatgen to be installed.")

  pbc = np.array(PBC)
  structure = struct
  neighbors = structure.get_all_neighbors(cutoff, include_index=True)
  symbols = structure.species
  site_idxs = [
      i for i, sitetype in enumerate(structure.site_properties['SiteTypes'])
      if sitetype == 'A1'
  ]
  site_sym_map = {}
  sym_site_map = {}
  for i, new_ele in enumerate(structure.species):
    sym_site_map[new_ele] = structure.site_properties['SiteTypes'][i]
    site_sym_map[structure.site_properties['SiteTypes'][i]] = new_ele

  site_envs = []
  for site_idx in site_idxs:
    local_env_sym = [symbols[site_idx]]
    local_env_xyz = [structure[site_idx].coords]
    local_env_dist = [0.0]
    local_env_sitemap = [site_idx]
    for n in neighbors[site_idx]:
      # if PBC condition is fulfilled..
      c = np.around(n[0].frac_coords, 10)
      withinPBC = np.logical_and(0 <= c, c < 1)
      if np.all(withinPBC[~pbc]):
        local_env_xyz.append(n[0].coords)
        local_env_sym.append(n[0].specie)
        local_env_dist.append(n[1])
        local_env_sitemap.append(n[2])
    local_env_xyz = np.subtract(local_env_xyz, np.mean(local_env_xyz, 0))

    perm = []
    if get_permutations:
      finder = PointGroupAnalyzer(
          Molecule(local_env_sym, local_env_xyz), eigen_tolerance=eigen_tol)
      pg = finder.get_pointgroup()
      for i, op in enumerate(pg):
        newpos = op.operate_multi(local_env_xyz)
        perm.append(np.argmin(cdist(local_env_xyz, newpos), axis=1).tolist())

    site_env = {
        'pos': local_env_xyz,
        'sitetypes': [sym_site_map[s] for s in local_env_sym],
        'env2config': local_env_sitemap,
        'permutations': perm,
        'dist': local_env_dist
    }
    site_envs.append(site_env)
  return site_envs
