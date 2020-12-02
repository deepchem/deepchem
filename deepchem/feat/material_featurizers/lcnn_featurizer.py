import numpy as np
from deepchem.feat import MaterialStructureFeaturizer
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable, Union, DefaultDict


class LCNNFeaturizer(MaterialStructureFeaturizer):
  """
  Calculates the 2-D Surface graph features in 6 diffrent permutaions-

  Based on the implementation of Lattice Graph Convolution Neural
  Network (LCNN). This method produces the Atom wise features ( One Hot Encoding)
  and Adjacent neighbour in the specified order of permutations. Neighbors are
  determined using a distance metric and Each Permutation of the Neighbors are
  calculated such a manner in which, first element is the node itself followed
  by a randomly selected neighboring site. The next site shares a surface Pt atom
  with the previous site. And they are picked consecutively.

  First, the template of the Primitive cell needs to be defined and then each
  structure(Data Point i.e different configuration of adsorbate atoms) is passed
  for featurization.

  The permuted neighbors are calculated using the Primitive cells i.e periodic cells
  in all the data points are built via lattice transformation of the primitive cell.

  [1] The Primitive Template file must be passed as raw text string or path file.\n
  [2] The datapoint must be passed as raw text string.\n

  References
  ----------
  .. [1] Jonathan Lym and Geun Ho Gu, J. Phys. Chem. C 2019, 123, 18951−18959

  Examples
  --------
  >>> primitive_cell = '''#Primitive Cell
  >>> 2.81852800e+00  0.00000000e+00  0.00000000e+00 T
  >>> -1.40926400e+00  2.44091700e+00  0.00000000e+00 T
  >>> 0.00000000e+00  0.00000000e+00  2.55082550e+01 F
  >>> 1 1
  >>> 1 0 2
  >>> 6
  >>> 0.666670000000  0.333330000000  0.090220999986 S1
  >>> 0.333330000000  0.666670000000  0.180439359180 S1
  >>> 0.000000000000  0.000000000000  0.270657718374 S1
  >>> 0.666670000000  0.333330000000  0.360876077568 S1
  >>> 0.333330000000  0.666670000000  0.451094436762 S1
  >>> 0.000000000000  0.000000000000  0.496569911270 A1
  >>> '''
  >>> structure = '''2.81859800e+00  0.00000000e+00  0.00000000e+00
  >>> -1.40929900e+00  2.44097800e+00  0.00000000e+00
  >>> 0.00000000e+00  0.00000000e+00  2.55082550e+01
  >>> 6
  >>> 0.666670000000  0.333330000000  0.090220999986 S1
  >>> 0.333330000000  0.666670000000  0.180439359180 S1
  >>> 0.000000000000  0.000000000000  0.270657718374 S1
  >>> 0.666670000000  0.333330000000  0.360876077568 S1
  >>> 0.333330000000  0.666670000000  0.451094436762 S1
  >>> 0.000000000000  0.000000000000  0.496569911270 A1 0
  >>> '''
  >>> featuriser = LCNNFeaturizer(np.around(6.00), primitive_cell)
  >>> data = featuriser._featurize(structure)
  >>> print(data.keys())
  dict_keys(['X_Sites', 'X_NSs'])

  Notes
  -----
  This Class requires pymatgen , networkx , scipy installed.

  `Primitive cell Format:`

  - [comment]
  - [ax][ay][az][pbc]
  - [bx][by][bz][pbc]
  - [cx][cy][cz][pbc]
  - [number of spectator site type][number of active site type]
  - [os1][os2][os3]
  - [number sites]
  - [site1a][site1b][site1c][site type]
  - [site2a][site2b][site2c][site type]

  `Data point Structure Format(Configuration of Atoms):`

  - [ax][ay][az]
  - [bx][by][bz]
  - [cx][cy][cz]
  - [number sites]
  - [site1a][site1b][site1c][site type][occupation state if active site]
  - [site2a][site2b][site2c][site type][occupation state if active site]

  [1] ax,ay, ... are cell basis vector\n
  [2] pbc is either T or F indication of the periodic boundary condition\n
  [3] os# is the name of the possible occupation state (interpretted as string)\n
  [4] site1a,site1b,site1c are the scaled coordinates of site 1\n
  [5] site type can be either S1, S2, ... or A1, A2,... indicating spectator\n

  """

  def __init__(self, cutoff: float, template: str):
    """
    Parameters
    ----------
    cutoff: float
      Cutoff of radius for getting local environment.Only
      used down to 2 digits.

    template: str
      Template primitive stucture in string format
    """
    self.cutoff = np.around(cutoff, 2)
    self.setup_env = load_primitive_cell(template, cutoff)

  def _featurize(self, structure) -> Dict[str, np.ndarray]:
    """
    Parameters
    ----------
    structure: str
      Structure information as raw text data input as a string

    Returns
    -------
    dict: Dict[str, np.ndarray]
      Node features, All edges for each node in diffrent permutations
    """
    xSites, xNSs = self.setup_env.read_datum(structure)
    return {"X_Sites": np.array(xSites), "X_NSs": np.array(xNSs)}


def input_reader(text: str, template: bool = False) -> Iterable[Union[List[str], np.ndarray, List[int], int]]:
  """
  Read Input structures in a format which can produce the coordinate dimensions
  and axes. If it is a primitive cell, it returns the lattice cell, coordinates,
  site types and occupation state. Else if it is a data point, it returns
  the additional periodic boundary and type of occupation state.

  Parameters
  ----------
  text : str
    structure as a string
  template: bool(default False)
    Set to true for primitive cell, and false for data point

  Returns
  -------
  list of local_env : list of local_env class
  """

  s = text.rstrip('\n').split('\n')
  nl = 0
  # read comment
  if template:
    datum = False
    nl += 1
  else:
    datum = True

  # load cell and pbc
  cell = np.zeros((3, 3))
  pbc = np.array([True, True, True])
  for i in range(3):
    t = s[nl].split()
    cell[i, :] = [float(i) for i in t[0:3]]
    if not datum and t[3] == 'F':
      pbc[i] = False
    nl += 1
  # read sites if primitive
  if not datum:
    t = s[nl].split()
    ns = int(t[0])
    na = int(t[1])
    nl += 1
    aos = s[nl].split()
    nl += 1
  # read positions
  nS = int(s[nl])
  nl += 1
  coord = np.zeros((nS, 3))
  st = []
  oss = []
  for i in range(nS):
    t = s[nl].split()
    coord[i, :] = [float(i) for i in t[0:3]]
    st.append(t[3])
    if datum and len(t) == 5:
      oss.append(t[4])
    nl += 1

  if datum:
    return cell, coord, st, oss
  else:
    return cell, pbc, coord, st, ns, na, aos


class _SiteEnvironment(object):

  def __init__(self,
               pos: List[np.ndarray],
               sitetypes: List[str],
               env2config: Union[List[int], np.ndarray],
               permutations: List[List[int]],
               cutoff: float,
               Grtol: float = 0.0,
               Gatol: float = 0.01,
               rtol: float = 0.01,
               atol: float = 0.0,
               tol: float = 0.01,
               grtol: float = 0.01):
    """
    Initialize site environment

    This class contains local site environment information. This is used
    to find neighborlist in the datum.

    Parameters
    ----------
    pos : Union[list, np.ndarray]
      n x 3 list or numpy array of (non-scaled) positions. n is the
      number of atom.
    sitetypes : list
      n list of string. String must be S or A followed by a
      number. S indicates a spectator sites and A indicates a active
      sites.
    permutations : list
      p x n list of list of integer. p is the permutation
      index and n is the number of sites.
    cutoff : float
      cutoff used for pooling neighbors. for aesthetics only
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
    try:
      from scipy.spatial.distance import pdist, squareform
    except:
      raise ImportError("This class requires scipy to be installed.")
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
    self.grtol = 1e-3
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

  def _construct_graph(self, pos: List[np.ndarray], sitetypes: List[str]):
    """
    Returns local environment graph using networkx and
    tolerance specified.

    Parameters
    ----------
    pos: list
      ns x 3. coordinates of positions. ns is the number of sites.
      sitetypes: ns. sitetype for each site

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

    try:
      from scipy.spatial.distance import cdist, pdist
    except:
      raise ImportError("This class requires scipy to be installed.")

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

  def get_mapping(self,
                  env: Dict[str, Union[List[int], List[str], np.ndarray]]
                  ) -> Dict[int, int]:
    """
    Returns mapping of sites from input to this object

    Pymatgen molecule_matcher does not work unfortunately as it needs to be
    a reasonably physical molecule.
    Here, the graph is constructed by connecting the nearest neighbor, and
    isomorphism is performed to find matches, then kabsch algorithm is
    performed to make sure it is a match. NetworkX is used for portability.

    Parameters
    ----------

    env : Dict[str, list]
      dictionary that contains information of local environment of a
      site in datum. See _get_SiteEnvironments defintion in the class
      _SiteEnvironments for what this variable should be.

    Returns
    -------
    dict : Union[Dict[int, int], None]
      Atom mapping. None if there is no mapping
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
      print(len(self.G.edges), len(G.edges))
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
      R = self._kabsch(self.pos, xyz)
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

  def _kabsch(self, P: List[np.ndarray], Q: List[np.ndarray]) -> List[np.ndarray]:
    """
    Returns rotation matrix to align coordinates using
    Kabsch algorithm.

    Parameters
    ----------
    P: np.ndarray
    Q: np.ndarray

    Returns
    -------
    R: np.ndarray
      Rotation matrix
    """
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
      S[-1] = -S[-1]
      V[:, -1] = -V[:, -1]
    R = np.dot(V, W)
    return R


class _SiteEnvironments(object):

  def __init__(self, site_envs: List[_SiteEnvironment], ns: int, na: int,
               aos: List[str], eigen_tol: float, pbc: List[bool], cutoff: float):
    """
    Initialize
    Use Load to intialize this class.

    Parameters
    ----------
    site_envs : List[_SiteEnvironment]
      list of _SiteEnvironment object
    ns : int
      number of spectator sites types
    na : int
      number of active sites types
    aos : List[str]
      Avilable occupational states for active sites
      string should be the name of the occupancy. (consistent with the input data)
    eigen_tol : float
      tolerance for eigenanalysis of point group analysis in pymatgen.
    pbc : List[str]
      periodic boundary condition.
    cutoff : float
      Cutoff radius in angstrom for pooling sites to construct local environment
    """
    self.site_envs = site_envs
    self.unique_site_types: List[str] = [env.sitetypes[0] for env in self.site_envs]
    self.ns = ns
    self.na = na
    self.aos = aos
    self.eigen_tol = eigen_tol
    self.pbc = pbc
    self.cutoff = cutoff

  def read_datum(self, text: str,
                 cutoff_factor: float = 1.1) -> Tuple[List[float], List[list]]:
    """
    Load structure data and return neighbor information

    Parameters
    ----------
    text : str
      raw string of the structure
    cutoff_factor : float
      this is extra buffer factor multiplied to cutoff to
      ensure pooling all relevant sites.

    Return
    ------
    XSites : List[float]
      One hot encoding features of the site.
    XNSs : List[list]
      Neighbors calculated in diffrent permutations
    """
    cell, coord, st, oss = input_reader(text)
    # Construct one hot encoding
    XSites = np.zeros((len(oss), len(self.aos)))
    for i, o in enumerate(oss):
      XSites[i, self.aos.index(o)] = 1
    # get mapping between all site index to active site index
    alltoactive = {}
    n = 0
    for i, s in enumerate(st):
      if 'A' in s:
        alltoactive[i] = n
        n += 1
    # Get Neighbors
    # Read Data
    site_envs = _get_SiteEnvironments(
        coord,
        cell,
        st,
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
          new_env['env2config'][mapping[i]] for i in range(len(new_env['env2config']))
      ]
      # apply permutations
      nni_perm = np.take(aligned_idx, self.site_envs[i].permutations)
      # remove spectators
      nni_perm = nni_perm[:, self.site_envs[i].activesiteidx]
      # map it to active sites
      nni_perm = np.vectorize(alltoactive.__getitem__)(nni_perm)
      XNSs[i].append(nni_perm.tolist())
    return XSites.tolist(), XNSs

  @classmethod
  def _truncate(cls, env_ref: _SiteEnvironment,
                env: Dict[str, Union[List[Union[int, float, str]], np.ndarray]]) -> Dict[str, Union[List[int], List[str], np.ndarray]]:
    """
    When cutoff_factor is used, it will pool more site than cutoff factor specifies.
    This will rule out nonrelevant sites by distance.

    Parameters
    ----------
    env_ref: _SiteEnvironment
    env: _SiteEnvironment

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
    env['env2config']: List[int] = [env['env2config'][i] for i in siteidx]
    del env['dist']
    return env


def load_primitive_cell(path: str,
                        cutoff: float,
                        eigen_tol: float = 1e-5) -> _SiteEnvironments:
  """
  This loads the primitive cell, along with all the permutations
  required for creating a neighbor. This produces the site environments of
  the primitive cell.

  Parameters
  ----------
  path : str
    Primitive Cell as a raw string
  cutoff : float.
    cutoff distance in angstrom for collecting local
    environment.
  eigen_tol : float
    tolerance for eigenanalysis of point group analysis in
    pymatgen.

  Returns
  -------
  SiteEnvironments: _SiteEnvironments
    Instance of the _SiteEnvironments object
  """
  cell, pbc, coord, st, ns, na, aos = input_reader(path, template=True)
  site_envs = _get_SiteEnvironments(
      coord, cell, st, cutoff, pbc, True, eigen_tol=eigen_tol)
  site_envs = [
      _SiteEnvironment(e['pos'], e['sitetypes'], e['env2config'],
                        e['permutations'], cutoff) for e in site_envs
  ]

  ust = [env.sitetypes[0] for env in site_envs]
  usi = np.unique(ust, return_index=True)[1]
  site_envs = [site_envs[i] for i in usi]
  return _SiteEnvironments(site_envs, ns, na, aos, eigen_tol, pbc, cutoff)


def _get_SiteEnvironments(coord: Union[List[np.ndarray], np.ndarray],
                          cell: Union[List[np.ndarray], np.ndarray],
                          SiteTypes: List[str],
                          cutoff: float,
                          pbc: List[bool],
                          get_permutations: bool = True,
                          eigen_tol: float = 1e-5) -> List[Dict[str, Union[List[Union[int, str, float]], np.ndarray]]]:

  """
  Used to extract information about both primitve cells and data points.
  Extract local environments from primitive cell. Using the two diffrent types
  site types ,

  Parameters
  ----------
  coord : Union[list, np.ndarray]
    n x 3 list or numpy array of scaled positions. n is the number
    of atom.
  cell : np.ndarray
    3 x 3 list or numpy array
  SiteTypes : List[str]
    n list of string. String must be S or A followed by a
    number. S indicates a spectator sites and A indicates a active
    sites.
  cutoff : float
    cutoff distance in angstrom for collecting local
    environment.
  pbc : list[str]
    Periodic boundary condition
  get_permutations : bool (default True)
    Whether to find permutatated neighbor list or not.
  eigen_tol : float
    Tolerance for eigenanalysis of point group analysis in
    pymatgen.

  Returns
  ------
  site_envs : List[_SiteEnvironment]
    list of local_env class
  """
  try:
    from pymatgen import Element, Structure, Molecule, Lattice
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
  except:
    raise ImportError("This class requires pymatgen to be installed.")

  try:
    from scipy.spatial.distance import cdist
  except:
    raise ImportError("This class requires scipy to be installed.")

  assert isinstance(coord, (list, np.ndarray))
  assert isinstance(cell, (list, np.ndarray))
  assert len(coord) == len(SiteTypes)

  coord = np.mod(coord, 1)
  pbc = np.array(pbc)

  # Available pymatgne functions are very limited when DummySpecie is
  # involved. This may be perhaps fixed in the future. Until then, we
  # simply bypass this by mapping site to an element
  # Find available atomic number to map site to it
  availableAN = [i + 1 for i in reversed(range(0, 118))]

  # Organize Symbols and record mapping
  symbols = []
  site_idxs = []
  SiteSymMap = {}  # mapping
  SymSiteMap = {}
  for i, SiteType in enumerate(SiteTypes):
    if SiteType not in SiteSymMap:
      symbol = Element.from_Z(availableAN.pop())
      SiteSymMap[SiteType] = symbol
      SymSiteMap[symbol] = SiteType

    else:
      symbol = SiteSymMap[SiteType]
    symbols.append(symbol)
    if 'A' in SiteType:
      site_idxs.append(i)

  # Find neighbors and permutations using pymatgen
  lattice = Lattice(cell)
  structure = Structure(lattice, symbols, coord)
  neighbors = structure.get_all_neighbors(cutoff, include_index=True)
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
        'sitetypes': [SymSiteMap[s] for s in local_env_sym],
        'env2config': local_env_sitemap,
        'permutations': perm,
        'dist': local_env_dist
    }
    site_envs.append(site_env)
  return site_envs
