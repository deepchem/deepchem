import numpy as np
from deepchem.feat import Featurizer
from collections import defaultdict
from pymatgen import Element, Structure, Molecule, Lattice
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
import networkx as nx
import networkx.algorithms.isomorphism as iso
from scipy.spatial.distance import cdist, pdist, squareform


class LCNNFeaturizer(Featurizer):
  """
  Calculates the 2-D Surface graph features in 6 diffrent permutaions-

  Based on the implementation of Lattice Graph Convolution Nueral 
  Network (LCNN) . This method produces the Atom wise features ( One Hot Enconding)
  and Ajacent neighbor in the specificed order of permutations. Neighbors are determined
  distance metric and Each Permutation of the Neighbors are calculated
  is manner in which , First element is the node itself followed by a randomly
  selected neighboring site. The next site shares a surface Pt atom with the previous
  site. And they are picked consecutively.
  First the template of the Primitive cell needs to be defined and then Each
  structure(Data Point or diffrent condifuration of adsorbate atoms) is passed
  for featurization.
  
  [1] The Primitive Template file must be passed as raw text string or path file
  [2] The datapoint must be passed as raw text string. If a text file is present use
  >>> open(input_file_path).read() 
  
  References
  ----------
  [1] Jonathan Lym and Geun Ho Gu, J. Phys. Chem. C 2019, 123, 18951−18959
  
  Examples
  ----------
      
  The input format for primitive cell Template is stored in a file named
  
  [comment]
  [ax][ay][az][pbc]
  [bx][by][bz][pbc]
  [cx][cy][cz][pbc]
  [number of spectator site type][number of active site type]
  [os1][os2][os3]...
  [number sites]
  [site1a][site1b][site1c][site type]
  [site2a][site2b][site2c][site type]
  
  - ax,ay, ... are cell basis vector
  - pbc is either T or F indication of the periodic boundary condition
  - os# is the name of the possible occupation state (interpretted as string)
  - site1a,site1b,site1c are the scaled coordinates of site 1
  - site type can be either S1, S2, ... or A1, A2,... indicating spectator 
  ...
  Example:
  #Primitive Cell 
  2.81859800e+00  0.00000000e+00  0.00000000e+00 T
  -1.40929900e+00  2.44097800e+00  0.00000000e+00 T
  0.00000000e+00  0.00000000e+00  2.55082550e+01 T
  1 1
  -1 0 1
  6
  0.00000000e+00  0.00000000e+00  9.02210000e-02 S1
  6.66666666e-01  3.33333333e-01  1.80442000e-01 S1
  3.33333333e-01  6.66666666e-01  2.69674534e-01 S1
  0.00000000e+00  0.00000000e+00  3.58978557e-01 S1
  6.66666666e-01  3.33333333e-01  4.49958662e-01 S1
  3.33333333e-01  6.66666666e-01  5.01129144e-01 A1


  The input format for a data point Structure (Configuration of Atoms):
  ...
  [ax][ay][az]
  [bx][by][bz]
  [cx][cy][cz]
  [number sites]
  [site1a][site1b][site1c][site type][occupation state if active site]
  [site2a][site2b][site2c][site type][occupation state if active site]
  ...  
  - property value indicates the trained value. It must start with #y=...
  ...
  Example:
  #y=-1.209352
  2.81859800e+00  0.00000000e+00  0.00000000e+00
  -1.40929900e+00  2.44097800e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  2.55082550e+01
  6
  0.000000000000  0.000000000000  0.090220999986 S1
  0.500000499894  0.622008360788  0.180442000011 S1
  0.999999500106  0.666666711253  0.270892474701 S1
  0.000000000000  0.000000000000  0.361755713893 S1
  0.500000499894  0.622008360788  0.454395429618 S1
  0.000000000000  0.666667212896  0.502346789304 A1 1
  ...
  Python Script:
  ...
  >>> template_file_path = os.join.path(data_dir , "input.in")
  >>> Featurizer =  np.around(6.00,2)
  >>> Data_point_path = "Structure1.txt" 
  >>> Data_point_text = open(Data_point_path).read()
  >>> graph_ob = Featurizer._featurize(Data_point_text)
  >>> print(graph_ob)
  """

  def __init__(self, cutoff, template=None):
    """
    Parameters
    ----------
    cutoff: cutoff of radius for getting local environment.Only 
        used down to 2 digits.
    
    input_file_path: Template primitive stucture file path
    """
    self.cutoff = np.around(cutoff, 2)
    self.setup_env = SiteEnvironments.Load(template, cutoff)

  def _featurize(self, structure):
    """
    Parameters
    ----------
    structure: Raw text data input as a string 

    Returns
    -------
    obj.X_Sites: Node features
    obj.X_NSs: All edges for each node in diffrent permutations. Node 1 consist of
            6 diffrent permutation , each consisting of neighbors. 
    """
    xSites, xNSs = self.setup_env.ReadDatum(structure)
    return {"X_Sites": np.array(xSites), "X_NSs": np.array(xNSs)}


def InputReader(text, template=False):
  """
  Read Input Files

  Parameters
  ----------
  path : input text file

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


class SiteEnvironment(object):
  def __init__(self, pos, sitetypes, env2config, permutations, cutoff,\
               Grtol=0.0, Gatol=0.01, rtol=0.01, atol=0.0, tol=0.01, grtol=0.01):
    """ 
    Initialize site environment

    This class contains local site enrivonment information. This is used
    to find neighborlist in the datum (see GetMapping).

    Parameters
    ----------
    pos : n x 3 list or numpy array of (non-scaled) positions. n is the 
        number of atom.
    sitetypes : n list of string. String must be S or A followed by a 
        number. S indicates a spectator sites and A indicates a active 
        sites.
    permutations : p x n list of list of integer. p is the permutation 
        index and n is the number of sites.
    cutoff : float. cutoff used for pooling neighbors. for aesthetics only
    Grtol : relative tolerance in distance for forming an edge in graph
    Gatol : absolute tolerance in distance for forming an edge in graph
    rtol : relative tolerance in rmsd in distance for graph matching
    atol : absolute tolerance in rmsd in distance for graph matching
    tol : maximum tolerance of position RMSD to decide whether two 
        environment are the same
    grtol : tolerance for deciding symmetric nodes
    """
    self.pos = pos
    self.sitetypes = sitetypes
    self.activesiteidx = [i for i, s in enumerate(self.sitetypes) if 'A' in s]
    self.formula = defaultdict(int)
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
    self.G = self._ConstructGraph(pos, sitetypes)
    # matcher options
    self._nm = iso.categorical_node_match('n', '')
    self._em = iso.numerical_edge_match('d', 0, rtol, 0)

  def _ConstructGraph(self, pos, sitetypes):
    """
    Returns local environment graph using networkx and
    tolerance specified.

    parameters
    ----------
    pos: ns x 3. coordinates of positions. ns is the number of sites.
    sitetypes: ns. sitetype for each site

    Returns
    ------
    networkx graph used for matching site positions in
    datum. 
    """
    # construct graph
    G = nx.Graph()
    dists = cdist([[0, 0, 0]], pos - np.mean(pos, 0))[0]
    sdists = np.sort(dists)
    # https://stackoverflow.com/questions/37847053/uniquify-an-array-list-with-a-tolerance-in-python-uniquetol-equivalent
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

  def __repr__(self):
    s = '<' + self.sitetypes[0] + \
        '|%i active neighbors' % (len([s for s in self.sitetypes if 'A' in s]) - 1) + \
        '|%i spectator neighbors' % len([s for s in self.sitetypes if 'S' in s]) + \
        '|%4.2f Ang Cutoff' % self.cutoff + '| %i permutations>' % len(self.permutations)
    return s

  def __eq__(self, o):
    """
    Local environment comparison is done by comparing represented site
    """
    if not isinstance(o, SiteEnvironment):
      raise ValueError
    return self.sitetypes[0] == o.sitetypes[0]

  def __ne__(self, o):
    """
    Local environment comparison is done by comparing represented site
    """
    if isinstance(o, SiteEnvironment):
      raise ValueError
    return not self.__eq__(o)

  def GetMapping(self, env, path=None):
    """
    Returns mapping of sites from input to this object

    Pymatgen molecule_matcher does not work unfortunately as it needs to be
    a reasonably physical molecule.
    Here, the graph is constructed by connecting the nearest neighbor, and 
    isomorphism is performed to find matches, then kabsch algorithm is
    performed to make sure it is a match. NetworkX is used for portability.

    Parameters
    ----------
    env : dictionary that contains information of local environment of a 
        site in datum. See _GetSiteEnvironments defintion in the class
        SiteEnvironments for what this variable should be.

    Returns
    -------
    dict : atom mapping. None if there is no mapping
    """
    # construct graph
    G = self._ConstructGraph(env['pos'], env['sitetypes'])
    if len(self.G.nodes) != len(G.nodes):
      s = 'Number of nodes is not equal.\n'
      raise ValueError(s)
    elif len(self.G.edges) != len(G.edges):
      print(len(self.G.edges), len(G.edges))
      s = 'Number of edges is not equal.\n'
      s += "- Is the data point's cell a redefined lattice of primitive cell?\n"
      s += '- If relaxed structure is used, you may want to check structure or increase Gatol\n'
      if path:
        s += path
      raise ValueError(s)
    GM = iso.GraphMatcher(self.G, G, self._nm, self._em)
    # ####################### Most Time Consuming Part #####################
    ams = list(GM.isomorphisms_iter())
    # Perhaps parallelize it?
    # ####################### Most Time Consuming Part #####################
    if not ams:
      s = 'No isomorphism found.\n'
      s += "- Is the data point's cell a redefined lattice of primitive cell?\n"
      s += '- If relaxed structure is used, you may want to check structure or increase rtol\n'
      if path:
        s += path
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

  def _kabsch(self, P, Q):
    """
    Returns rotation matrix to align coordinates using
    Kabsch algorithm. 
    """
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
      S[-1] = -S[-1]
      V[:, -1] = -V[:, -1]
    R = np.dot(V, W)
    return R


class SiteEnvironments(object):

  def __init__(self, site_envs, ns, na, aos, eigen_tol, pbc, cutoff):
    """
    Initialize
    Use Load to intialize this class.

    Parameters
    ----------
    site_envs : list of SiteEnvironment object
    ns : int. number of spectator sites types
    na : int. number of active sites types
    aos : list of string. avilable occupational states for active sites
        string should be the name of the occupancy. (consistent with the input data)
    eigen_tol : tolerance for eigenanalysis of point group analysis in
        pymatgen.
    pbc : periodic boundary condition.
    cutoff : float. Cutoff radius in angstrom for pooling sites to 
        construct local environment 
    """
    self.site_envs = site_envs
    self.unique_site_types = [env.sitetypes[0] for env in self.site_envs]
    self.ns = ns
    self.na = na
    self.aos = aos
    self.eigen_tol = eigen_tol
    self.pbc = pbc
    self.cutoff = cutoff

  def __repr__(self):
    s = '<%i active sites' % (self.na) + '|%i spectator sites' % (self.ns) + '>'
    return s

  def __getitem__(self, el):
    """
    Returns a site environment
    """
    return self.site_envs[el]

  def ReadDatum(self, text, cutoff_factor=1.1):
    """
    Load structure data and return neighbor information

    Parameters
    ----------
    path : path of the structure
    cutoff_factor : float. this is extra buffer factor multiplied 
        to cutoff to ensure pooling all relevant sites. 
            
    Return
    ------
    XSites : one hot encoding of the site. See DataLoader in Data.py
        for detailed instruction.
    neighborlist : s x n x p x i. s is the type of site index, 
        n is the site index, p is the permutation,
        index and i is the neighbor sites index (0 being the site itself).
        See DataLoader in Data.py for detailed instruction.
    """
    cell, coord, st, oss = InputReader(text)
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
    site_envs = self._GetSiteEnvironments(
        coord,
        cell,
        st,
        self.cutoff * cutoff_factor,
        self.pbc,
        get_permutations=False,
        eigen_tol=self.eigen_tol)
    XNSs = [[] for _ in range(len(self.site_envs))]
    for env in site_envs:
      i = self.unique_site_types.index(env['sitetypes'][0])
      env = self._truncate(self.site_envs[i], env)

      # get map between two environment
      mapping = self.site_envs[i].GetMapping(env)
      # align input to the primitive cell (reference)
      aligned_idx = [
          env['env2config'][mapping[i]] for i in range(len(env['env2config']))
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
  def _truncate(cls, env_ref, env):
    """
    When cutoff_factor is used, it will pool more site than cutoff factor specifies.
    This will rule out nonrelevant sites by distance.
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

  @classmethod
  def Load(cls, path, cutoff, eigen_tol=1e-5):
    """
    Load Primitive cell and return SiteEnvironments

    Parameters
    ----------
    path : input file path
    cutoff : float. cutoff distance in angstrom for collecting local
    environment.
    eigen_tol : tolerance for eigenanalysis of point group analysis in
    pymatgen.

    """
    cell, pbc, coord, st, ns, na, aos = InputReader(path, template=True)
    site_envs = cls._GetSiteEnvironments(
        coord, cell, st, cutoff, pbc, True, eigen_tol=eigen_tol)
    site_envs = [
        SiteEnvironment(e['pos'], e['sitetypes'], e['env2config'],
                        e['permutations'], cutoff) for e in site_envs
    ]

    ust = [env.sitetypes[0] for env in site_envs]
    usi = np.unique(ust, return_index=True)[1]
    site_envs = [site_envs[i] for i in usi]
    return cls(site_envs, ns, na, aos, eigen_tol, pbc, cutoff)

  @classmethod
  def _GetSiteEnvironments(cls,
                           coord,
                           cell,
                           SiteTypes,
                           cutoff,
                           pbc,
                           get_permutations=True,
                           eigen_tol=1e-5):
    """
    Extract local environments from primitive cell

    Parameters
    ----------
    coord : n x 3 list or numpy array of scaled positions. n is the number 
        of atom.
    cell : 3 x 3 list or numpy array
    SiteTypes : n list of string. String must be S or A followed by a 
        number. S indicates a spectator sites and A indicates a active 
        sites.
    cutoff : float. cutoff distance in angstrom for collecting local
        environment.
    pbc : list of boolean. Periodic boundary condition
    get_permutations : boolean. Whether to find permutatated neighbor list or not.
    eigen_tol : tolerance for eigenanalysis of point group analysis in
        pymatgen.

    Returns
    ------
    list of local_env : list of local_env class
    """
    # %% Check error
    assert isinstance(coord, (list, np.ndarray))
    assert isinstance(cell, (list, np.ndarray))
    assert len(coord) == len(SiteTypes)
    # %% Initialize
    # TODO: Technically, user doesn't even have to supply site index, because
    #       pymatgen can be used to automatically categorize sites..
    coord = np.mod(coord, 1)
    pbc = np.array(pbc)
    # %% Map sites to other elements..
    # TODO: Available pymatgne functions are very limited when DummySpecie is
    #       involved. This may be perhaps fixed in the future. Until then, we
    #       simply bypass this by mapping site to an element
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
    # %% Get local environments of each site
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


def _chunks(length, number):
  """
  Yield successive n-sized chunks from length.
  """
  for i in range(0, len(length), number):
    yield length[i:i + number]
