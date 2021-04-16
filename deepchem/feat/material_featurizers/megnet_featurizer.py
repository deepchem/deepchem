import numpy as np
from deepchem.feat import MaterialStructureFeaturizer
from deepchem.utils.typing import PymatgenStructure
from typing import Dict, Callable


class MegnetFeaturizer(MaterialStructureFeaturizer):
  """
    Calculate structure graph features for crystals

    Based on the implementation of "Graph Networks as a Universal
    Machine Learning Framework for Molecules and Crystals" (MEGNET).
    The method constructs a crystal graph representation including
    atom features and bond features (neighbor distances). Neighbors
    are determined by searching in a sphere around atoms in the unit
    cell. A Gaussian filter is applied to neighbor distances.
    All units are in angstrom.

    1. Node feature -  The atomic number of element (1-94)
    2. Edge feature -  Expanded distance with Gaussian basis exp(−(r − r0)^2/σ2)
        centered at 100 points linearly placed between 0 and 5 and σ = 0.5

    References
    ----------
    .. [1] Chi Chen et al, Chem. Mater. 2019, 31, 9, 3564–3572

    Examples
    --------
    >>> import pymatgen as mg
    >>> lattice = mg.Lattice.cubic(4.2)
    >>> structure = mg.Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> featurizer = MegnetFeaturizer(bond_edge=5)
    >>> features = featurizer.featurize([structure])
    >>> feature = features[0]
    >>> print(feature.keys())
    dict_keys(['atom', 'bond', 'state', 'index1', 'index2'])

    Notes
    -----
    This Class requires pymatgen , networkx , scipy installed.
    """

  def __init__(self,
               atom_converter: Callable = None,
               center: int = 100,
               width: float = 0.5,
               bond_converter: Callable = None,
               cutoff: float = 5.0):
    """
        Parameters
        ----------
        atom_converter: Callable
            A function to convert Atomic weight into corresponding embedding vectors.
        center: int
            centers for the Gaussian basis
        width: float
            width of Gaussian basis
        bond_converter: Callable
        cutoff: float
            cutoff radius
        """
    try:
      from megnet.data.crystal import CrystalGraph
      from megnet.data.graph import GaussianDistance
    except:
      raise ImportError(
          "This class requires MEGNET and Pymatgen to be installed.")

    self.atom_converter = atom_converter
    self.bond_converter = bond_converter
    if bond_converter is None:
      self.bond_converter = GaussianDistance(np.linspace(0, 5, center), width)
    self.cnv = CrystalGraph(
        atom_converter=self.atom_converter,
        bond_converter=self.bond_converter,
        cutoff=cutoff)

  def _featurize(self, struct: PymatgenStructure) -> Dict[str, list]:
    output = self.cnv.convert(struct)
    if self.atom_converter is not None:
      output["atom"] = self.atom_converter.convert(output["atom"])
    if self.bond_converter is not None:
      output["bond"] = self.bond_converter.convert(output["bond"])
    return output
