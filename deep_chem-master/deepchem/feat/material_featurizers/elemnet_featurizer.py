import numpy as np
from typing import DefaultDict, Optional

from deepchem.utils.typing import PymatgenComposition
from deepchem.feat import MaterialCompositionFeaturizer

elements_tl = [
    'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
    'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
    'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'
]


class ElemNetFeaturizer(MaterialCompositionFeaturizer):
  """
  Fixed size vector of length 86 containing raw fractional elemental
  compositions in the compound. The 86 chosen elements are based on the
  original implementation at https://github.com/NU-CUCIS/ElemNet.

  Returns a vector containing fractional compositions of each element
  in the compound.

  References
  ----------
  .. [1] Jha, D., Ward, L., Paul, A. et al. Sci Rep 8, 17593 (2018).
     https://doi.org/10.1038/s41598-018-35934-y

  Examples
  --------
  >>> import pymatgen as mg
  >>> comp = "Fe2O3"
  >>> featurizer = ElemNetFeaturizer()
  >>> features = featurizer.featurize([comp])

  Note
  ----
  This class requires Pymatgen to be installed.
  """

  def get_vector(self, comp: DefaultDict) -> Optional[np.ndarray]:
    """
    Converts a dictionary containing element names and corresponding
    compositional fractions into a vector of fractions.

    Parameters
    ----------
    comp: collections.defaultdict object
      Dictionary mapping element names to fractional compositions.

    Returns
    -------
    fractions: np.ndarray
      Vector of fractional compositions of each element.
    """
    if all(e in elements_tl for e in comp):
      return np.array([comp[e] if e in comp else 0 for e in elements_tl],
                      np.float32)
    else:
      return None

  def _featurize(self,
                 composition: PymatgenComposition) -> Optional[np.ndarray]:
    """
    Calculate 86 dimensional vector containing fractional compositions of
    each element in the compound.

    Parameters
    ----------
    composition: pymatgen.core.Composition object
      Composition object.

    Returns
    -------
    feats: np.ndarray
      86 dimensional vector containing fractional compositions of elements.
    """
    fractions = composition.fractional_composition.get_el_amt_dict()
    return self.get_vector(fractions)
