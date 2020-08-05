import numpy as np

from deepchem.utils.typing import PymatgenComposition
from deepchem.feat import MaterialCompositionFeaturizer


class ElementPropertyFingerprint(MaterialCompositionFeaturizer):
  """
  Fingerprint of elemental properties from composition.

  Based on the data source chosen, returns properties and statistics
  (min, max, range, mean, standard deviation, mode) for a compound
  based on elemental stoichiometry. E.g., the average electronegativity
  of atoms in a crystal structure. The chemical fingerprint is a
  vector of these statistics. For a full list of properties and statistics,
  see ``matminer.featurizers.composition.ElementProperty(data_source).feature_labels()``.

  This featurizer requires the optional dependencies pymatgen and
  matminer. It may be useful when only crystal compositions are available
  (and not 3D coordinates).

  See references [1]_ [2]_ [3]_ [4]_ for more details.

  References
  ----------
  .. [1] MagPie data: Ward, L. et al. npj Comput Mater 2, 16028 (2016).
     https://doi.org/10.1038/npjcompumats.2016.28
  .. [2] Deml data: Deml, A. et al. Physical Review B 93, 085142 (2016).
     10.1103/PhysRevB.93.085142
  .. [3] Matminer: Ward, L. et al. Comput. Mater. Sci. 152, 60-69 (2018).
  .. [4] Pymatgen: Ong, S.P. et al. Comput. Mater. Sci. 68, 314-319 (2013).

  Examples
  --------
  >>> import pymatgen as mg
  >>> comp = mg.Composition("Fe2O3")
  >>> featurizer = ElementPropertyFingerprint()
  >>> features = featurizer.featurize([comp])

  Notes
  -----
  This class requires matminer and Pymatgen to be installed.
  `NaN` feature values are automatically converted to 0 by this featurizer.
  """

  def __init__(self, data_source: str = 'matminer'):
    """
    Parameters
    ----------
    data_source: str of "matminer", "magpie" or "deml" (default "matminer")
      Source for element property data.
    """

    self.data_source = data_source

  def _featurize(self, composition: PymatgenComposition) -> np.ndarray:
    """
    Calculate chemical fingerprint from crystal composition.

    Parameters
    ----------
    composition: pymatgen.Composition object
      Composition object.

    Returns
    -------
    feats: np.ndarray
      Vector of properties and statistics derived from chemical
      stoichiometry. Some values may be NaN.
    """
    try:
      from matminer.featurizers.composition import ElementProperty
    except ModuleNotFoundError:
      raise ValueError("This class requires matminer to be installed.")

    ep = ElementProperty.from_preset(self.data_source)

    try:
      feats = ep.featurize(composition)
    except:
      feats = []

    return np.nan_to_num(np.array(feats))
