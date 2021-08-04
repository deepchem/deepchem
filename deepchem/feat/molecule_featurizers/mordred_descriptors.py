from typing import Callable, List, Optional
import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer


class MordredDescriptors(MolecularFeaturizer):
  """Mordred descriptors.

  This class computes a list of chemical descriptors using Mordred.
  Please see the details about all descriptors from [1]_, [2]_.

  Attributes
  ----------
  descriptors: List[str]
    List of Mordred descriptor names used in this class.

  References
  ----------
  .. [1] Moriwaki, Hirotomo, et al. "Mordred: a molecular descriptor calculator."
     Journal of cheminformatics 10.1 (2018): 4.
  .. [2] http://mordred-descriptor.github.io/documentation/master/descriptors.html

  Note
  ----
  This class requires Mordred to be installed.

  Examples
  --------
  >>> import deepchem as dc
  >>> smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
  >>> featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
  >>> features = featurizer.featurize(smiles)
  >>> type(features[0])
  <class 'numpy.ndarray'>
  >>> features[0].shape
  (1613,)

  """

  def __init__(self, ignore_3D: bool = True):
    """
    Parameters
    ----------
    ignore_3D: bool, optional (default True)
      Whether to use 3D information or not.
    """
    self.ignore_3D = ignore_3D
    self.calc: Optional[Callable] = None
    self.descriptors: Optional[List] = None

  def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
    """
    Calculate Mordred descriptors.

    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      1D array of Mordred descriptors for `mol`.
      If ignore_3D is True, the length is 1613.
      If ignore_3D is False, the length is 1826.
    """
    if 'mol' in kwargs:
      datapoint = kwargs.get("mol")
      raise DeprecationWarning(
          'Mol is being phased out as a parameter, please pass "datapoint" instead.'
      )
    if self.calc is None:
      try:
        from mordred import Calculator, descriptors, is_missing
        self.is_missing = is_missing
        self.calc = Calculator(descriptors, ignore_3D=self.ignore_3D)
        self.descriptors = list(descriptors.__all__)
      except ModuleNotFoundError:
        raise ImportError("This class requires Mordred to be installed.")

    feature = self.calc(datapoint)
    # convert errors to zero
    feature = [
        0.0 if self.is_missing(val) or isinstance(val, str) else val
        for val in feature
    ]
    return np.asarray(feature)
