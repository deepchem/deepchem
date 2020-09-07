"""
Basic molecular features.
"""

import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer


class RDKitDescriptors(MolecularFeaturizer):
  """RDKit descriptors.

  This class computes a list of chemical descriptors using RDKit.

  Attributes
  ----------
  descriptors: List[str]
    List of RDKit descriptor names used in this class.

  Notes
  -----
  This class requires RDKit to be installed.
  """

  def __init__(self):
    try:
      from rdkit.Chem import Descriptors
    except ModuleNotFoundError:
      raise ValueError("This class requires RDKit to be installed.")

    self.descriptors = []
    self.descList = []
    for descriptor, function in Descriptors.descList:
      self.descriptors.append(descriptor)
      self.descList.append((descriptor, function))

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    """
    Calculate RDKit descriptors.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      1D array of RDKit descriptors for `mol`. The length is 200.
    """
    rval = []
    for desc_name, function in self.descList:
      rval.append(function(mol))
    return np.asarray(rval)
