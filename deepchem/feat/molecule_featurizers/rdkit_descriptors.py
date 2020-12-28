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

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self, use_fragment=True, ipc_avg=True):
    """Initialize this featurizer.

    Parameters
    ----------
    use_fragment: bool, optional (default True)
      If True, the return value includes the fragment binary descriptors like 'fr_XXX'.
    ipc_avg: bool, optional (default True)
      If True, the IPC descriptor calculates with avg=True option.
      Please see this issue: https://github.com/rdkit/rdkit/issues/1527.
    """
    self.use_fragment = use_fragment
    self.ipc_avg = ipc_avg
    self.descriptors = []
    self.descList = []

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
      1D array of RDKit descriptors for `mol`.
      The length is `len(self.descriptors)`.
    """
    # initialize
    if len(self.descList) == 0:
      try:
        from rdkit.Chem import Descriptors
        for descriptor, function in Descriptors.descList:
          if self.use_fragment is False and descriptor.startswith('fr_'):
            continue
          self.descriptors.append(descriptor)
          self.descList.append((descriptor, function))
      except ModuleNotFoundError:
        raise ImportError("This class requires RDKit to be installed.")

    # check initialization
    assert len(self.descriptors) == len(self.descList)

    features = []
    for desc_name, function in self.descList:
      if desc_name == 'Ipc' and self.ipc_avg:
        feature = function(mol, avg=True)
      else:
        feature = function(mol)
      features.append(feature)
    return np.asarray(features)
