import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer


class MACCSKeysFingerprint(MolecularFeaturizer):
  """MACCS Keys Fingerprint.

  The MACCS (Molecular ACCess System) keys are one of the most commonly used structural keys.
  Please confirm the details in [1]_, [2]_.

  Examples
  --------
  >>> import deepchem as dc
  >>> smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
  >>> featurizer = dc.feat.MACCSKeysFingerprint()
  >>> features = featurizer.featurize([smiles])
  >>> type(features[0])
  <class 'numpy.ndarray'>
  >>> features[0].shape
  (167,)

  References
  ----------
  .. [1] Durant, Joseph L., et al. "Reoptimization of MDL keys for use in drug discovery."
     Journal of chemical information and computer sciences 42.6 (2002): 1273-1280.
  .. [2] https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py

  Note
  ----
  This class requires RDKit to be installed.

  """

  def __init__(self):
    """Initialize this featurizer."""
    self.calculator = None

  def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
    """
    Calculate MACCS keys fingerprint.

    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      1D array of RDKit descriptors for `mol`. The length is 167.
    """
    if 'mol' in kwargs:
      datapoint = kwargs.get("mol")
      raise DeprecationWarning(
          'Mol is being phased out as a parameter, please pass "datapoint" instead.'
      )

    if self.calculator is None:
      try:
        from rdkit.Chem.AllChem import GetMACCSKeysFingerprint
        self.calculator = GetMACCSKeysFingerprint
      except ModuleNotFoundError:
        raise ImportError("This class requires RDKit to be installed.")

    return self.calculator(datapoint)
