import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer


class MACCSKeysFingerprint(MolecularFeaturizer):
  """MACCS Keys Fingerprint.

  The MACCS (Molecular ACCess System) keys are one of the most commonly used structural keys.
  Please confirm the details in [1]_, [2]_.

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

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    """
    Calculate MACCS keys fingerprint.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      1D array of RDKit descriptors for `mol`. The length is 167.
    """
    if self.calculator is None:
      try:
        from rdkit.Chem.AllChem import GetMACCSKeysFingerprint
        self.calculator = GetMACCSKeysFingerprint
      except ModuleNotFoundError:
        raise ImportError("This class requires RDKit to be installed.")

    return self.calculator(mol)
