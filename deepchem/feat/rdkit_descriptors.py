"""
Basic molecular features.
"""

import numpy as np
from deepchem.feat.base_classes import MolecularFeaturizer


class RDKitDescriptors(MolecularFeaturizer):
  """RDKit descriptors.

  This class comptues a list of chemical descriptors using RDKit.

  See http://rdkit.org/docs/GettingStartedInPython.html
  #list-of-available-descriptors.

  Attributes
  ----------
  descriptors: np.ndarray
    1D array of RDKit descriptor names used in this class.

  Note
  ----
  This class requires RDKit to be installed.
  """

  # (ytz): This is done to avoid future compatibility issues like inclusion of
  # the 3D descriptors or changing the feature size.
  allowedDescriptors = set([
      'MaxAbsPartialCharge', 'MinPartialCharge', 'MinAbsPartialCharge',
      'HeavyAtomMolWt', 'MaxAbsEStateIndex', 'NumRadicalElectrons',
      'NumValenceElectrons', 'MinAbsEStateIndex', 'MaxEStateIndex',
      'MaxPartialCharge', 'MinEStateIndex', 'ExactMolWt', 'MolWt', 'BalabanJ',
      'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n',
      'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc',
      'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10',
      'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2',
      'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
      'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3',
      'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9',
      'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2',
      'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
      'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10',
      'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4',
      'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
      'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
      'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
      'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',
      'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
      'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
      'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
      'NumRotatableBonds', 'NumSaturatedCarbocycles',
      'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP',
      'MolMR'
  ])

  def __init__(self):
    try:
      from rdkit.Chem import Descriptors
    except ModuleNotFoundError:
      raise ValueError("This class requires RDKit to be installed.")
    self.descriptors = []
    self.descList = []
    for descriptor, function in Descriptors.descList:
      if descriptor in self.allowedDescriptors:
        self.descriptors.append(descriptor)
        self.descList.append((descriptor, function))

  def _featurize(self, mol):
    """
    Calculate RDKit descriptors.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.

    Returns
    -------
    rval: np.ndarray
      1D array of RDKit descriptors for `mol`
    """
    rval = []
    for desc_name, function in self.descList:
      rval.append(function(mol))
    return np.asarray(rval)
