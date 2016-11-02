"""
Topological fingerprints.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "LGPL v2.1+"

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from deepchem.feat import Featurizer


class CircularFingerprint(Featurizer):
  """
  Circular (Morgan) fingerprints.

  Parameters
  ----------
  radius : int, optional (default 2)
      Fingerprint radius.
  size : int, optional (default 2048)
      Length of generated bit vector.
  chiral : bool, optional (default False)
      Whether to consider chirality in fingerprint generation.
  bonds : bool, optional (default True)
      Whether to consider bond order in fingerprint generation.
  features : bool, optional (default False)
      Whether to use feature information instead of atom information; see
      RDKit docs for more info.
  sparse : bool, optional (default False)
      Whether to return a dict for each molecule containing the sparse
      fingerprint.
  smiles : bool, optional (default False)
      Whether to calculate SMILES strings for fragment IDs (only applicable
      when calculating sparse fingerprints).
  """
  name = 'circular'

  def __init__(self, radius=2, size=2048, chiral=False, bonds=True,
             features=False, sparse=False, smiles=False):
    self.radius = radius
    self.size = size
    self.chiral = chiral
    self.bonds = bonds
    self.features = features
    self.sparse = sparse
    self.smiles = smiles

  def _featurize(self, mol):
    """
    Calculate circular fingerprint.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    if self.sparse:
      info = {}
      fp = rdMolDescriptors.GetMorganFingerprint(
          mol, self.radius, useChirality=self.chiral,
          useBondTypes=self.bonds, useFeatures=self.features,
          bitInfo=info)
      fp = fp.GetNonzeroElements()  # convert to a dict

      # generate SMILES for fragments
      if self.smiles:
        fp_smiles = {}
        for fragment_id, count in fp.items():
          root, radius = info[fragment_id][0]
          env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, root)
          frag = Chem.PathToSubmol(mol, env)
          smiles = Chem.MolToSmiles(frag)
          fp_smiles[fragment_id] = {'smiles': smiles, 'count': count}
        fp = fp_smiles
    else:
      fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
          mol, self.radius, nBits=self.size, useChirality=self.chiral,
          useBondTypes=self.bonds, useFeatures=self.features)
    return fp
