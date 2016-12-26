"""
Feature calculations.
"""
import types
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdGeometry, rdMolTransforms
from deepchem.utils.save import log

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "BSD 3-clause"

class ComplexFeaturizer(object):
  """"
  Abstract class for calculating features for mol/protein complexes.
  """

  def featurize_complexes(self, mol_pdbs, protein_pdbs, verbose=True,
                          log_every_n=1000):
    """
    Calculate features for mol/protein complexes.

    Parameters
    ----------
    mol_pdbs: list
      List of PDBs for molecules. Each PDB should be a list of lines of the
      PDB file.
    protein_pdbs: list
      List of PDBs for proteins. Each PDB should be a list of lines of the
      PDB file.
    """
    features = []
    for i, (mol_pdb, protein_pdb) in enumerate(zip(mol_pdbs, protein_pdbs)):
      if verbose and i % log_every_n == 0:
        log("Featurizing %d / %d" % (i, len(mol_pdbs)))
      features.append(self._featurize_complex(mol_pdb, protein_pdb))
    features = np.asarray(features)
    return features

  def _featurize_complex(self, mol_pdb, complex_pdb):
    """
    Calculate features for single mol/protein complex.

    Parameters
    ----------
    mol_pdb: list
      Should be a list of lines of the PDB file.
    complex_pdb: list
      Should be a list of lines of the PDB file.
    """
    raise NotImplementedError('Featurizer is not defined.')

class Featurizer(object):
  """
  Abstract class for calculating a set of features for a molecule.

  Child classes implement the _featurize method for calculating features
  for a single molecule.
  """

  def featurize(self, mols, verbose=True, log_every_n=1000):
    """
    Calculate features for molecules.

    Parameters
    ----------
    mols : iterable
        RDKit Mol objects.
    """
    mols = list(mols)
    features = []
    for i, mol in enumerate(mols):
      if mol is not None:
        features.append(self._featurize(mol))
      else:
        features.append(np.array([]))

    features = np.asarray(features)
    return features

  def _featurize(self, mol):
    """
    Calculate features for a single molecule.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    raise NotImplementedError('Featurizer is not defined.')

  def __call__(self, mols):
    """
    Calculate features for molecules.

    Parameters
    ----------
    mols : iterable
        RDKit Mol objects.
    """
    return self.featurize(mols)


class UserDefinedFeaturizer(Featurizer):
  """Directs usage of user-computed featurizations."""

  def __init__(self, feature_fields):
    """Creates user-defined-featurizer."""
    self.feature_fields = feature_fields
