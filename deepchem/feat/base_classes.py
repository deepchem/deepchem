"""
Feature calculations.
"""
import os
import logging
import types
import numpy as np
import multiprocessing
import logging
from deepchem.utils import rdkit_util

logger = logging.getLogger(__name__)


class Featurizer(object):
  """
  Abstract class for calculating a set of features for a
  datapoint.

  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. In
  that case, you might want to make a child class which
  implements the _featurize method for calculating features for
  a single datapoints if you'd like to make a featurizer for a
  new datatype.
  """

  def featurize(self, datapoints, log_every_n=1000):
    """
    Calculate features for datapoints.

    Parameters
    ----------
    datapoints: object 
       Any blob of data you like. Subclasss should instantiate
       this. 

    Returns
    -------
    A numpy array containing a featurized representation of
    `datapoints`.
    """
    raise NotImplementedError

  def __call__(self, datapoints):
    """
    Calculate features for datapoints.

    Parameters
    ----------
    datapoints: object 
       Any blob of data you like. Subclasss should instantiate
       this. 
    """
    return self.featurize(datapoints)


class MolecularFeaturizer(object):
  """
  Abstract class for calculating a set of features for a
  molecule.


  The defining feature of a `MolecularFeaturizer` is that it
  uses SMILES strings and RDKIT molecule objecgs to represent
  small molecules. All other featurizers which are subclasses of
  this class should plan to process input which comes as smiles
  strings or RDKIT molecules. 

  Child classes need to implement the _featurize method for
  calculating features for a single molecule.
  """

  def featurize(self, mols, log_every_n=1000):
    """
    Calculate features for molecules.

    Parameters
    ----------
    mols : iterable
        RDKit Mol, or SMILES string, or filename for
        mol2/sdf/pdb/pdbqt file.

    Returns
    -------
    A numpy array containing a featurized representation of
    `datapoints`.
    """
    from rdkit import Chem
    from rdkit.Chem.rdchem import Mol
    # Special case handling of single molecule
    if isinstance(mols, str) or isinstance(mols, Mol):
      mols = [mols]
    else:
      # Convert iterables to list
      mols = list(mols)
    features = []
    for i, mol in enumerate(mols):
      if mol is not None:
        if isinstance(mol, str):
          # Check if filename
          if os.path.exists(mol):
            _, mol = rdkit_util.load_molecule(mol)
          else:
            # mol must be a SMILES string so parse
            mol = Chem.MolFromSmiles(mol)
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
    mol : Object 
        Either a RDKit Mol or SMILES string.
    """
    raise NotImplementedError('Featurizer is not defined.')

  def __call__(self, mols):
    """
    Calculate features for molecules.

    Parameters
    ----------
    mols : iterable
        RDKit Mol or SMILES strings.
    """
    return self.featurize(mols)


class ReactionFeaturizer(object):
  """
  Abstract class for calculating a set of features for a
  reaction.

  Child classes implement the _featurize method for calculating
  features for a single reaction.
  """

  def featurize(self, rxns, log_every_n=1000):
    """
    Calculate features for reactions.

    Parameters
    ----------
    rxns: iterable
      Contains reactions in some representation. 
    """
    rxns = list(rxns)
    features = []
    for i, rxn in enumerate(rxns):
      if rxn is not None:
        features.append(self._featurize(rxn))
      else:
        features.append(np.array([]))

    features = np.asarray(features)
    return features

  def _featurize(self, rxn):
    """
    Calculate features for a single reaction.

    Parameters
    ----------
    rxn: Object 
        Reaction as some object..
    """
    raise NotImplementedError('Featurizer is not defined.')

  def __call__(self, rxns):
    """
    Calculate features for reactions.

    Parameters
    ----------
    rxns: iterable
        Contains reactions.
    """
    return self.featurize(rxns)


def _featurize_complex(featurizer, molecular_complex):
  return featurizer._featurize_complex(molecular_complex)


class ComplexFeaturizer(object):
  """"Abstract class for calculating features for mol/protein complexes.
  """

  def featurize_complexes(self, molecular_complexes, parallelize=True):
    """
    Calculate features for mol/protein complexes.

    Parameters
    ----------
    molecular_complexes: list
      Each entry in this list should be one or more files that contain
      molecular complexes. If a single file, then just a string with
      filename. If multiple files, then a tuple. 
    parallelize: bool
      Use multiprocessing to parallelize

    Returns
    -------
    features: np.array
      Array of features
    failures: list
      Indices of complexes that failed to featurize.
    """
    if parallelize:
      pool = multiprocessing.Pool()
      results = []
      for i, mol_complex in enumerate(molecular_complexes):
        logger.info("Featurizing %d / %d" % (i, len(molecular_complexes)))
        results.append(
            pool.apply_async(_featurize_complex, (self, mol_complex)))
      pool.close()
    else:
      results = []
      for i, mol_complex in enumerate(molecular_complexes):
        logger.info("Featurizing %d / %d" % (i, len(molecular_complexes)))
        results.append(_featurize_complex(self, mol_complex))
    features = []
    failures = []
    for ind, result in enumerate(results):
      if parallelize:
        new_features = result.get()
      else:
        new_features = result
      # Handle loading failures which return None
      if new_features is not None:
        features.append(new_features)
      else:
        failures.append(ind)
    features = np.asarray(features)
    return features, failures

  def _featurize_complex(self, mol_complex):
    """
    Calculate features for single mol/protein complex.

    Parameters
    ----------
    mol_complex: Object
      Should be some representation of a molecular complex.
    """
    raise NotImplementedError('Featurizer is not defined.')


class UserDefinedFeaturizer(Featurizer):
  """Directs usage of user-computed featurizations."""

  def __init__(self, feature_fields):
    """Creates user-defined-featurizer."""
    self.feature_fields = feature_fields


class ReactionFeaturizer(Featurizer):
  """Abstract class that featurizes reactions."""

  def _featurize(self, smarts):
    """"
    Calculate features for a single reaction.

    Parameters
    ----------
    smarts: str
      SMARTS string defining reaction.
    """
    raise NotImplementedError('Featurizer is not defined')
