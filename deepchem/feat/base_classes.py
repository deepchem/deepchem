"""
Feature calculations.
"""
import logging
import numpy as np
import multiprocessing
from typing import Any, Dict, List, Iterable, Sequence, Tuple

logger = logging.getLogger(__name__)


class Featurizer(object):
  """Abstract class for calculating a set of features for a datapoint.

  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. In
  that case, you might want to make a child class which
  implements the `_featurize` method for calculating features for
  a single datapoints if you'd like to make a featurizer for a
  new datatype.
  """

  def featurize(self, datapoints: Iterable[Any],
                log_every_n: int = 1000) -> np.ndarray:
    """Calculate features for datapoints.

    Parameters
    ----------
    datapoints: Iterable[Any]
      A sequence of objects that you'd like to featurize. Subclassses of
      `Featurizer` should instantiate the `_featurize` method that featurizes
      objects in the sequence.
    log_every_n: int, default 1000
      Logs featurization progress every `log_every_n` steps.

    Returns
    -------
    np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    datapoints = list(datapoints)
    features = []
    for i, point in enumerate(datapoints):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)
      try:
        features.append(self._featurize(point))
      except:
        logger.warning(
            "Failed to featurize datapoint %d. Appending empty array")
        features.append(np.array([]))

    features = np.asarray(features)
    return features

  def __call__(self, datapoints: Iterable[Any]):
    """Calculate features for datapoints.

    Parameters
    ----------
    datapoints: Iterable[Any]
      Any blob of data you like. Subclasss should instantiate this.
    """
    return self.featurize(datapoints)

  def _featurize(self, datapoint: Any):
    """Calculate features for a single datapoint.

    Parameters
    ----------
    datapoint: Any
      Any blob of data you like. Subclass should instantiate this.
    """
    raise NotImplementedError('Featurizer is not defined.')


class ComplexFeaturizer(object):
  """"
  Abstract class for calculating features for mol/protein complexes.
  """

  def featurize(self, mol_files: Sequence[str],
                protein_pdbs: Sequence[str]) -> Tuple[np.ndarray, List]:
    """
    Calculate features for mol/protein complexes.

    Parameters
    ----------
    mols: List[str]
      List of PDB filenames for molecules.
    protein_pdbs: List[str]
      List of PDB filenames for proteins.

    Returns
    -------
    features: np.ndarray
      Array of features
    failures: List
      Indices of complexes that failed to featurize.
    """

    pool = multiprocessing.Pool()
    results = []
    for i, (mol_file, protein_pdb) in enumerate(zip(mol_files, protein_pdbs)):
      log_message = "Featurizing %d / %d" % (i, len(mol_files))
      results.append(
          pool.apply_async(ComplexFeaturizer._featurize_callback,
                           (self, mol_file, protein_pdb, log_message)))
    pool.close()
    features = []
    failures = []
    for ind, result in enumerate(results):
      new_features = result.get()
      # Handle loading failures which return None
      if new_features is not None:
        features.append(new_features)
      else:
        failures.append(ind)
    features = np.asarray(features)
    return features, failures

  def _featurize(self, mol_pdb: str, complex_pdb: str):
    """
    Calculate features for single mol/protein complex.

    Parameters
    ----------
    mol_pdb : str
      The PDB filename.
    complex_pdb : str
      The PDB filename.
    """
    raise NotImplementedError('Featurizer is not defined.')

  @staticmethod
  def _featurize_callback(featurizer, mol_pdb_file, protein_pdb_file,
                          log_message):
    logging.info(log_message)
    return featurizer._featurize(mol_pdb_file, protein_pdb_file)


class MolecularFeaturizer(Featurizer):
  """Abstract class for calculating a set of features for a
  molecule.

  The defining feature of a `MolecularFeaturizer` is that it
  uses SMILES strings and RDKit molecule objects to represent
  small molecules. All other featurizers which are subclasses of
  this class should plan to process input which comes as smiles
  strings or RDKit molecules.

  Child classes need to implement the _featurize method for
  calculating features for a single molecule.

  Notes
  -----
  The subclasses of this class require RDKit to be installed.
  """

  def featurize(self, molecules, log_every_n=1000):
    """Calculate features for molecules.

    Parameters
    ----------
    molecules: rdkit.Chem.rdchem.Mol / SMILES string / iterable
      RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
      strings.
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.

    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import rdmolfiles
      from rdkit.Chem import rdmolops
      from rdkit.Chem.rdchem import Mol
    except ModuleNotFoundError:
      raise ValueError("This class requires RDKit to be installed.")

    # Special case handling of single molecule
    if isinstance(molecules, str) or isinstance(molecules, Mol):
      molecules = [molecules]
    else:
      # Convert iterables to list
      molecules = list(molecules)

    features = []
    for i, mol in enumerate(molecules):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)
      try:
        if isinstance(mol, str):
          # mol must be a RDKit Mol object, so parse a SMILES
          mol = Chem.MolFromSmiles(mol)
          # SMILES is unique, so set a canonical order of atoms
          new_order = rdmolfiles.CanonicalRankAtoms(mol)
          mol = rdmolops.RenumberAtoms(mol, new_order)
        features.append(self._featurize(mol))
      except:
        logger.warning(
            "Failed to featurize datapoint %d. Appending empty array")
        features.append(np.array([]))

    features = np.asarray(features)
    return features


class MaterialStructureFeaturizer(Featurizer):
  """
  Abstract class for calculating a set of features for an
  inorganic crystal structure.

  The defining feature of a `MaterialStructureFeaturizer` is that it
  operates on 3D crystal structures with periodic boundary conditions.
  Inorganic crystal structures are represented by Pymatgen structure
  objects. Featurizers for inorganic crystal structures that are subclasses of
  this class should plan to process input which comes as pymatgen
  structure objects.

  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. Child
  classes need to implement the _featurize method for calculating
  features for a single crystal structure.

  Notes
  -----
  Some subclasses of this class will require pymatgen and matminer to be
  installed.
  """

  def featurize(self,
                structures: Iterable[Dict[str, Any]],
                log_every_n: int = 1000) -> np.ndarray:
    """Calculate features for crystal structures.

    Parameters
    ----------
    structures: Iterable[Dict[str, Any]]
      Iterable sequence of pymatgen structure dictionaries.
      Dictionary representations of pymatgen.Structure
      https://pymatgen.org/pymatgen.core.structure.html
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.

    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of
      `structures`.
    """
    try:
      from pymatgen import Structure
    except ModuleNotFoundError:
      raise ValueError("This class requires pymatgen to be installed.")

    structures = list(structures)
    features = []
    for idx, structure in enumerate(structures):
      if idx % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % idx)
      try:
        s = Structure.from_dict(structure)
        features.append(self._featurize(s))
      except:
        logger.warning(
            "Failed to featurize datapoint %i. Appending empty array" % idx)
        features.append(np.array([]))

    features = np.asarray(features)
    return features


class MaterialCompositionFeaturizer(Featurizer):
  """
  Abstract class for calculating a set of features for an
  inorganic crystal composition.

  The defining feature of a `MaterialCompositionFeaturizer` is that it
  operates on 3D crystal chemical compositions.
  Inorganic crystal compositions are represented by Pymatgen composition
  objects. Featurizers for inorganic crystal compositions that are
  subclasses of this class should plan to process input which comes as
  Pymatgen composition objects.

  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. Child
  classes need to implement the _featurize method for calculating
  features for a single crystal composition.

  Notes
  -----
  Some subclasses of this class will require pymatgen and matminer to be
  installed.
  """

  def featurize(self, compositions: Iterable[str],
                log_every_n: int = 1000) -> np.ndarray:
    """Calculate features for crystal compositions.

    Parameters
    ----------
    compositions: Iterable[str]
      Iterable sequence of composition strings, e.g. "MoS2".
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.

    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of
      `compositions`.
    """
    try:
      from pymatgen import Composition
    except ModuleNotFoundError:
      raise ValueError("This class requires pymatgen to be installed.")

    compositions = list(compositions)
    features = []
    for idx, composition in enumerate(compositions):
      if idx % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % idx)
      try:
        c = Composition(composition)
        features.append(self._featurize(c))
      except:
        logger.warning(
            "Failed to featurize datapoint %i. Appending empty array" % idx)
        features.append(np.array([]))

    features = np.asarray(features)
    return features


class UserDefinedFeaturizer(Featurizer):
  """Directs usage of user-computed featurizations."""

  def __init__(self, feature_fields):
    """Creates user-defined-featurizer."""
    self.feature_fields = feature_fields
