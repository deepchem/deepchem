import time
import logging
import pandas as pd
import numpy as np
from deepchem.data import DiskDataset
from deepchem.feat import UserDefinedFeaturizer
from deepchem.data.data_loader.base_loader import DataLoader

logger = logging.getLogger(__name__)


class CSVLoader(DataLoader):
  """
  Creates `Dataset` objects from input CSV files. 

  This class provides conveniences to load data from CSV files.
  It's possible to directly featurize data from CSV files using
  pandas, but this class may prove useful if you're processing
  large CSV files that you don't want to manipulate directly in
  memory.
  """

  def __init__(self,
               tasks,
               smiles_field=None,
               id_field=None,
               featurizer=None,
               log_every_n=1000):
    """Initializes CSVLoader.

    Parameters
    ----------
    tasks: list[str]
      List of task names
    smiles_field: str, optional
      Name of field that holds smiles string 
    id_field: str, optional
      Name of field that holds sample identifier
    featurizer: dc.feat.Featurizer, optional
      Featurizer to use to process data
    log_every_n: int, optional
      Writes a logging statement this often.
    """
    if not isinstance(tasks, list):
      raise ValueError("tasks must be a list.")
    self.tasks = tasks
    self.smiles_field = smiles_field
    if id_field is None:
      self.id_field = smiles_field
    else:
      self.id_field = id_field
    #self.mol_field = mol_field
    self.user_specified_features = None
    if isinstance(featurizer, UserDefinedFeaturizer):
      self.user_specified_features = featurizer.feature_fields
    self.featurizer = featurizer
    self.log_every_n = log_every_n

  def _get_shards(self, input_files, shard_size):
    """Defines a generator which returns data for each shard"""
    shard_num = 1
    for filename in input_files:
      if shard_size is None:
        yield pd.read_csv(filename)
      else:
        logger.info("About to start loading CSV from %s" % filename)
        for df in pd.read_csv(filename, chunksize=shard_size):
          logger.info(
              "Loading shard %d of size %s." % (shard_num, str(shard_size)))
          df = df.replace(np.nan, str(""), regex=True)
          shard_num += 1
          yield df

  def _featurize_shard(self, shard):
    """Featurize individual compounds in dataframe.
  
    Private helper that given a featurizer that operates on individual
    chemical compounds or macromolecules, compute & add features for
    that compound to the features dataframe
  
    Parameters
    ----------
    shard: pd.DataFrame
      DataFrame that holds SMILES strings
    """
    sample_elems = shard[self.smiles_field].tolist()

    from rdkit import Chem
    from rdkit.Chem import rdmolfiles
    from rdkit.Chem import rdmolops

    features = []
    valid_inds = []
    for ind, elem in enumerate(sample_elems):
      mol = Chem.MolFromSmiles(elem)
      # TODO (ytz) this is a bandage solution to reorder the atoms
      # so that they're always in the same canonical order.
      # Presumably this should be correctly implemented in the
      # future for graph mols.
      if mol:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

      # logging
      if ind % self.log_every_n == 0:
        logger.info("Featurizing sample %d" % ind)

      # add feature
      feat = self.featurizer.featurize([mol])
      is_valid = True if feat.size > 0 else False
      valid_inds.append(is_valid)
      if is_valid:
        features.append(feat)

    return np.squeeze(np.array(features), axis=1), valid_inds


class UserCSVLoader(CSVLoader):
  """
  Handles loading of CSV files with user-defined featurizers.
  """

  def _featurize_shard(self, shard):
    """Extract and merge user specified features.

    Private helper methods that merges features included in dataset
    provided by user into final features dataframe
  
    Three types of featurization here:
  
      1) Molecule featurization
        -) Smiles string featurization
        -) Rdkit MOL featurization
      2) Complex featurization
        -) PDB files for interacting molecules.
      3) User specified featurizations.
  
    Parameters
    ----------
    shard: pd.DataFrame
      DataFrame that holds SMILES strings
    """
    assert isinstance(self.featurizer, UserDefinedFeaturizer)
    time1 = time.time()
    shard[self.featurizer.feature_fields] = \
      shard[self.featurizer.feature_fields].apply(pd.to_numeric)
    X_shard = shard[self.featurizer.feature_fields].to_numpy()
    time2 = time.time()
    logger.info(
        "TIMING: user specified processing took %0.3f s" % (time2 - time1))
    return (X_shard, np.ones(len(X_shard), dtype=bool))
