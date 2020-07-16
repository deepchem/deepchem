import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from deepchem.data import DiskDataset
from deepchem.feat import Featurizer
from deepchem.data.data_loader.base_loader import DataLoader

logger = logging.getLogger(__name__)


# NOTE: We should remove...
def load_csv_files(filenames, shard_size=None, verbose=True):
  """Load data as pandas dataframe."""
  # First line of user-specified CSV *must* be header.
  shard_num = 1
  for filename in filenames:
    if shard_size is None:
      yield pd.read_csv(filename)
    else:
      log("About to start loading CSV from %s" % filename, verbose)
      for df in pd.read_csv(filename, chunksize=shard_size):
        log("Loading shard %d of size %s." % (shard_num, str(shard_size)),
            verbose)
        df = df.replace(np.nan, str(""), regex=True)
        shard_num += 1
        yield df


class SDFLoader(DataLoader):
  """
  Creates `Dataset` from SDF input files. 

  This class provides conveniences to load data from SDF files.
  """

  def __init__(self,
               tasks: List[str],
               sanitize: Optional[bool] = False,
               featurizer: Optional[Featurizer] = None,
               log_every_n: Optional[int] = 1000):
    """Initialize SDF Loader

    Parameters
    ----------
    tasks: list[str]
      List of tasknames. These will be loaded from the SDF file.
    sanitize: bool, optional (default False)
      Whether to sanitize molecules.
    featurizer: dc.feat.Featurizer, optional
      Featurizer to use to process data
    log_every_n: int, optional (default 1000)
      Writes a logging statement this often.
    """
    self.featurizer = featurizer
    self.sanitize = sanitize
    self.tasks = tasks
    # The field in which dc.utils.save.load_sdf_files stores
    # RDKit mol objects
    self.mol_field = "mol"
    # The field in which load_sdf_files return value stores
    # smiles
    self.id_field = "smiles"
    self.log_every_n = log_every_n

  # NOTE : why this method consider about csv file...?
  def _get_shards(self, input_files, shard_size):
    """Defines a generator which returns data for each shard"""
    from rdkit import Chem

    dataframes = []
    for input_file in input_files:
      # Tasks are either in .sdf.csv file or in the .sdf file itself
      has_csv = os.path.isfile(input_file + ".csv")
      # Structures are stored in .sdf file
      print("Reading structures from %s." % input_file)
      suppl = Chem.SDMolSupplier(str(input_file), self.sanitize, False, False)
      df_rows = []
      for ind, mol in enumerate(suppl):
        if mol is None:
          continue
        smiles = Chem.MolToSmiles(mol)
        df_row = [ind, smiles, mol]
        if not has_csv:  # Get task targets from .sdf file
          for task in self.tasks:
            df_row.append(mol.GetProp(str(task)))
        df_rows.append(df_row)
      if has_csv:
        mol_df = pd.DataFrame(df_rows, columns=('mol_id', 'smiles', 'mol'))
        # NOTE : why load csv...?
        raw_df = next(load_csv_files([input_file + ".csv"], shard_size=None))
        dataframes.append(pd.concat([mol_df, raw_df], axis=1, join='inner'))
      else:
        mol_df = pd.DataFrame(
            df_rows, columns=('mol_id', 'smiles', 'mol') + tuple(self.tasks))
        dataframes.append(mol_df)
    return dataframes

  def _featurize_shard(self,
                       shard: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Featurize individual samples in dataframe.

    Helper that given a featurizer that operates on individual
    samples, computes & adds features for that sample to the 
    features dataframe.

    Parameters
    ----------
    shard: pd.DataFrame
      DataFrame that holds data to be featurized.

    Returns
    -------
    features : np.ndarray
      Array of feature vectors.
    valid_inds : np.ndarray
      Boolean values indicating successfull featurization.
    """
    logger.info("Currently featurizing feature_type: %s" %
                self.featurizer.__class__.__name__)
    sample_elems = shard[self.mol_field].tolist()

    features = []
    valid_inds = []
    for ind, mol in enumerate(sample_elems):
      # logging
      if ind % self.log_every_n == 0:
        logger.info("Featurizing sample %d" % ind)

      feat = self.featurizer.featurize([mol])
      is_valid = True if feat.size > 0 else False
      valid_inds.append(is_valid)
      if is_valid:
        features.append(feat)

    return np.squeeze(np.array(features), axis=1), valid_inds
