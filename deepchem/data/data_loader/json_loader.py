import time
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Iterator
from deepchem.data import DiskDataset
from deepchem.feat import UserDefinedFeaturizer, Featurizer
from deepchem.data.data_loader.base_loader import DataLoader

logger = logging.getLogger(__name__)


class JsonLoader(DataLoader):
  """
  Creates `Dataset` objects from input json files. 

  This class provides conveniences to load data from json files.
  It's possible to directly featurize data from json files using
  pandas, but this class may prove useful if you're processing
  large json files that you don't want to manipulate directly in
  memory.

  It is meant to load JSON files formatted as "records" in line
  delimited format, which allows for sharding.
  ``list like [{column -> value}, ... , {column -> value}]``.

  Examples
  --------
  >> import pandas as pd
  >> df = pd.DataFrame(some_data)
  >> df.columns.tolist()
  .. ['sample_data', 'sample_name', 'weight', 'task']
  >> df.to_json('file.json', orient='records', lines=True)
  >> loader = JsonLoader(tasks=['task'], feature_field='sample_data',
      label_field='task', weight_field='weight', id_field='sample_name')
  >> dataset = loader.create_dataset('file.json')
  
  """

  def __init__(self,
               tasks: List[str],
               feature_field: str,
               label_field: str = None,
               weight_field: str = None,
               id_field: str = None,
               featurizer: Optional[Featurizer] = None,
               log_every_n: int = 1000):
    """Initializes JsonLoader.

    Parameters
    ----------
    tasks : List[str]
      List of task names
    feature_field : str
      JSON field with data to be featurized.
    label_field : str, default None
      Field with target variables.
    weight_field : str, default None
      Field with weights.
    id_field : str, default None
      Field for identifying samples.
    featurizer : dc.feat.Featurizer, optional
      Featurizer to use to process data
    log_every_n : int, optional
      Writes a logging statement this often.

    """

    if not isinstance(tasks, list):
      raise ValueError("Tasks must be a list.")
    self.tasks = tasks
    self.feature_field = feature_field
    self.label_field = label_field
    self.weight_field = weight_field
    self.id_field = id_field

    self.user_specified_features = None
    if isinstance(featurizer, UserDefinedFeaturizer):
      self.user_specified_features = featurizer.feature_fields
    self.featurizer = featurizer
    self.log_every_n = log_every_n

  def create_dataset(self,
                     input_files: List[str],
                     data_dir: Optional[str] = None,
                     shard_size: Optional[int] = 8192) -> DiskDataset:
    """Creates a `Dataset` from input JSON files.

    Parameters
    ----------
    input_files: List[str]
      List of JSON filenames.
    data_dir: Optional[str], default None
      Name of directory where featurized data is stored.
    shard_size: Optional[int], default 8192
      Shard size when loading data.

    Returns
    -------
    dataset: dc.data.Dataset
      A `Dataset` object containing a featurized representation of data
      from `input_files`.

    """

    if not isinstance(input_files, list):
      input_files = [input_files]

    def shard_generator():
      """Yield X, y, w, and ids for shards."""
      for shard_num, shard in enumerate(
          self._get_shards(input_files, shard_size)):

        time1 = time.time()
        X, valid_inds = self._featurize_shard(shard)
        if self.id_field:
          ids = shard[self.id_field].values
        else:
          ids = np.ones(len(X))
        ids = ids[valid_inds]

        if len(self.tasks) > 0:
          # Featurize task results if they exist.
          y, w = _convert_df_to_numpy(shard, self.tasks)

          if self.label_field:
            y = shard[self.label_field]
          if self.weight_field:
            w = shard[self.weight_field]

          # Filter out examples where featurization failed.
          y, w = (y[valid_inds], w[valid_inds])
          assert len(X) == len(ids) == len(y) == len(w)
        else:
          # For prospective data where results are unknown, it
          # makes no sense to have y values or weights.
          y, w = (None, None)
          assert len(X) == len(ids)

        time2 = time.time()
        logger.info("TIMING: featurizing shard %d took %0.3f s" %
                    (shard_num, time2 - time1))
        yield X, y, w, ids

    return DiskDataset.create_dataset(shard_generator(), data_dir)

  def _get_shards(self,
                  input_files: List[str],
                  shard_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
    """Defines a generator which returns data for each shard
    
    Parameters
    ----------
    input_files : List[str]
      List of json filenames.
    shard_size : int, optional
      Chunksize for reading json files.

    Yields
    ------
    df : pandas.DataFrame
      Shard of dataframe.

    Notes
    -----
    To load shards from a json file into a Pandas dataframe, the file
      must be originally saved with
    ``df.to_json('filename.json', orient='records', lines=True)``
    """
    shard_num = 1
    for filename in input_files:
      if shard_size is None:
        yield pd.read_json(filename, orient='records', lines=True)
      else:
        logger.info("About to start loading json from %s." % filename)
        for df in pd.read_json(
            filename, orient='records', chunksize=shard_size, lines=True):
          logger.info(
              "Loading shard %d of size %s." % (shard_num, str(shard_size)))
          df = df.replace(np.nan, str(""), regex=True)
          shard_num += 1
          yield df

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

    features = []
    valid_inds = []
    field = self.feature_field
    data = shard[field].tolist()

    for idx, datapoint in enumerate(data):
      feat = self.featurizer.featurize([datapoint])
      is_valid = True if feat.size > 0 else False
      valid_inds.append(is_valid)
      if is_valid:
        features.append(feat)

    return np.squeeze(np.array(features), axis=1), valid_inds
