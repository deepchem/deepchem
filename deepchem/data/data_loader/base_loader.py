import logging
import warnings
from typing import List, Optional, Tuple
from deepchem.data import DiskDataset
from deepchem.feat import UserDefinedFeaturizer, Featurizer

logger = logging.getLogger(__name__)


def _convert_df_to_numpy(df, tasks):
  """Transforms a dataframe containing deepchem input into numpy arrays

  This is a private helper method intended to help parse labels and
  weights arrays from a pandas dataframe. Here `df` is a dataframe
  which has columns for each task in `tasks`. These labels are
  extracted into a labels array `y`. Weights `w` are initialized to
  all ones, but weights for any missing labels are set to 0.

  Parameters
  ----------
  df: pd.DataFrame
    Pandas dataframe with columns for all tasks
  tasks: list
    List of tasks
  """
  n_samples = df.shape[0]
  n_tasks = len(tasks)

  time1 = time.time()
  y = np.hstack(
      [np.reshape(np.array(df[task].values), (n_samples, 1)) for task in tasks])
  time2 = time.time()

  w = np.ones((n_samples, n_tasks))
  missing = np.zeros_like(y).astype(int)
  feature_shape = None

  for ind in range(n_samples):
    for task in range(n_tasks):
      if y[ind, task] == "":
        missing[ind, task] = 1

  # ids = df[id_field].values
  # Set missing data to have weight zero
  for ind in range(n_samples):
    for task in range(n_tasks):
      if missing[ind, task]:
        y[ind, task] = 0.
        w[ind, task] = 0.

  return y.astype(float), w.astype(float)


class DataLoader(object):
  """Handles loading/featurizing of data from disk.

  The main use of `DataLoader` and its child classes is to make it
  easier to load large datasets into `Dataset` objects.`

  `DataLoader` is an abstract superclass that provides a
  general framework for loading data into DeepChem. This class should
  never be instantiated directly.  To load your own type of data, make
  a subclass of `DataLoader` and provide your own implementation for
  the `create_dataset()` method.

  To construct a `Dataset` from input data, first instantiate a
  concrete data loader (that is, an object which is an instance of a
  subclass of `DataLoader`) with a given `Featurizer` object. Then
  call the data loader's `create_dataset()` method on a list of input
  files that hold the source data to process. Note that each subclass
  of `DataLoader` is specialized to handle one type of input data so
  you will have to pick the loader class suitable for your input data
  type.
  
  Note that it isn't necessary to use a data loader to process input
  data. You can directly use `Featurizer` objects to featurize
  provided input into numpy arrays, but note that this calculation
  will be performed in memory, so you will have to write generators
  that walk the source files and write featurized data to disk
  yourself. `DataLoader` and its subclasses make this process easier
  for you by performing this work under the hood.
  """

  def __init__(self,
               tasks: List[str],
               id_field: Optional[str] = None,
               featurizer: Optional[Featurizer] = None,
               log_every_n: int = 1000):
    """Construct a DataLoader object.

    This constructor is provided as a template mainly. You
    shouldn't ever call this constructor directly as a user.

    Parameters
    ----------
    tasks: list[str]
      List of task names
    id_field: str, optional
      Name of field that holds sample identifier. Note that the
      meaning of "field" depends on the input data type and can have a
      different meaning in different subclasses. For example, a CSV
      file could have a field as a column, and an SDF file could have
      a field as molecular property.
    featurizer: dc.feat.Featurizer, optional
      Featurizer to use to process data
    log_every_n: int, optional
      Writes a logging statement this often.
    """
    if self.__class__ is DataLoader:
      raise ValueError(
          "DataLoader should never be instantiated directly. Use a subclass instead."
      )
    if not isinstance(tasks, list):
      raise ValueError("tasks must be a list.")
    self.tasks = tasks
    self.id_field = id_field
    self.user_specified_features = None
    if isinstance(featurizer, UserDefinedFeaturizer):
      self.user_specified_features = featurizer.feature_fields
    self.featurizer = featurizer
    self.log_every_n = log_every_n

  def featurize(self, input_files, data_dir=None, shard_size=8192):
    """Featurize provided files and write to specified location.

    DEPRECATED: This method is now a wrapper for `create_dataset()`
    and calls that method under the hood.

    For large datasets, automatically shards into smaller chunks
    for convenience. This implementation assumes that the helper
    methods `_get_shards` and `_featurize_shard` are implemented and
    that each shard returned by `_get_shards` is a pandas dataframe.
    You may choose to reuse or override this method in your subclass
    implementations.

    Parameters
    ----------
    input_files: list
      List of input filenames.
    data_dir: str, optional
      Directory to store featurized dataset.
    shard_size: int, optional
      Number of examples stored in each shard.

    Returns
    -------
    A `Dataset` object containing a featurized representation of data
    from `input_files`.
    """
    warnings.warn(
        "featurize() is deprecated and has been renamed to create_dataset(). featurize() will be removed in DeepChem 3.0",
        FutureWarning)
    return self.create_dataset(input_files, data_dir, shard_size)

  def create_dataset(self,
                     input_files: List[str],
                     data_dir: Optional[str] = None,
                     shard_size: Optional[int] = 8192) -> DiskDataset:
    """Creates and returns a `Dataset` object by featurizing provided files.

    Reads in `input_files` and uses `self.featurizer` to featurize the
    data in these input files.  For large files, automatically shards
    into smaller chunks of `shard_size` datapoints for convenience.
    Returns a `Dataset` object that contains the featurized dataset.

    This implementation assumes that the helper methods `_get_shards`
    and `_featurize_shard` are implemented and that each shard
    returned by `_get_shards` is a pandas dataframe.  You may choose
    to reuse or override this method in your subclass implementations.

    Parameters
    ----------
    input_files: list
      List of input filenames.
    data_dir: str, optional
      Directory to store featurized dataset.
    shard_size: int, optional
      Number of examples stored in each shard.

    Returns
    -------
    A `Dataset` object containing a featurized representation of data
    from `input_files`.
    """
    logger.info("Loading raw samples now.")
    logger.info("shard_size: %d" % shard_size)

    if not isinstance(input_files, list):
      input_files = [input_files]

    def shard_generator():
      for shard_num, shard in enumerate(
          self._get_shards(input_files, shard_size)):
        time1 = time.time()
        X, valid_inds = self._featurize_shard(shard)
        ids = shard[self.id_field].values
        ids = ids[valid_inds]
        if len(self.tasks) > 0:
          # Featurize task results iff they exist.
          y, w = _convert_df_to_numpy(shard, self.tasks)
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

    return DiskDataset.create_dataset(shard_generator(), data_dir, self.tasks)

  def _get_shards(self, input_files, shard_size):
    """Stub for children classes.

    Should implement a generator that walks over the source data in
    `input_files` and returns a "shard" at a time. Here a shard is a
    chunk of input data that can reasonably be handled in memory. For
    example, this may be a set of rows from a CSV file or a set of
    molecules from a SDF file. To re-use the
    `DataLoader.create_dataset()` method, each shard must be a pandas
    dataframe.

    If you chose to override `create_dataset()` directly you don't
    need to override this helper method.
    
    Parameters
    ----------
    input_files: list
      List of input filenames.
    shard_size: int, optional
      Number of examples stored in each shard.
    """
    raise NotImplementedError

  def _featurize_shard(self, shard):
    """Featurizes a shard of input data.

    Recall a shard is a chunk of input data that can reasonably be
    handled in memory. For example, this may be a set of rows from a
    CSV file or a set of molecules from a SDF file. Featurize this
    shard in memory and return the results.
    """
    raise NotImplementedError
