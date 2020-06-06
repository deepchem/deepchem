"""
Process an input dataset into a format suitable for machine learning.
"""
import os
import gzip
import pandas as pd
import numpy as np
import csv
import numbers
import tempfile
import time
import sys
import logging
import warnings
from deepchem.utils.save import load_csv_files
from deepchem.utils.save import load_sdf_files
from deepchem.utils.genomics import encode_fasta_sequence
from deepchem.feat import UserDefinedFeaturizer
from deepchem.data import DiskDataset, NumpyDataset, ImageDataset
import zipfile

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


def _featurize_smiles_df(df, featurizer, field, log_every_n=1000):
  """Featurize individual compounds in dataframe.

  Private helper that given a featurizer that operates on individual
  chemical compounds or macromolecules, compute & add features for
  that compound to the features dataframe

  Parameters
  ----------
  df: pd.DataFrame
    DataFrame that holds SMILES strings
  featurizer: Featurizer
    A featurizer object
  field: str
    The name of a column in `df` that holds SMILES strings
  log_every_n: int, optional (default 1000)
    Emit a logging statement every `log_every_n` rows.
  """
  sample_elems = df[field].tolist()

  features = []
  from rdkit import Chem
  from rdkit.Chem import rdmolfiles
  from rdkit.Chem import rdmolops
  for ind, elem in enumerate(sample_elems):
    mol = Chem.MolFromSmiles(elem)
    # TODO (ytz) this is a bandage solution to reorder the atoms
    # so that they're always in the same canonical order.
    # Presumably this should be correctly implemented in the
    # future for graph mols.
    if mol:
      new_order = rdmolfiles.CanonicalRankAtoms(mol)
      mol = rdmolops.RenumberAtoms(mol, new_order)
    if ind % log_every_n == 0:
      logger.info("Featurizing sample %d" % ind)
    features.append(featurizer.featurize([mol]))
  valid_inds = np.array(
      [1 if elt.size > 0 else 0 for elt in features], dtype=bool)
  features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
  return np.squeeze(np.array(features), axis=1), valid_inds


def _get_user_specified_features(df, featurizer):
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
  df: pd.DataFrame
    DataFrame that holds SMILES strings
  featurizer: Featurizer
    A featurizer object
  """
  time1 = time.time()
  df[featurizer.feature_fields] = df[featurizer.feature_fields].apply(
      pd.to_numeric)
  X_shard = df[featurizer.feature_fields].to_numpy()
  time2 = time.time()
  logger.info(
      "TIMING: user specified processing took %0.3f s" % (time2 - time1))
  return X_shard


def _featurize_mol_df(df, featurizer, field, log_every_n=1000):
  """Featurize individual compounds in dataframe.

  Used when processing .sdf files, so the 3-D structure should be
  preserved. We use the rdkit "mol" object created from .sdf
  instead of smiles string. Some featurizers such as
  CoulombMatrix also require a 3-D structure.  Featurizing from
  .sdf is currently the only way to perform CM feautization.

  Parameters
  ----------
  df: Pandas Dataframe
    Should be created by dc.utils.save.load_sdf_files.
  featurizer: dc.feat.MolecularFeaturizer
    Featurizer for molecules.
  log_every_n: int, optional
    Controls how often logging statements are emitted.
  """
  sample_elems = df[field].tolist()

  features = []
  for ind, mol in enumerate(sample_elems):
    if ind % log_every_n == 0:
      logger.info("Featurizing sample %d" % ind)
    features.append(featurizer.featurize([mol]))
  valid_inds = np.array(
      [1 if elt.size > 0 else 0 for elt in features], dtype=bool)
  features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
  return np.squeeze(np.array(features)), valid_inds


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

  def __init__(self, tasks, id_field=None, featurizer=None, log_every_n=1000):
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

  def create_dataset(self, input_files, data_dir=None, shard_size=8192):
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


class CSVLoader(DataLoader):
  """
  Creates `Dataset` objects from input CSF files. 

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
    return load_csv_files(input_files, shard_size)

  def _featurize_shard(self, shard):
    """Featurizes a shard of an input dataframe."""
    return _featurize_smiles_df(
        shard,
        self.featurizer,
        field=self.smiles_field,
        log_every_n=self.log_every_n)


class UserCSVLoader(CSVLoader):
  """
  Handles loading of CSV files with user-defined featurizers.
  """

  def _get_shards(self, input_files, shard_size):
    """Defines a generator which returns data for each shard"""
    return load_csv_files(input_files, shard_size)

  def _featurize_shard(self, shard):
    """Featurizes a shard of an input dataframe."""
    assert isinstance(self.featurizer, UserDefinedFeaturizer)
    X = _get_user_specified_features(shard, self.featurizer)
    return (X, np.ones(len(X), dtype=bool))


class SDFLoader(DataLoader):
  """
  Creates `Dataset` from SDF input files. 

  This class provides conveniences to load data from SDF files.
  """

  def __init__(self, tasks, sanitize=False, featurizer=None, log_every_n=1000):
    """Initialize SDF Loader

    Parameters
    ----------
    tasks: list[str]
      List of tasknames. These will be loaded from the SDF file.
    sanitize: bool, optional
      Whether to sanitize molecules.
    featurizer: dc.feat.Featurizer, optional
      Featurizer to use to process data
    log_every_n: int, optional
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

  def _get_shards(self, input_files, shard_size):
    """Defines a generator which returns data for each shard"""
    return load_sdf_files(input_files, self.sanitize, tasks=self.tasks)

  def _featurize_shard(self, shard):
    """Featurizes a shard of an input dataframe."""
    logger.info("Currently featurizing feature_type: %s" %
                self.featurizer.__class__.__name__)
    return _featurize_mol_df(
        shard,
        self.featurizer,
        field=self.mol_field,
        log_every_n=self.log_every_n)


class FASTALoader(DataLoader):
  """Handles loading of FASTA files.

  FASTA files are commonly used to hold sequence data. This
  class provides convenience files to lead FASTA data and
  one-hot encode the genomic sequences for use in downstream
  learning tasks.
  """

  def __init__(self):
    """Initialize loader."""
    pass

  def create_dataset(self, input_files, data_dir=None, shard_size=None):
    """Creates a `Dataset` from input FASTA files.

    At present, FASTA support is limited and only allows for one-hot
    featurization, and doesn't allow for sharding.

    Parameters
    ----------
    input_files: list
      List of fasta files.
    data_dir: str, optional
      Name of directory where featurized data is stored.
    shard_size: int, optional
      For now, this argument is ignored and each FASTA file gets its
      own shard. 

    Returns
    -------
    A `Dataset` object containing a featurized representation of data
    from `input_files`.
    """
    if not isinstance(input_files, list):
      input_files = [input_files]

    def shard_generator():
      for input_file in input_files:
        X = encode_fasta_sequence(input_file)
        ids = np.ones(len(X))
        # (X, y, w, ids)
        yield X, None, None, ids

    return DiskDataset.create_dataset(shard_generator(), data_dir)


class ImageLoader(DataLoader):
  """Handles loading of image files.

  This class allows for loading of images in various formats.
  For user convenience, also accepts zip-files and directories
  of images and uses some limited intelligence to attempt to
  traverse subdirectories which contain images.
  """

  def __init__(self, tasks=None):
    """Initialize image loader.

    At present, custom image featurizers aren't supported by this
    loader class.

    Parameters
    ----------
    tasks: list[str]
      List of task names for image labels.
    """
    if tasks is None:
      tasks = []
    self.tasks = tasks

  def create_dataset(self,
                     input_files,
                     labels=None,
                     weights=None,
                     in_memory=False):
    """Creates and returns a `Dataset` object by featurizing provided image files and labels/weights.

    Parameters
    ----------
    input_files: list
      Each file in this list should either be of a supported
      image format (.png, .tif only for now) or of a compressed
      folder of image files (only .zip for now).
    labels: optional
      If provided, a numpy ndarray of image labels
    weights: optional
      If provided, a numpy ndarray of image weights
    in_memory: bool
      If true, return in-memory NumpyDataset. Else return ImageDataset.

    Returns
    -------
    A `Dataset` object containing a featurized representation of data
    from `input_files`, `labels`, and `weights`.
    """
    if not isinstance(input_files, list):
      input_files = [input_files]

    image_files = []
    # Sometimes zip files contain directories within. Traverse directories
    while len(input_files) > 0:
      remainder = []
      for input_file in input_files:
        filename, extension = os.path.splitext(input_file)
        extension = extension.lower()
        # TODO(rbharath): Add support for more extensions
        if os.path.isdir(input_file):
          dirfiles = [
              os.path.join(input_file, subfile)
              for subfile in os.listdir(input_file)
          ]
          remainder += dirfiles
        elif extension == ".zip":
          zip_dir = tempfile.mkdtemp()
          zip_ref = zipfile.ZipFile(input_file, 'r')
          zip_ref.extractall(path=zip_dir)
          zip_ref.close()
          zip_files = [
              os.path.join(zip_dir, name) for name in zip_ref.namelist()
          ]
          for zip_file in zip_files:
            _, extension = os.path.splitext(zip_file)
            extension = extension.lower()
            if extension in [".png", ".tif"]:
              image_files.append(zip_file)
        elif extension in [".png", ".tif"]:
          image_files.append(input_file)
        else:
          raise ValueError("Unsupported file format")
      input_files = remainder

    if in_memory:
      return NumpyDataset(
          self.load_img(image_files), y=labels, w=weights, ids=image_files)
    else:
      return ImageDataset(image_files, y=labels, w=weights, ids=image_files)

  @staticmethod
  def load_img(image_files):
    from PIL import Image
    images = []
    for image_file in image_files:
      _, extension = os.path.splitext(image_file)
      extension = extension.lower()
      if extension == ".png":
        image = np.array(Image.open(image_file))
        images.append(image)
      elif extension == ".tif":
        im = Image.open(image_file)
        imarray = np.array(im)
        images.append(imarray)
      else:
        raise ValueError("Unsupported image filetype for %s" % image_file)
    return np.array(images)
