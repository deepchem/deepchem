"""
Process an input dataset into a format suitable for machine learning.
"""
from __future__ import division
from __future__ import unicode_literals
import os
import gzip
import pandas as pd
import numpy as np
import csv
import numbers
import tempfile
import time
import sys
from deepchem.utils.save import log
from deepchem.utils.save import load_csv_files
from deepchem.utils.save import load_sdf_files
from deepchem.utils.genomics import encode_fasta_sequence
from deepchem.feat import UserDefinedFeaturizer
from deepchem.data import DiskDataset, NumpyDataset, ImageDataset
from scipy import misc
import zipfile
from PIL import Image


def convert_df_to_numpy(df, tasks, verbose=False):
  """Transforms a dataframe containing deepchem input into numpy arrays"""
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


def featurize_smiles_df(df, featurizer, field, log_every_N=1000, verbose=True):
  """Featurize individual compounds in dataframe.

  Given a featurizer that operates on individual chemical compounds
  or macromolecules, compute & add features for that compound to the
  features dataframe
  """
  sample_elems = df[field].tolist()

  features = []
  from rdkit import Chem
  from rdkit.Chem import rdmolfiles
  from rdkit.Chem import rdmolops
  for ind, elem in enumerate(sample_elems):
    mol = Chem.MolFromSmiles(elem)
    # TODO (ytz) this is a bandage solution to reorder the atoms so
    # that they're always in the same canonical order. Presumably this
    # should be correctly implemented in the future for graph mols.
    if mol:
      new_order = rdmolfiles.CanonicalRankAtoms(mol)
      mol = rdmolops.RenumberAtoms(mol, new_order)
    if ind % log_every_N == 0:
      log("Featurizing sample %d" % ind, verbose)
    features.append(featurizer.featurize([mol]))
  valid_inds = np.array(
      [1 if elt.size > 0 else 0 for elt in features], dtype=bool)
  features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
  return np.squeeze(np.array(features), axis=1), valid_inds


def featurize_smiles_np(arr, featurizer, log_every_N=1000, verbose=True):
  """Featurize individual compounds in a numpy array.

  Given a featurizer that operates on individual chemical compounds
  or macromolecules, compute & add features for that compound to the
  features array
  """
  features = []
  from rdkit import Chem
  from rdkit.Chem import rdmolfiles
  from rdkit.Chem import rdmolops
  for ind, elem in enumerate(arr.tolist()):
    mol = Chem.MolFromSmiles(elem)
    if mol:
      new_order = rdmolfiles.CanonicalRankAtoms(mol)
      mol = rdmolops.RenumberAtoms(mol, new_order)
    if ind % log_every_N == 0:
      log("Featurizing sample %d" % ind, verbose)
    features.append(featurizer.featurize([mol]))

  valid_inds = np.array(
      [1 if elt.size > 0 else 0 for elt in features], dtype=bool)
  features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
  features = np.squeeze(np.array(features))
  return features.reshape(-1,)


def get_user_specified_features(df, featurizer, verbose=True):
  """Extract and merge user specified features.

  Merge features included in dataset provided by user
  into final features dataframe

  Three types of featurization here:

    1) Molecule featurization
      -) Smiles string featurization
      -) Rdkit MOL featurization
    2) Complex featurization
      -) PDB files for interacting molecules.
    3) User specified featurizations.

  """
  time1 = time.time()
  df[featurizer.feature_fields] = df[featurizer.feature_fields].apply(
      pd.to_numeric)
  X_shard = df.as_matrix(columns=featurizer.feature_fields)
  time2 = time.time()
  log("TIMING: user specified processing took %0.3f s" % (time2 - time1),
      verbose)
  return X_shard


def featurize_mol_df(df, featurizer, field, verbose=True, log_every_N=1000):
  """Featurize individual compounds in dataframe.

  Featurizes .sdf files, so the 3-D structure should be preserved
  so we use the rdkit "mol" object created from .sdf instead of smiles
  string. Some featurizers such as CoulombMatrix also require a 3-D
  structure.  Featurizing from .sdf is currently the only way to
  perform CM feautization.
  """
  sample_elems = df[field].tolist()

  features = []
  for ind, mol in enumerate(sample_elems):
    if ind % log_every_N == 0:
      log("Featurizing sample %d" % ind, verbose)
    features.append(featurizer.featurize([mol]))
  valid_inds = np.array(
      [1 if elt.size > 0 else 0 for elt in features], dtype=bool)
  features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
  return np.squeeze(np.array(features)), valid_inds


class DataLoader(object):
  """
  Handles loading/featurizing of chemical samples (datapoints).

  Currently knows how to load csv-files/pandas-dataframes/SDF-files. Writes a
  dataframe object to disk as output.
  """

  def __init__(self,
               tasks,
               smiles_field=None,
               id_field=None,
               mol_field=None,
               featurizer=None,
               verbose=True,
               log_every_n=1000):
    """Extracts data from input as Pandas data frame"""
    if not isinstance(tasks, list):
      raise ValueError("tasks must be a list.")
    self.verbose = verbose
    self.tasks = tasks
    self.smiles_field = smiles_field
    if id_field is None:
      self.id_field = smiles_field
    else:
      self.id_field = id_field
    self.mol_field = mol_field
    self.user_specified_features = None
    if isinstance(featurizer, UserDefinedFeaturizer):
      self.user_specified_features = featurizer.feature_fields
    self.featurizer = featurizer
    self.log_every_n = log_every_n

  def featurize(self, input_files, data_dir=None, shard_size=8192):
    """Featurize provided files and write to specified location.

    For large datasets, automatically shards into smaller chunks
    for convenience.

    Parameters
    ----------
    input_files: list
      List of input filenames.
    data_dir: str
      (Optional) Directory to store featurized dataset.
    shard_size: int
      (Optional) Number of examples stored in each shard.
    """
    log("Loading raw samples now.", self.verbose)
    log("shard_size: %d" % shard_size, self.verbose)

    if not isinstance(input_files, list):
      input_files = [input_files]

    def shard_generator():
      for shard_num, shard in enumerate(
          self.get_shards(input_files, shard_size)):
        time1 = time.time()
        X, valid_inds = self.featurize_shard(shard)
        ids = shard[self.id_field].values
        ids = ids[valid_inds]
        if len(self.tasks) > 0:
          # Featurize task results iff they exist.
          y, w = convert_df_to_numpy(shard, self.tasks, self.id_field)
          # Filter out examples where featurization failed.
          y, w = (y[valid_inds], w[valid_inds])
          assert len(X) == len(ids) == len(y) == len(w)
        else:
          # For prospective data where results are unknown, it makes
          # no sense to have y values or weights.
          y, w = (None, None)
          assert len(X) == len(ids)

        time2 = time.time()
        log(
            "TIMING: featurizing shard %d took %0.3f s" %
            (shard_num, time2 - time1), self.verbose)
        yield X, y, w, ids

    return DiskDataset.create_dataset(
        shard_generator(), data_dir, self.tasks, verbose=self.verbose)

  def get_shards(self, input_files, shard_size):
    """Stub for children classes."""
    raise NotImplementedError

  def featurize_shard(self, shard):
    """Featurizes a shard of an input dataframe."""
    raise NotImplementedError


class CSVLoader(DataLoader):
  """
  Handles loading of CSV files.
  """

  def get_shards(self, input_files, shard_size, verbose=True):
    """Defines a generator which returns data for each shard"""
    return load_csv_files(input_files, shard_size, verbose=verbose)

  def featurize_shard(self, shard):
    """Featurizes a shard of an input dataframe."""
    return featurize_smiles_df(shard, self.featurizer, field=self.smiles_field)


class UserCSVLoader(DataLoader):
  """
  Handles loading of CSV files with user-defined featurizers.
  """

  def get_shards(self, input_files, shard_size):
    """Defines a generator which returns data for each shard"""
    return load_csv_files(input_files, shard_size)

  def featurize_shard(self, shard):
    """Featurizes a shard of an input dataframe."""
    assert isinstance(self.featurizer, UserDefinedFeaturizer)
    X = get_user_specified_features(shard, self.featurizer)
    return (X, np.ones(len(X), dtype=bool))


class SDFLoader(DataLoader):
  """
  Handles loading of SDF files.
  """

  def __init__(self, tasks, clean_mols=False, **kwargs):
    super(SDFLoader, self).__init__(tasks, **kwargs)
    self.clean_mols = clean_mols
    self.tasks = tasks
    self.smiles_field = "smiles"
    self.mol_field = "mol"
    self.id_field = "smiles"

  def get_shards(self, input_files, shard_size):
    """Defines a generator which returns data for each shard"""
    return load_sdf_files(input_files, self.clean_mols, tasks=self.tasks)

  def featurize_shard(self, shard):
    """Featurizes a shard of an input dataframe."""
    log(
        "Currently featurizing feature_type: %s" %
        self.featurizer.__class__.__name__, self.verbose)
    return featurize_mol_df(shard, self.featurizer, field=self.mol_field)


class FASTALoader(DataLoader):
  """
  Handles loading of FASTA files.
  """

  def __init__(self, verbose=True):
    """Initialize loader."""
    self.verbose = verbose

  def featurize(self, input_files, data_dir=None):
    """Featurizes fasta files.

    Parameters
    ----------
    input_files: list
      List of fasta files.
    data_dir: str
      (Optional) Name of directory where featurized data is stored.
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
  """
  Handles loading of image files.

  This class allows for loading of images in various formats. For user
  convenience, also accepts zip-files and directories of images and uses some
  limited intelligence to attempt to traverse subdirectories which contain
  images.
  """

  def __init__(self, tasks=None):
    """Initialize image loader."""
    if tasks is None:
      tasks = []
    self.tasks = tasks

  def featurize(self, input_files, labels=None, weights=None, in_memory=False):
    """Featurizes image files.

    Parameters
    ----------
    input_files: list
      Each file in this list should either be of a supported image format
      (.png, .tif only for now) or of a compressed folder of image files
      (only .zip for now).
    in_memory: bool
      If true, return in-memory NumpyDataset. Else return ImageDataset.
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
    images = []
    for image_file in image_files:
      _, extension = os.path.splitext(image_file)
      extension = extension.lower()
      if extension == ".png":
        image = misc.imread(image_file)
        images.append(image)
      elif extension == ".tif":
        im = Image.open(image_file)
        imarray = np.array(im)
        images.append(imarray)
      else:
        raise ValueError("Unsupported image filetype for %s" % image_file)
    return np.array(images)
