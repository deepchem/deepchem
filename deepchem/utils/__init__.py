"""
Miscellaneous utility functions.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "BSD 3-clause"

import gzip
import numpy as np
import os
import pandas as pd
import sys
import tempfile
import tarfile
import zipfile

from urllib.request import urlretrieve


def pad_array(x, shape, fill=0, both=False):
  """
  Pad an array with a fill value.

  Parameters
  ----------
  x : ndarray
      Matrix.
  shape : tuple or int
      Desired shape. If int, all dimensions are padded to that size.
  fill : object, optional (default 0)
      Fill value.
  both : bool, optional (default False)
      If True, split the padding on both sides of each axis. If False,
      padding is applied to the end of each axis.
  """
  x = np.asarray(x)
  if not isinstance(shape, tuple):
    shape = tuple(shape for _ in range(x.ndim))
  pad = []
  for i in range(x.ndim):
    diff = shape[i] - x.shape[i]
    assert diff >= 0
    if both:
      a, b = divmod(diff, 2)
      b += a
      pad.append((a, b))
    else:
      pad.append((0, diff))
  pad = tuple(pad)
  x = np.pad(x, pad, mode='constant', constant_values=fill)
  return x


def get_data_dir():
  """Get the DeepChem data directory."""
  if 'DEEPCHEM_DATA_DIR' in os.environ:
    return os.environ['DEEPCHEM_DATA_DIR']
  return tempfile.gettempdir()


# The number of elements to print for dataset ids/tasks
_print_threshold = 10


def get_print_threshold():
  """Return the printing threshold for datasets.

  The print threshold is the number of elements from ids/tasks to
  print when printing representations of `Dataset` objects.

  Returns
  ----------
  threshold: int
    Number of elements that will be printed
  """
  return _print_threshold


def set_print_threshold(threshold):
  """Set print threshold

  The print threshold is the number of elements from ids/tasks to
  print when printing representations of `Dataset` objects.

  Parameters
  ----------
  threshold: int
    Number of elements to print.
  """
  global _print_threshold
  _print_threshold = threshold


# If a dataset contains more than this number of elements, it won't
# print any dataset ids
_max_print_size = 1000


def get_max_print_size():
  """Return the max print size for a datset.

  If a dataset is large, printing `self.ids` as part of a string
  representation can be very slow. This field controls the maximum
  size for a dataset before ids are no longer printed.

  Returns
  -------
  max_print_size: int
    Maximum length of a dataset for ids to be printed in string
    representation.
  """
  return _max_print_size


def set_max_print_size(max_print_size):
  """Set max_print_size

  If a dataset is large, printing `self.ids` as part of a string
  representation can be very slow. This field controls the maximum
  size for a dataset before ids are no longer printed.

  Parameters
  ----------
  max_print_size: int
    Maximum length of a dataset for ids to be printed in string
    representation.
  """
  global _max_print_size
  _max_print_size = max_print_size


def download_url(url, dest_dir=get_data_dir(), name=None):
  """Download a file to disk.

  Parameters
  ----------
  url: str
    the URL to download from
  dest_dir: str
    the directory to save the file in
  name: str
    the file name to save it as.  If omitted, it will try to extract a file name from the URL
  """
  if name is None:
    name = url
    if '?' in name:
      name = name[:name.find('?')]
    if '/' in name:
      name = name[name.rfind('/') + 1:]
  urlretrieve(url, os.path.join(dest_dir, name))


def untargz_file(file, dest_dir=get_data_dir(), name=None):
  """Untar and unzip a .tar.gz file to disk.

  Parameters
  ----------
  file: str
    the filepath to decompress
  dest_dir: str
    the directory to save the file in
  name: str
    the file name to save it as.  If omitted, it will use the file name
  """
  if name is None:
    name = file
  tar = tarfile.open(name)
  tar.extractall(path=dest_dir)
  tar.close()


def unzip_file(file, dest_dir=None, name=None):
  """Unzip a .zip file to disk.

  Parameters
  ----------
  file: str
    the filepath to decompress
  dest_dir: str
    the directory to save the file in
  name: str
    the directory name to unzip it to.  If omitted, it will use the file
    name
  """
  if name is None:
    name = file
  if dest_dir is None:
    dest_dir = os.path.join(get_data_dir, name)
  with zipfile.ZipFile(file, "r") as zip_ref:
    zip_ref.extractall(dest_dir)


class ScaffoldGenerator(object):
  """
  Generate molecular scaffolds.

  Parameters
  ----------
  include_chirality : : bool, optional (default False)
      Include chirality in scaffolds.
  """

  def __init__(self, include_chirality=False):
    self.include_chirality = include_chirality

  def get_scaffold(self, mol):
    """
    Get Murcko scaffolds for molecules.

    Murcko scaffolds are described in DOI: 10.1021/jm9602928.
    They are essentially that part of the molecule consisting of
    rings and the linker atoms between them.

    Parameters
    ----------
    mols : array_like
        Molecules.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=self.include_chirality)
