"""
Contains an abstract base class that supports data transformations.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import numpy as np
import warnings
from functools import partial
from deepchem.utils.save import save_to_disk
from deepchem.utils.save import load_from_disk
from deepchem.utils import pad_array

# TODO(rbharath): The handling of X/y transforms in the same class is
# awkward. Is there a better way to handle this work. 
class Transformer(object):
  """
  Abstract base class for different ML models.
  """
  # Hack to allow for easy unpickling:
  # http://stefaanlippens.net/pickleproblem
  __module__ = os.path.splitext(os.path.basename(__file__))[0]
  def __init__(self, transform_X=False, transform_y=False, dataset=None):
    """Initializes transformation based on dataset statistics."""
    self.dataset = dataset
    self.transform_X = transform_X
    self.transform_y = transform_y
    # One, but not both, transform_X or tranform_y is true
    assert transform_X or transform_y
    assert not (transform_X and transform_y)

  def transform_row(self, i, df):
    """
    Transforms the data (X, y, w, ...) in a single row).
    """
    raise NotImplementedError(
      "Each Transformer is responsible for its own tranform_row method.")

  def untransform(self, z):
    """Reverses stored transformation on provided data."""
    raise NotImplementedError(
      "Each Transformer is responsible for its own untransfomr method.")

  # TODO(rbharath): Change this function to behave like the featurization
  # (accept IPyparallel pool objects to allow for multi-node parallelism)
  def transform(self, dataset, parallel=False):
    """
    Transforms all internally stored data.

    Adds X-transform, y-transform columns to metadata.
    """
    dataset.update_moments()
    df = dataset.metadata_df
    indices = range(0, df.shape[0])
    transform_row_partial = partial(
        _transform_row, df=df, transformer=self)
    if parallel:
      pool = mp.Pool(int(mp.cpu_count()/4))
      pool.map(transform_row_partial, indices)
      pool.terminate()
    else:
      for index in indices:
        transform_row_partial(index)
    dataset.save_to_disk()

def _transform_row(i, df, transformer):
  """
  Transforms the data (X, y, w,...) in a single row.

  Writes X-transforme,d y-transformed to disk.
  """
  transformer.transform_row(i, df)

class NormalizationTransformer(Transformer):

  def __init__(self, transform_X=False, transform_y=False, dataset=None):
    """Initialize normalization transformation."""
    super(NormalizationTransformer, self).__init__(
        transform_X=transform_X, transform_y=transform_y, dataset=dataset)
    X_means, X_stds, y_means, y_stds = dataset.get_statistics()
    self.X_means = X_means 
    self.X_stds = X_stds
    self.y_means = y_means 
    self.y_stds = y_stds

  def transform(self, dataset, parallel=False):
    X_means, X_stds, y_means, y_stds = dataset.get_statistics()
    self.X_means = X_means 
    self.X_stds = X_stds
    self.y_means = y_means 
    self.y_stds = y_stds
    super(NormalizationTransformer, self).transform(
        dataset, parallel=parallel)
    

  def transform_row(self, i, df):
    """
    Normalizes the data (X, y, w, ...) in a single row).
    """
    row = df.iloc[i]

    if self.transform_X:
      X = load_from_disk(row['X-transformed'])
      X = np.nan_to_num((X - self.X_means) / self.X_stds)
      save_to_disk(X, row['X-transformed'])

    if self.transform_y:
      y = load_from_disk(row['y-transformed'])
      y = np.nan_to_num((y - self.y_means) / self.y_stds)
      save_to_disk(y, row['y-transformed'])

  def untransform(self, z):
    """
    Undo transformation on provided data.
    """
    if self.transform_X:
      return z * self.X_stds + self.X_means
    elif self.transform_y:
      out = z * self.y_stds + self.y_means
      return z * self.y_stds + self.y_means

class ClippingTransformer(Transformer):

  def __init__(self, transform_X=False, transform_y=False, dataset=None,
               max_val=5.):
    """Initialize clipping transformation."""
    super(ClippingTransformer, self).__init__(transform_X=transform_X,
                                              transform_y=transform_y,
                                              dataset=dataset)
    self.max_val = max_val

  def transform_row(self, i, df):
    """
    Clips outliers for the data (X, y, w, ...) in a single row).
    """
    row = df.iloc[i]
    if self.transform_X:
      X = load_from_disk(row['X-transformed'])
      X[X > self.max_val] = self.max_val
      X[X < (-1.0*self.max_val)] = -1.0 * self.max_val
      save_to_disk(X, row['X-transformed'])
    if self.transform_y:
      y = load_from_disk(row['y-transformed'])
      y[y > trunc] = trunc
      y[y < (-1.0*trunc)] = -1.0 * trunc
      save_to_disk(y, row['y-transformed'])

  def untransform(self, z):
    warnings.warn("Clipping cannot be undone.")
    return z

class LogTransformer(Transformer):

  def transform_row(self, i, df):
    """Logarithmically transforms data in dataset."""
    row = df.iloc[i]
    if self.transform_X:
      X = load_from_disk(row['X-transformed'])
      X = np.log(X)
      save_to_disk(X, row['X-transformed'])

    if self.transform_y:
      y = load_from_disk(row['y-transformed'])
      y = np.log(y)
      save_to_disk(y, row['y-transformed'])

  def untransform(self, z):
    """Undoes the logarithmic transformation."""
    return np.exp(z)

class CoulombRandomizationTransformer(Transformer):

  def __init__(self, transform_X=False, transform_y=False, dataset=None,
               seed=None):
    """Iniitialize coulomb matrix randomization transformation. """
    super(CoulombRandomizationTransformer, self).__init__(
        transform_X=transform_X, transform_y=transform_y, dataset=dataset)
    self.seed = seed

  def construct_cm_from_triu(self, x):
    """
    Constructs unpadded coulomb matrix from upper triangular portion.
    """
    d = int((np.sqrt(8*len(x)+1)-1)/2)
    cm = np.zeros([d,d])
    cm[np.triu_indices_from(cm)] = x
    for i in xrange(len(cm)):
      for j in xrange(i+1,len(cm)):
        cm[j,i] = cm[i,j]
    return cm

  def unpad_randomize_and_flatten(self, cm):
    """
    1. Remove zero padding on Coulomb Matrix
    2. Randomly permute the rows and columns for n_samples
    3. Flatten each sample to upper triangular portion

    Returns list of feature vectors
    """
    max_atom_number = len(cm) 
    atom_number = 0
    for i in cm[0]:
        if atom_number == max_atom_number: break
        elif i != 0.: atom_number += 1
        else: break

    upcm = cm[0:atom_number,0:atom_number]

    row_norms = np.asarray(
        [np.linalg.norm(row) for row in upcm], dtype=float)
    rng = np.random.RandomState(self.seed)
    e = rng.normal(size=row_norms.size)
    p = np.argsort(row_norms+e)
    rcm = upcm[p][:,p]
    rcm = pad_array(rcm, len(cm))
    rcm = rcm[np.triu_indices_from(rcm)]

    return rcm

  def transform_row(self, i, df):
    """
    Randomly permute a Coulomb Matrix in a dataset
    """
    row = df.iloc[i]
    if self.transform_X:
      X = load_from_disk(row['X-transformed'])
      for j in xrange(len(X)):
        cm = self.construct_cm_from_triu(X[j])
        X[j] = self.unpad_randomize_and_flatten(cm)
      save_to_disk(X, row['X-transformed'])

    if self.transform_y:
      print("y will not be transformed by "
            "CoulombRandomizationTransformer.")

  def untransform(self, z):
    print("Cannot undo CoulombRandomizationTransformer.")

class CoulombBinarizationTransformer(CoulombRandomizationTransformer):

  def __init__(self, transform_X=False, transform_y=False, dataset=None,
               theta=1):
    """Initialize binarization transformation."""
    super(CoulombBinarizationTransformer, self).__init__(
        transform_X=transform_X, transform_y=transform_y, dataset=dataset)
    self.theta = theta
    self.feature_max = np.zeros(dataset.get_data_shape()) 

  def set_max(self, df):
    
    for _, row in df.iterrows(): # Iterate over entire df by rows
      X = load_from_disk(row['X-transformed'])
      self.feature_max = np.maximum(self.feature_max,X.max(axis=0))

  def transform_row(self, i, df):
    """
    Binarizes data in dataset with sigmoid function
    """

    row = df.iloc[i]
    X_bin = []
    if i == 0: self.set_max(df)
    if self.transform_X:
      X = load_from_disk(row['X-transformed'])
      for i in range(X.shape[1]):
        for k in np.arange(0,self.feature_max[i]+self.theta,self.theta):
          X_bin += [np.tanh((X[:,i]-k)/self.theta)]

      X_bin = np.array(X_bin).T
      save_to_disk(X_bin, row['X-transformed'])

    if self.transform_y:
      print("y will not be transformed by "
            "CoulombBinarizationTransformer.")

  def untranform(self, z):
    print("Cannot undo CoulombBinarizationTransformer.")
