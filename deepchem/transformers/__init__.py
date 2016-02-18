"""
Contains an abstract base class that supports data transformations.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from deepchem.utils.save import save_to_disk

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

  def transform_row(self, i, df):
    """
    Transforms the data (X, y, w, ...) in a single row).
    """
    raise NotImplementedError(
      "Each Transformer is responsible for its own tranform_row method.")

  def untransform(self, z, X_transform=False, y_transform=True):
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
    self._transform(dataset, parallel=parallel)
    df = dataset.metadata_df
    indices = range(0, df.shape[0])
    transform_row_partial = partial(_transform_row, df=df, transformer=self)
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

class NormalizeTransformer(Transformer):

  def __init__(self, i, df, X_means, X_stds, y_means, y_stds):
    """Initialize clipping transformation."""
    super(ClippingTransformer, self).__init__(i, df)
    self.X_means = X_means 
    self.X_stds = X_stds
    self.y_means = y_means 
    self.y_stds = y_stds

  def transform_row(self, i, df):
    """
    Normalizes the data (X, y, w, ...) in a single row).
    """
    row = df.iloc[i]

    if self.transform_X:
      X = load_from_disk(row['X'])
      X = np.nan_to_num((X - self.X_means) / self.X_stds)
      save_to_disk(X, row['X-transformed'])

    if self.transform_y:
      y = load_from_disk(row['y'])
      y = np.nan_to_num((y - self.y_means) / self.y_stds)
      save_to_disk(y, row['y-transformed'])

  def untransform(z, X_transform=False, y_transform=True):
    """
    Undo transformation on provided data.
    """
    if X_transform:
      return z * self.X_stds + X_means
    elif y_transform:
      return z * self.y_stds + y_means

class ClippingTransformer(Transformer):

  def __init__(self, i, df, max_val):
    """Initialize clipping transformation."""
    super(ClippingTransformer, self).__init__(i, df)
    self.max_val = max_val

  def transform_row(self, i, df):
    """
    Clips outliers for the data (X, y, w, ...) in a single row).
    """
    row = df.iloc[i]
    if self.transform_X:
      X[X > self.max_val] = self.max_val
      X[X < (-1.0*self.max_val)] = -1.0 * self.max_val
      save_to_disk(X, row['X-transformed'])
    if self.transform_y:
      y[y > trunc] = trunc
      y[y < (-1.0*trunc)] = -1.0 * trunc
      save_to_disk(y, row['y-transformed'])

  def untransform(z, X_transform=False, y_transform=True):
    raise NotImplementedError("Clipping cannot be undone.")

class LogTransformer(Transformer):

  def transform_row(i, df):
    """Logarithmically transforms data in dataset."""
    row = df.iloc[i]
    if self.transform_X:
      X = load_from_disk(row['X'])
      X = np.log(X)
      save_to_disk(X, row['X-transformed'])

    if self.transform_y:
      y = load_from_disk(row['y'])
      y = np.log(y)
      save_to_disk(y, row['y-transformed'])

  def untransform(z, X_transform=False, y_transform=True):
    """Undoes the logarithmic transformation."""
    return np.exp(z)
