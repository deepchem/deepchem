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
import shutil
from deepchem.data import DiskDataset, NumpyDataset

def undo_transforms(y, transformers):
  """Undoes all transformations applied."""
  # Note that transformers have to be undone in reversed order
  for transformer in reversed(transformers):
    if transformer.transform_y:
      y = transformer.untransform(y)
  return y

def undo_grad_transforms(grad, tasks, transformers):
  for transformer in reversed(transformers):
    if transformer.transform_y:
      grad = transformer.untransform_grad(grad, tasks)
  return grad

def get_grad_statistics(dataset):
  """Computes and returns statistics of a dataset

  This function assumes that the first task of a dataset holds the energy for
  an input system, and that the remaining tasks holds the gradient for the
  system.
  """
  if len(dataset) == 0:
    return None, None, None, None
  y = dataset.y
  energy = y[:,0]
  grad = y[:,1:]
  for i in range(energy.size):
    grad[i] *= energy[i]
  ydely_means = np.sum(grad, axis=0)/len(energy)
  return grad, ydely_means

class Transformer(object):
  """
  Abstract base class for different ML models.
  """
  # Hack to allow for easy unpickling:
  # http://stefaanlippens.net/pickleproblem
  __module__ = os.path.splitext(os.path.basename(__file__))[0]
  def __init__(self, transform_X=False, transform_y=False, transform_w=False,
               dataset=None):
    """Initializes transformation based on dataset statistics."""
    self.dataset = dataset
    self.transform_X = transform_X
    self.transform_y = transform_y
    self.transform_w = transform_w
    # One, but not both, transform_X or tranform_y is true
    assert transform_X or transform_y or transform_w
    # Use fact that bools add as ints in python
    assert (transform_X + transform_y + transform_w) == 1 

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays."""
    raise NotImplementedError(
      "Each Transformer is responsible for its own transform_array method.")

  def untransform(self, z):
    """Reverses stored transformation on provided data."""
    raise NotImplementedError(
      "Each Transformer is responsible for its own untransfomr method.")

  def transform(self, dataset, parallel=False):
    """
    Transforms all internally stored data.
    Adds X-transform, y-transform columns to metadata.
    """
    return dataset.transform(lambda X, y, w: self.transform_array(X, y, w))

  def transform_on_array(self, X, y, w):
    """
    Transforms numpy arrays X, y, and w
    """
    X, y, w = self.transform_array(X, y, w)    
    return X, y, w

class NormalizationTransformer(Transformer):

  def __init__(self, transform_X=False, transform_y=False, transform_w=False,
               dataset=None, transform_gradients=False):
    """Initialize normalization transformation."""
    super(NormalizationTransformer, self).__init__(
        transform_X=transform_X, transform_y=transform_y,
        transform_w=transform_w, dataset=dataset)
    if transform_X:
      X_means, X_stds = dataset.get_statistics(X_stats=True, y_stats=False)
      self.X_means = X_means 
      self.X_stds = X_stds
    elif transform_y:
      y_means, y_stds = dataset.get_statistics(X_stats=False, y_stats=True)
      self.y_means = y_means 
      # Control for pathological case with no variance.
      y_stds = np.array(y_stds)
      y_stds[y_stds == 0] = 1.
      self.y_stds = y_stds
    self.transform_gradients = transform_gradients
    if self.transform_gradients:
      true_grad, ydely_means = get_grad_statistics(dataset)
      self.grad = np.reshape(true_grad, (true_grad.shape[0],-1,3))
      self.ydely_means = ydely_means

  def transform(self, dataset, parallel=False):
    return super(NormalizationTransformer, self).transform(
        dataset, parallel=parallel)

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays."""
    if self.transform_X:
      X = np.nan_to_num((X - self.X_means) / self.X_stds)
    if self.transform_y:
      y = np.nan_to_num((y - self.y_means) / self.y_stds)
    return (X, y, w)

  def untransform(self, z):
    """
    Undo transformation on provided data.
    """
    if self.transform_X:
      return z * self.X_stds + self.X_means
    elif self.transform_y:
      return z * self.y_stds + self.y_means

  def untransform_grad(self, grad, tasks):
    """
    Undo transformation on gradient.
    """
    if self.transform_y:

      grad_means = self.y_means[1:]
      energy_var = self.y_stds[0]        
      grad_var = 1/energy_var*(self.ydely_means-self.y_means[0]*self.y_means[1:])
      energy = tasks[:,0]
      transformed_grad = []

      for i in range(energy.size):
        Etf = energy[i]
        grad_Etf = grad[i].flatten()
        grad_E = Etf*grad_var+energy_var*grad_Etf+grad_means
        grad_E = np.reshape(grad_E, (-1,3))
        transformed_grad.append(grad_E)   

      transformed_grad = np.asarray(transformed_grad)
      return transformed_grad

class AtomicNormalizationTransformer(Transformer):
  """
  TODO(rbharath): Needs more discussion of what a gradient is semantically.
  It's evident that not every Dataset has meaningful gradient information, so
  this transformer can't be applied to all data. Should there be a subclass of
  Dataset named GradientDataset perhaps?
  """

  def __init__(self, transform_X=False, transform_y=False, transform_w=False,
               dataset=None):
    """Initialize normalization transformation."""
    super(AtomicNormalizationTransformer, self).__init__(
        transform_X=transform_X, transform_y=transform_y,
        transform_w=transform_w, dataset=dataset)
    X_means, X_stds, y_means, y_stds = dataset.get_statistics()
    self.X_means = X_means 
    self.X_stds = X_stds
    self.y_means = y_means 
    # Control for pathological case with no variance.
    y_stds[y_stds == 0] = 1.
    self.y_stds = y_stds
    true_grad, ydely_means = get_grad_statistics(dataset)
    self.grad = np.reshape(true_grad, (true_grad.shape[0],-1,3))
    self.ydely_means = ydely_means

  def transform(self, dataset, parallel=False):
    return super(AtomicNormalizationTransformer, self).transform(
        dataset, parallel=parallel)
    

  def transform_row(self, i, df, data_dir):
    """
    Normalizes the data (X, y, w, ...) in a single row).
    """
    row = df.iloc[i]

    if self.transform_X:
      X = load_from_disk(
          os.path.join(data_dir, row['X-transformed']))
      X = np.nan_to_num((X - self.X_means) / self.X_stds)
      save_to_disk(X, os.path.join(data_dir, row['X-transformed']))

    if self.transform_y:

      y = load_from_disk(os.path.join(data_dir, row['y-transformed']))

      # transform tasks as normal
      y = np.nan_to_num((y - self.y_means) / self.y_stds)

      # add 2nd order correction term to gradients
      grad_var = 1/self.y_stds[0]*(self.ydely_means-self.y_means[0]*self.y_means[1:])
      for i in range(y.shape[0]):
        y[i,1:] = y[i,1:] - grad_var*y[i,0]/self.y_stds[0]

      save_to_disk(y, os.path.join(data_dir, row['y-transformed']))

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays."""
    if self.transform_X:
      X = np.nan_to_num((X - self.X_means) / self.X_stds)
    if self.transform_y:
      # transform tasks as normal
      y = np.nan_to_num((y - self.y_means) / self.y_stds)
      # add 2nd order correction term to gradients
      grad_var = 1/self.y_stds[0]*(self.ydely_means-self.y_means[0]*self.y_means[1:])
      for i in range(y.shape[0]):
        y[i,1:] = y[i,1:] - grad_var*y[i,0]/self.y_stds[0]
    return (X, y, w)

  def untransform(self, z):
    """
    Undo transformation on provided data.
    """
    if self.transform_X:
      return z * self.X_stds + self.X_means
    elif self.transform_y:

      # untransform grad
      grad_var = 1/self.y_stds[0]*(self.ydely_means-self.y_means[0]*self.y_means[1:])
      for i in range(z.shape[0]):
        z[i,1:] = z[i,0]*grad_var + self.y_stds[0]*z[i,1:] + self.y_means[1:] 
      # untransform energy
      z[:,0] = z[:,0] * self.y_stds[0] + self.y_means[0]

      return z

  def untransform_grad(self, grad, tasks):
    """
    Undo transformation on gradient.
    """
    if self.transform_y:

      grad_means = self.y_means[1:]
      energy_var = self.y_stds[0]        
      grad_var = 1/energy_var*(self.ydely_means-self.y_means[0]*self.y_means[1:])
      energy = tasks[:,0]
      transformed_grad = []

      for i in range(energy.size):
        Etf = energy[i]
        grad_Etf = grad[i].flatten()
        grad_E = Etf*grad_var+energy_var*grad_Etf+grad_means
        grad_E = np.reshape(grad_E, (-1,3))
        transformed_grad.append(grad_E)   

      transformed_grad = np.asarray(transformed_grad)
      return transformed_grad


class ClippingTransformer(Transformer):

  def __init__(self, transform_X=False, transform_y=False,
               transform_w=False, dataset=None, max_val=5.):
    """Initialize clipping transformation."""
    super(ClippingTransformer, self).__init__(transform_X=transform_X,
                                              transform_y=transform_y,
                                              transform_w=transform_w,
                                              dataset=dataset)
    self.max_val = max_val

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays."""
    if self.transform_X:
      X[X > self.max_val] = self.max_val
      X[X < (-1.0*self.max_val)] = -1.0 * self.max_val
    if self.transform_y:
      y[y > trunc] = trunc
      y[y < (-1.0*trunc)] = -1.0 * trunc
    return (X, y, w)

  def untransform(self, z):
    warnings.warn("Clipping cannot be undone.")
    return z

class LogTransformer(Transformer):

  def __init__(self, transform_X=False, transform_y=False,
               features=None, tasks=None,
               dataset=None):
    self.features=features
    self.tasks=tasks
    """Initialize log  transformation."""
    super(LogTransformer, self).__init__(
        transform_X=transform_X, transform_y=transform_y,
        dataset=dataset)

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays."""
    if self.transform_X:
      num_features=len(X[0])
      if self.features is None:
        X = np.log(X+1)
      else:
        for j in range(num_features):
          if j in self.features:
            X[:,j] = np.log(X[:,j]+1)
          else:
            X[:,j] = X[:,j]
    if self.transform_y:
      num_tasks=len(y[0])
      if self.tasks is None:
        y = np.log(y+1)
      else:
        for j in range(num_tasks):
          if j in self.tasks:
            y[:,j] = np.log(y[:,j]+1)
          else:
            y[:,j] = y[:,j]
    return (X, y, w)

  def untransform(self, z):
    """
    Undo transformation on provided data.
    """
    if self.transform_X:
      num_features=len(z[0])
      if self.features is None:
        return np.exp(z)-1
      else:
        for j in range(num_features):
          if j in self.features:
            z[:,j] = np.exp(z[:,j])-1
          else:
            z[:,j] = z[:,j]
        return z
    elif self.transform_y:
      num_tasks=len(z[0])
      if self.tasks is None:
        return np.exp(z)-1
      else:
        for j in range(num_tasks):
          if j in self.tasks:
            z[:,j] = np.exp(z[:,j])-1
          else:
            z[:,j] = z[:,j]
        return z

class BalancingTransformer(Transformer):
  """Balance positive and negative examples for weights."""
  def __init__(self, transform_X=False, transform_y=False,
               transform_w=False, dataset=None, seed=None):
    super(BalancingTransformer, self).__init__(
        transform_X=transform_X, transform_y=transform_y,
        transform_w=transform_w, dataset=dataset)
    # BalancingTransformer can only transform weights.
    assert not transform_X
    assert not transform_y
    assert transform_w

    # Compute weighting factors from dataset.
    y = self.dataset.y
    w = self.dataset.w
    # Ensure dataset is binary
    np.testing.assert_allclose(sorted(np.unique(y)), np.array([0., 1.]))
    weights = []
    for ind, task in enumerate(self.dataset.get_task_names()):
      task_w = w[:, ind]
      task_y = y[:, ind]
      # Remove labels with zero weights
      task_y = task_y[task_w != 0]
      num_positives = np.count_nonzero(task_y)
      num_negatives = len(task_y) - num_positives
      if num_positives > 0:
        pos_weight = float(num_negatives)/num_positives
      else:
        pos_weight = 1
      neg_weight = 1
      weights.append((neg_weight, pos_weight))
    self.weights = weights

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays."""
    w_balanced = np.zeros_like(w)
    for ind, task in enumerate(self.dataset.get_task_names()):
      task_y = y[:, ind]
      task_w = w[:, ind]
      zero_indices = np.logical_and(task_y==0, task_w != 0)
      one_indices = np.logical_and(task_y==1, task_w != 0)
      w_balanced[zero_indices, ind] = self.weights[ind][0]
      w_balanced[one_indices, ind] = self.weights[ind][1]
    return (X, y, w_balanced)

class CoulombRandomizationTransformer(Transformer):

  def __init__(self, transform_X=False, transform_y=False,
               transform_w=False, dataset=None, seed=None):
    """Iniitialize coulomb matrix randomization transformation. """
    super(CoulombRandomizationTransformer, self).__init__(
        transform_X=transform_X, transform_y=transform_y,
        transform_w=transform_w, dataset=dataset)
    self.seed = seed

  def construct_cm_from_triu(self, x):
    """
    Constructs unpadded coulomb matrix from upper triangular portion.
    """
    d = int((np.sqrt(8*len(x)+1)-1)/2)
    cm = np.zeros([d,d])
    cm[np.triu_indices_from(cm)] = x
    for i in range(len(cm)):
      for j in range(i+1,len(cm)):
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

  def transform_array(self, X, y, w):
    """
    Randomly permute a Coulomb Matrix passed as an array
    """
    if self.transform_X:
      for j in range(len(X)):
        cm = self.construct_cm_from_triu(X[j])
        X[j] = self.unpad_randomize_and_flatten(cm)

    if self.transform_y:
      print("y will not be transformed by CoulombRandomizationTransformer.")

    return X, y, w

  def untransform(self, z):
    print("Cannot undo CoulombRandomizationTransformer.")

class CoulombBinarizationTransformer(Transformer):

  def __init__(self, transform_X=False, transform_y=False,
               transform_w=False, dataset=None,
               theta=1, update_state=True):
    """Initialize binarization transformation."""
    super(CoulombBinarizationTransformer, self).__init__(
        transform_X=transform_X, transform_y=transform_y, dataset=dataset)
    self.theta = theta
    self.feature_max = np.zeros(dataset.get_data_shape())
    self.update_state = update_state

  def set_max(self, df, data_dir):
    
    for _, row in df.iterrows(): 
      X = load_from_disk(os.path.join(data_dir, row['X-transformed']))
      self.feature_max = np.maximum(self.feature_max,X.max(axis=0))

  def transform(self, dataset, parallel=False):

    dataset = super(CoulombBinarizationTransformer, self).transform(dataset,
          parallel=parallel)

    df = dataset.metadata_df
    Xt = []

    for _, row in df.iterrows():
      X_t = load_from_disk(os.path.join(dataset.data_dir, row['X-transformed']))
      Xt.append(np.array(X_t))

    X = np.vstack(Xt)
    X_means = X.mean(axis=0)
    X_stds = (X-X_means).std()

    for i, row in df.iterrows():
      X_t = (Xt[i]-X_means)/X_stds
      save_to_disk(X_t, os.path.join(dataset.data_dir, row['X-transformed']))
    return dataset

  def transform_array(self, X, y, w):
    """
    Binarizes data passed as arrays with sigmoid function
    """

    X_bin = []
    if self.update_state: 
      self.set_max(df, data_dir)
      self.update_state = False
    if self.transform_X:
      for i in range(X.shape[1]):
        for k in np.arange(0,self.feature_max[i]+self.theta,self.theta):
          X_bin += [np.tanh((X[:,i]-k)/self.theta)]

      X = np.array(X_bin).T
      X_means = X.mean(axis=0)
      X_stds = (X-X_means).std()
      X = (X-X_means)/X_stds

    if self.transform_y:
      print("y will not be transformed by CoulombBinarizationTransformer.")

    return X, y, w

  def untranform(self, z):
    print("Cannot undo CoulombBinarizationTransformer.")

class CDFTransformer(Transformer):
  """Histograms the data and assigns values based on sorted list."""
  """Acts like a Cumulative Distribution Function (CDF)."""
  def __init__(self, transform_X=False,
               transform_y=False, dataset=None,
               bins=2):
    self.transform_X = transform_X
    self.transform_y = transform_y
    self.bins = bins
    self.y = dataset.y
    #self.w = dataset.w
  # TODO (flee2): for transform_y, figure out weights

  def transform(self, dataset, bins):
    """Performs CDF transform on data."""
    X, y, w, ids = (dataset.X, dataset.y, dataset.w, dataset.ids)  
    w_t = w
    ids_t = ids
    if self.transform_X:
      X_t = get_cdf_values(X,self.bins)
      y_t = y
    if self.transform_y:
      X_t = X
      y_t = get_cdf_values(y,self.bins)
      #print("y will not be transformed by CDFTransformer, for now.")
    return NumpyDataset(X_t, y_t, w_t, ids_t)

  def untransform(self, z):
    # print("Cannot undo CDF Transformer, for now.")
    # Need this for transform_y
    if self.transform_y:
      return self.y

def get_cdf_values(array, bins):
  #array = np.transpose(array)
  n_rows = array.shape[0] 
  n_cols = array.shape[1]
  array_t = np.zeros((n_rows,n_cols))
  parts = n_rows/bins
  hist_values = np.zeros(n_rows)
  sorted_hist_values = np.zeros(n_rows)
  for row in range(n_rows):
    if np.remainder(bins,2)==1:
      hist_values[row] = np.floor(np.divide(row,parts))/(bins-1)
    else:
      hist_values[row] = np.floor(np.divide(row,parts))/bins
  for col in range(n_cols):
    order = np.argsort(array[:,col], axis=0)
    sorted_hist_values = hist_values[order]
    array_t[:,col] = sorted_hist_values
 
  return array_t

class PowerTransformer(Transformer):
  """Takes power n transforms of the data based on an input vector."""
  def __init__(self, transform_X=False,
               transform_y=False,
               powers=[1]):
    self.transform_X = transform_X
    self.transform_y = transform_y
    self.powers = powers
      
  def transform(self, dataset):
    """Performs power transform on data."""
    X, y, w, ids = (dataset.X, dataset.y, dataset.w, dataset.ids)     
    w_t = w
    ids_t = ids
    n_powers = len(self.powers)
    if self.transform_X:
      X_t = np.power(X, self.powers[0])
      for i in range(1, n_powers):
      	X_t = np.hstack((X_t,np.power(X, self.powers[i])))
      y_t = y
    if self.transform_y:
      # print("y will not be transformed by PowerTransformer, for now.")
      y_t = np.power(y, self.powers[0])
      for i in range(1, n_powers):
      	y_t = np.hstack((y_t,np.power(y, self.powers[i])))
      X_t = X
    """
    shutil.rmtree(dataset.data_dir)
    os.makedirs(dataset.data_dir)
    DiskDataset.from_numpy(dataset.data_dir, X_t, y_t, w_t, ids_t)
    return dataset
    """
    return NumpyDataset(X_t, y_t, w_t, ids_t)

  def untransform(self, z):
    # print("Cannot undo Power Transformer, for now.")    
    n_powers = len(self.powers)
    orig_len = (z.shape[1])/n_powers
    z = z[:,:orig_len]
    z = np.power(z, 1/self.powers[0])
    return z
