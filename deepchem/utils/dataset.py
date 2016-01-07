"""
Contains wrapper class for datasets.
"""
import os
import numpy as np
import pandas as pd
from functools import partial
import glob
from rdkit import Chem
import joblib
from vs_utils.utils import ScaffoldGenerator

def save_to_disk(dataset, filename):
  """Save a dataset to file."""
  joblib.dump(dataset, filename, compress=0)

def load_from_disk(filename):
  """Load a dataset from file."""
  dataset = joblib.load(filename)
  return dataset

def generate_scaffold(smiles, include_chirality=False, smiles_field="smiles"):
  """Compute the Bemis-Murcko scaffold for a SMILES string."""
  mol = Chem.MolFromSmiles(smiles)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return scaffold

def df_to_numpy(df, mode, feature_types):
  """Transforms a featurized dataset df into standard set of numpy arrays"""
  # perform common train/test split across all tasks
  n_samples = df.shape[0]
  sorted_tasks = get_sorted_task_names(df)
  n_tasks = len(sorted_tasks)

  y = df[sorted_tasks].values
  w = np.ones((n_samples, n_tasks))
  w[np.where(y=='')] = 0

  tensors = []
  for i, datapoint in df.iterrows():
    feature_list = []
    for feature_type in feature_types:
      feature_list.append(datapoint[feature_type])
    features = np.squeeze(np.concatenate(feature_list))
    tensors.append(features)

  x = np.stack(tensors)
  sorted_ids = df['mol_id']
  return sorted_ids, x, y, w

def get_train_test_files(paths, splittype, train_proportion=0.8):
  """
  Randomly split files into train and test.
  """
  #all_files = []
  #for path in paths:
  #  all_files += glob(os.path.join(path, "*.joblib"))
  train_indices = list(
      np.random.choice(len(all_files), int(len(all_files)*train_proportion),
                       replace=False))
  test_indices = list(set(range(len(all_files)))-set(train_indices))

  train_files = [all_files[i] for i in train_indices]
  test_files = [f for f in all_files if f not in train_files]
  return train_files, test_files


# TODO(rbharath): Should this be a method?
def get_sorted_task_names(df):
  """
  Given metadata df, return sorted names of tasks.
  """
  column_names = df.keys()
  task_names = (set(column_names) - 
                set(FeaturizedSamples.colnames))
  return sorted(list(task_names))

class FeaturizedSamples(object):
  """
  Wrapper class for featurized data on disk.
  """

  # The standard columns for featurized data.
  colnames = ["mol_id", "smiles", "split", "features", "descriptors",
              "fingerprints"]

  def __init__(self, paths=None, dataset_files=[], compound_df=None):
    if paths is not None:
      for path in paths:
        dataset_files += glob.glob(os.path.join(path, "*.joblib"))
    self.dataset_files = dataset_files
    if compound_df is None:
      compound_df = self._get_compounds()
    self.compound_df = compound_df

  def _get_compounds(self):
    """
    Create dataframe containing metadata about compounds.
    """
    compound_rows = []
    for dataset_file in self.dataset_files:
      df = load_from_disk(dataset_file)
      compound_ids = list(df["mol_id"])
      smiles = list(df["smiles"])
      splits = list(df["split"])
      compound_rows += [list(elt) for elt in zip(compound_ids, smiles, splits)]
    compound_df = pd.DataFrame(compound_rows,
                               columns=("mol_id", "smiles", "split"))
    return compound_df

  def train_test_split(self, splittype, seed=None, frac_train=.8):
    """
    Splits self into train/test sets and returns two FeaturizedDatsets
    """
    if splittype == "random":
      return self.train_test_random_split(seed=seed, frac_train=frac_train)
    elif splittype == "scaffold":
      return self.train_test_scaffold_split(frac_train=frac_train)
    elif splittype == "specified":
      return self.train_test_specified_split()
    else:
      raise ValueError("improper splittype.")

  def _train_test_from_indices(self, train_inds, test_inds):
    """
    Helper to generate train/test FeaturizedSampless.
    """
    train_dataset = FeaturizedSamples(
        dataset_files=self.dataset_files,
        compound_df=self.compound_df.iloc[train_inds])
    test_dataset = FeaturizedSamples(
        dataset_files=self.dataset_files,
        compound_df=self.compound_df.iloc[test_inds])
    return train_dataset, test_dataset

  def train_test_random_split(self, seed=None, frac_train=.8):
    """
    Splits internal compounds randomly into train/test.
    """
    np.random.seed(seed)
    train_cutoff = frac_train * len(self.compound_df)
    shuffled = np.random.permutation(range(len(self.compound_df)))
    train_inds, test_inds = shuffled[:train_cutoff], shuffled[train_cutoff:]
    return self._train_test_from_indices(train_inds, test_inds)

  def train_test_scaffold_split(self, frac_train=.8):
    """
    Splits internal compounds into train/test by scaffold.
    """
    scaffolds = {}
    for ind, row in self.compound_df.iterrows():
      scaffold = generate_scaffold(row["smiles"])
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)
    # Sort from largest to smallest scaffold sets
    scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in
                     sorted(scaffolds.items(), key=lambda x: -len(x[1]))]
    train_cutoff = frac_train * len(self.compound_df)
    train_inds, test_inds = [], []
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        test_inds += scaffold_set
      else:
        train_inds += scaffold_set
    return self._train_test_from_indices(train_inds, test_inds)

  def train_test_specified_split(self):
    """
    Splits internal compounds into train/test by user-specification.
    """
    train_inds, test_inds = [], []
    for ind, row in self.compound_df.iterrows():
      if row["split"].lower() == "train":
        train_inds.append(ind)
      elif row["split"].lower() == "test":
        test_inds.append(ind)
      else:
        raise ValueError("Missing required split information.")
    return self._train_test_from_indices(train_inds, test_inds)
    
  def to_arrays(self, out_dir, mode, feature_types):
    """
    Turns featurized dataframes into numpy files, writes them & metadata to disk.
    """
    if not isinstance(feature_types, list):
      raise ValueError("feature_types must be a list.")

    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    write_dataset_single_partial = partial(
        write_dataset_single, out_dir=out_dir, mode=mode,
        feature_types=feature_types)

    metadata_rows = []
    for df_file in self.dataset_files:
      metadata_rows.append(write_dataset_single_partial(df_file))

    metadata_df = pd.DataFrame(metadata_rows, 
                               columns=('df_file', 'task_names', 'ids', 
                                        'X', 'X-transformed', 'y', 'y-transformed', 
                                        'w',
                                        'X_sums', 'X_sum_squares', 'X_n',
                                        'y_sums', 'y_sum_squares', 'y_n')) 
    return ShardedDataset(out_dir, metadata_df)

def write_dataset_single(df_file, out_dir, mode, feature_types):
  print("Examining %s" % df_file)
  df = load_from_disk(df_file)
  task_names = get_sorted_task_names(df)
  ids, X, y, w = df_to_numpy(df, mode, feature_types)
  X_sums, X_sum_squares, X_n = compute_sums_and_nb_sample(X)
  y_sums, y_sum_squares, y_n = compute_sums_and_nb_sample(y, w)

  basename = os.path.splitext(os.path.basename(df_file))[0]
  out_X = os.path.join(out_dir, "%s-X.joblib" % basename)
  out_X_transformed = os.path.join(out_dir, "%s-X-transformed.joblib" % basename)
  out_y = os.path.join(out_dir, "%s-y.joblib" % basename)
  out_y_transformed = os.path.join(out_dir, "%s-y-transformed.joblib" % basename)
  out_w = os.path.join(out_dir, "%s-w.joblib" % basename)
  out_ids = os.path.join(out_dir, "%s-ids.joblib" % basename)

  save_to_disk(X, out_X)
  save_to_disk(y, out_y)
  save_to_disk(w, out_w)
  save_to_disk(ids, out_ids)
  return([df_file, task_names, out_ids, out_X, out_X_transformed, out_y, 
          out_y_transformed, out_w,
          X_sums, X_sum_squares, X_n, 
          y_sums, y_sum_squares, y_n])


class ShardedDataset(object):
  """
  Wrapper class for dataset transformed into X, y, w numpy ndarrays.
  """
  # TODO(rbharath): Need to document the structure of this class more carefully.
  def __init__(self, data_dir, metadata_df=None):
    """
    Initializes a numpy dataset based off metadata dataframe.
    """
    self.data_dir = data_dir
    if metadata_df is None:
      metadata_df = load_from_disk(self.get_metadata_filename())
    self.metadata_df = metadata_df
    self.save_metadata()

  def get_task_names(self):
    """
    Gets learning tasks associated with this dataset.
    """
    if not len(self.metadata_df):
      raise ValueError("No data in dataset.")
    return self.metadata_df.iterrows().next()[1]['task_names']

  def get_data_shape(self):
    """
    Gets array shape of datapoints in this dataset.
    """
    if not len(self.metadata_df):
      raise ValueError("No data in dataset.")
    sample_X = load_from_disk(self.metadata_df.iterrows().next()[1]['X'])[0]
    return np.shape(sample_X)

  def get_metadata_filename(self):
    """
    Get standard location for metadata file.
    """
    metadata_filename = os.path.join(self.data_dir, "metadata.joblib")
    return metadata_filename

  def save_metadata(self):
    """
    Save metadata file to disk.
    """
    save_to_disk(
      self.metadata_df, self.get_metadata_filename())

  def get_number_shards(self):
    """
    Returns the number of shards for this dataset.
    """
    return self.metadata_df.shape[0]

  def itershards(self):
    """
    Iterates over all shards in dataset.
    """
    nb_shards = self.get_number_shards()
    for i, row in self.metadata_df.iterrows():
      print("Loading shard %d out of %d" % (i+1, nb_shards))
      X = load_from_disk(row['X-transformed'])
      y = load_from_disk(row['y-transformed'])
      w = load_from_disk(row['w'])
      ids = load_from_disk(row['ids'])
      yield (X, y, w, ids)
      

  def transform_data(self, input_transforms, output_transforms, parallel=False):
    (normalize_X, truncate_x, normalize_y, 
        truncate_y, log_X, log_y) = False, False, False, False, False, False

    if "truncate" in input_transforms:
      truncate_x=True
    if "normalize" in input_transforms:
      normalize_X=True
    if "log" in input_transforms:
      log_X = True 

    if "normalize" in output_transforms:
      normalize_y=True
    if "log" in output_transforms:
      log_y = True

    print("Transforming data.")
    X_means, X_stds, y_means, y_stds = self.transform(normalize_X, normalize_y,
                                                      truncate_x, truncate_y,
                                                      log_X, log_y,
                                                      parallel=parallel)
    # TODO(rbharath): This modification of internal state feels ugly. Better soln?
    nrow = self.metadata_df.shape[0]
    self.metadata_df['X_means'] = [X_means for i in range(nrow)]
    self.metadata_df['X_stds'] = [X_stds for i in range(nrow)]
    self.metadata_df['y_means'] = [y_means for i in range(nrow)]
    self.metadata_df['y_stds'] = [y_stds for i in range(nrow)]
    self.save_metadata()

  def get_label_means(self):
    return self.metadata_df["y_means"]

  def get_label_stds(self):
    return self.metadata_df["y_stds"]

  def transform(self, normalize_X=True, normalize_y=True, 
                truncate_X=True, truncate_y=True,
                log_X=False, log_y=False, parallel=False):
    df = self.metadata_df
    trunc = 5.0
    X_means, X_stds, y_means, y_stds = compute_mean_and_std(df)
    total = df.shape[0]
    indices = range(0, df.shape[0])
    transform_row_partial = partial(_transform_row, df=df, normalize_X=normalize_X, 
                                    normalize_y=normalize_y, truncate_X=truncate_X, 
                                    truncate_y=truncate_y, log_X=log_X,
                                    log_y=log_y, X_means=X_means, X_stds=X_stds,
                                    y_means=y_means, y_stds=y_stds, trunc=trunc)
    if parallel:
      pool = mp.Pool(int(mp.cpu_count()/4))
      pool.map(transform_row_partial, indices)
      pool.terminate()
    else:
      for index in indices:
        transform_row_partial(index)

    return X_means, X_stds, y_means, y_stds

def _transform_row(i, df, normalize_X, normalize_y, truncate_X, truncate_y,
                      log_X, log_y, X_means, X_stds, y_means, y_stds, trunc):
  total = df.shape[0]
  row = df.iloc[i]
  X = load_from_disk(row['X'])
  if normalize_X or log_X:
    if normalize_X:
      print("Normalizing X sample %d out of %d" % (i+1,total))
      X = np.nan_to_num((X - X_means) / X_stds)
      if truncate_X:
         print("Truncating X sample %d out of %d" % (i+1,total))
         X[X > trunc] = trunc
         X[X < (-1.0*trunc)] = -1.0 * trunc
    if log_X:
      X = np.log(X)
  save_to_disk(X, row['X-transformed'])

  y = load_from_disk(row['y'])
  if normalize_y or log_y:    
    if normalize_y:
      print("Normalizing y sample %d out of %d" % (i+1,total))
      y = np.nan_to_num((y - y_means) / y_stds)
      if truncate_y:
        y[y > trunc] = trunc
        y[y < (-1.0*trunc)] = -1.0 * trunc
    if log_y:
      y = np.log(y)
  save_to_disk(y, row['y-transformed'])  

# TODO(rbharath/enf): These need to be better integrated with new OO paradigm.
def compute_sums_and_nb_sample(tensor, W=None):
  """
  Computes sums, squared sums of tensor along axis 0.

  If W is specified, only nonzero weight entries of tensor are used.
  """
  if W is None:
    sums = np.sum(tensor, axis=0)
    sum_squares = np.sum(np.square(tensor), axis=0)
    nb_sample = np.shape(tensor)[0]
  else:
    nb_task = np.shape(tensor)[1]
    sums = np.zeros((nb_task))
    sum_squares = np.zeros((nb_task))
    nb_sample = np.zeros((nb_task))
    for task in range(0, nb_task):
      y_task = tensor[:,task]
      W_task = W[:,task]
      nonzero_indices = np.nonzero(W_task)
      y_task_nonzero = y_task[nonzero_indices]
      sums[task] = np.sum(y_task_nonzero)
      sum_squares[task] = np.dot(y_task_nonzero, y_task_nonzero)
      nb_sample[task] = np.shape(y_task_nonzero)[0]
  print("compute_sums_and_nb_sample()")
  print("np.shape(tensor)")
  print(np.shape(tensor))
  return (sums, sum_squares, nb_sample)

def compute_mean_and_std(df):
  """
  Compute means/stds of X/y from sums/sum_squares of tensors.
  """
  X_sums, X_sum_squares, X_n = (df['X_sums'], 
                                df['X_sum_squares'],
                                df['X_n'])
  n = np.sum(X_n)
  overall_X_sums = np.sum(X_sums, axis=0)
  overall_X_means = overall_X_sums / n
  overall_X_sum_squares = np.sum(X_sum_squares, axis=0)

  X_vars = (overall_X_sum_squares - np.square(overall_X_sums)/n)/(n)

  y_sums, y_sum_squares, y_n = (df['y_sums'].values, 
                                df['y_sum_squares'].values,
                                df['y_n'].values)
  y_sums = np.vstack(y_sums)
  y_sum_squares = np.vstack(y_sum_squares)
  n = np.sum(y_n)
  y_means = np.sum(y_sums, axis=0)/n
  y_vars = np.sum(y_sum_squares,axis=0)/n - np.square(y_means)
  return overall_X_means, np.sqrt(X_vars), y_means, np.sqrt(y_vars)
