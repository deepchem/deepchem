"""
Process an input dataset into a format suitable for machine learning.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import gzip
import pandas as pd
import numpy as np
import csv
import numbers
import dill
import multiprocessing as mp
from functools import partial
from rdkit import Chem
from deepchem.utils.save import log
from deepchem.utils.save import save_to_disk
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import load_pandas_from_disk
from deepchem.featurizers import Featurizer, ComplexFeaturizer
from deepchem.featurizers import UserDefinedFeaturizer
from deepchem.datasets import Dataset

def _process_field(val):
  """Parse data in a field."""
  if (isinstance(val, numbers.Number) or isinstance(val, np.ndarray)):
    return val
  elif isinstance(val, list):
    return [_process_field(elt) for elt in val]
  elif isinstance(val, str):
    try:
      return float(val)
    except ValueError:
      return val
  elif isinstance(val, Chem.Mol):
    return val
  else:
    raise ValueError("Field of unrecognized type: %s" % str(val))

def load_data(input_file):
  """Loads data from disk."""
  input_type = _get_input_type(input_file)
  if input_type == "sdf":
    raw_df = _load_sdf_file(input_file)
  else:
    raw_df = _load_csv_file(input_file)
  return raw_df

def _load_sdf_file(input_file):
  """Load SDF file into dataframe."""
  # Tasks are stored in .sdf.csv file
  raw_df = load_pandas_from_disk(input_file+".csv")
  # Structures are stored in .sdf file
  print("Reading structures from %s." % input_file)
  suppl = Chem.SDMolSupplier(str(input_file), removeHs=False)
  df_rows = []
  for ind, mol in enumerate(suppl):
    if mol is not None:
      smiles = Chem.MolToSmiles(mol)
      df_rows.append([ind,smiles,mol])
  mol_df = pd.DataFrame(df_rows, columns=('mol_id', 'smiles', 'mol'))
  raw_df = pd.concat([mol_df, raw_df], axis=1, join='inner')
  return raw_df

def _load_csv_file(input_file):
  """Loads CSV file into dataframe."""
  raw_df = load_pandas_from_disk(input_file)
  return raw_df

def _get_input_type(input_file):
  """Get type of input file. Must be csv/pkl.gz/sdf file."""
  filename, file_extension = os.path.splitext(input_file)
  # If gzipped, need to compute extension again
  if file_extension == ".gz":
    filename, file_extension = os.path.splitext(filename)
  if file_extension == ".csv":
    return "csv"
  elif file_extension == ".pkl":
    return "pandas-pickle"
  elif file_extension == ".joblib":
    return "pandas-joblib"
  elif file_extension == ".sdf":
    return "sdf"
  else:
    raise ValueError("Unrecognized extension %s" % file_extension)

def _get_fields(input_file):
  """Get the names of fields and field_types for input data."""
  # If CSV input, assume that first row contains labels
  input_type = _get_input_type(input_file)
  if input_type == "csv":
    with open(input_file, "rb") as inp_file_obj:
      return csv.reader(inp_file_obj).next()
  elif input_type == "pandas-joblib":
    df = load_from_disk(input_file)
    return df.keys()
  elif input_type == "pandas-pickle":
    df = load_pickle_from_disk(input_file)
    return df.keys()
  # If SDF input, assume that .sdf.csv file contains labels 
  elif input_type == "sdf":
    label_file = input_file + ".csv"
    print("Reading labels from %s" % label_file)
    with open(label_file, "rb") as inp_file_obj:
      return inp_file_obj.readline()
  else:
    raise ValueError("Unrecognized extension for %s" % input_file)

class DataFeaturizer(object):
  """
  Handles loading/featurizing of chemical samples (datapoints).

  Currently knows how to load csv-files/pandas-dataframes/SDF-files. Writes a
  dataframe object to disk as output.
  """

  def __init__(self, tasks, smiles_field,
               id_field=None, threshold=None,
               protein_pdb_field=None, ligand_pdb_field=None,
               ligand_mol2_field=None, mol_field=None,
               featurizers=[],
               verbosity=None, log_every_n=1000):
    """Extracts data from input as Pandas data frame"""
    if not isinstance(tasks, list):
      raise ValueError("tasks must be a list.")
    assert verbosity in [None, "low", "high"]
    self.verbosity = verbosity
    self.tasks = tasks
    self.smiles_field = smiles_field
    if id_field is None:
      self.id_field = smiles_field
    else:
      self.id_field = id_field
    self.threshold = threshold
    self.protein_pdb_field = protein_pdb_field
    self.ligand_pdb_field = ligand_pdb_field
    self.ligand_mol2_field = ligand_mol2_field
    self.mol_field = mol_field
    self.user_specified_features = None
    for featurizer in featurizers:
      if isinstance(featurizer, UserDefinedFeaturizer):
        self.user_specified_features = featurizer.feature_fields 
    self.featurizers = featurizers
    self.log_every_n = log_every_n

  def featurize(self, input_file, data_dir, shard_size=8192, worker_pool=None,
                reload=False):
    """Featurize provided file and write to specified location."""
    # If we are not to reload data, or data has not already been featurized.

    if not reload:
      log("Loading raw samples now.", self.verbosity)

      raw_df = load_data(input_file)
      fields = raw_df.keys()
      log("Loaded raw data frame from file.", self.verbosity)
      log("About to preprocess samples.", self.verbosity)

      def process_raw_sample_helper(row, fields, input_type):
        return self._process_raw_sample(input_type, row, fields)
      input_type = _get_input_type(input_file)
      process_raw_sample_helper_partial = partial(process_raw_sample_helper,
                                                  fields=fields,
                                                  input_type=input_type)


      nb_sample = raw_df.shape[0]
      interval_points = np.linspace(
          0, nb_sample, np.ceil(float(nb_sample)/shard_size)+1, dtype=int)

      metadata_rows = []
      # Construct partial function to write datasets.
      write_dataframe_partial = partial(
          Dataset.write_dataframe, data_dir=data_dir,
          featurizers=self.featurizers, tasks=self.tasks)

      for j in range(len(interval_points)-1):
        log("Sharding and standardizing into shard-%s / %s shards"
            % (str(j+1), len(interval_points)-1), self.verbosity)
        raw_df_shard = raw_df.iloc[range(interval_points[j], interval_points[j+1])]
        raw_df_shard = raw_df_shard.apply(
            process_raw_sample_helper_partial, axis=1, reduce=False)
        
        df = self._standardize_df(raw_df_shard) 
      
        field = "mol" if input_type == "sdf" else "smiles"
        for featurizer in self.featurizers:
          log("Currently featurizing feature_type: %s"
              % featurizer.__class__.__name__, self.verbosity)
          if isinstance(featurizer, UserDefinedFeaturizer):
            self._add_user_specified_features(df, featurizer)
          elif isinstance(featurizer, Featurizer):
            self._featurize_mol(df, featurizer, field=field,
                                worker_pool=worker_pool)
          elif isinstance(featurizer, ComplexFeaturizer):
            self._featurize_complexes(df, featurizer,
                                      worker_pool=worker_pool)
        basename = "shard-%d" % j
        metadata_rows.append(write_dataframe_partial((basename, df)))
    else:
      metadata_rows = None

    dataset = Dataset(data_dir=data_dir,
                      metadata_rows=metadata_rows,
                      reload=reload, verbosity=self.verbosity)

    return dataset 

  def _shard_files_exist(self, feature_dir):
    """Checks if data shard files already exist."""
    for filename in os.listdir(feature_dir):
      if "features_shard" in filename:
        return True
    return False

  def _process_raw_sample(self, input_type, row, fields):
    """Extract information from row data."""
    data = {}
    if input_type == "csv":
      for ind, field in enumerate(fields):
        data[field] = _process_field(row[ind])
    elif input_type in ["pandas-pickle", "pandas-joblib", "sdf"]:
      for field in fields:
        data[field] = _process_field(row[field])
    else:
      raise ValueError("Unrecognized input_type")
    if self.threshold is not None:
      for task in self.tasks:
        raw = _process_field(data[task])
        if not isinstance(raw, float):
          raise ValueError("Cannot threshold non-float fields.")
        data[field] = 1 if raw > self.threshold else 0
    return data

  def _standardize_df(self, ori_df):
    """Copy specified columns to new df with standard column names.

    TODO(rbharath): I think think function is now unnecessary (since the
                    dataframes are only temporary and not on disk). Should
                    be able to remove this function.
    """
    df = pd.DataFrame(ori_df[[self.id_field]])
    df.columns = ["mol_id"]
    df["smiles"] = ori_df[[self.smiles_field]]
    for task in self.tasks:
      df[task] = ori_df[[task]]
    if self.user_specified_features is not None:
      for feature in self.user_specified_features:
        df[feature] = ori_df[[feature]]
    if self.mol_field is not None:
      df["mol"] = ori_df[[self.mol_field]]
    if self.protein_pdb_field is not None:
      df["protein_pdb"] = ori_df[[self.protein_pdb_field]]
    if self.ligand_pdb_field is not None:
      df["ligand_pdb"] = ori_df[[self.ligand_pdb_field]]
    if self.ligand_mol2_field is not None:
      df["ligand_mol2"] = ori_df[[self.ligand_mol2_field]]
    return df

  def _featurize_complexes(self, df, featurizer, parallel=True,
                           worker_pool=None):
    """Generates circular fingerprints for dataset."""
    protein_pdbs = list(df["protein_pdb"])
    ligand_pdbs = list(df["ligand_pdb"])
    complexes = zip(ligand_pdbs, protein_pdbs)

    def featurize_wrapper(ligand_protein_pdb_tuple):
      ligand_pdb, protein_pdb = ligand_protein_pdb_tuple
      print("Featurizing %s" % ligand_pdb[0:2])
      molecule_features = featurizer.featurize_complexes(
          [ligand_pdb], [protein_pdb], verbosity=self.verbosity)
      return molecule_features

    if worker_pool is None:
      features = []
      for ligand_protein_pdb_tuple in zip(ligand_pdbs, protein_pdbs):
        features.append(featurize_wrapper(ligand_protein_pdb_tuple))
    else:
      features = worker_pool.map_sync(featurize_wrapper, 
                                      zip(ligand_pdbs, protein_pdbs))
      #features = featurize_wrapper(zip(ligand_pdbs, protein_pdbs))
    df[featurizer.__class__.__name__] = list(features)

  def _featurize_mol(self, df, featurizer, parallel=True, field="mol",
                     worker_pool=None):    
    """Featurize individual compounds.

       Given a featurizer that operates on individual chemical compounds 
       or macromolecules, compute & add features for that compound to the 
       features dataframe

       When featurizing a .sdf file, the 3-D structure should be preserved
       so we use the rdkit "mol" object created from .sdf instead of smiles
       string. Some featurizers such as CoulombMatrix also require a 3-D
       structure.  Featurizing from .sdf is currently the only way to
       perform CM feautization.

      TODO(rbharath): Needs to be merged with _featurize_compounds
    """
    assert field in ["mol", "smiles"]
    #sample_mols = df["mol"].tolist()
    sample_elems = df[field].tolist()

    if worker_pool is None:
      features = []
      for ind, elem in enumerate(sample_elems):
        if field == "smiles":
          mol = Chem.MolFromSmiles(elem)
        else:
          mol = elem
        if ind % self.log_every_n == 0:
          log("Featurizing sample %d" % ind, self.verbosity)
        ###################################### DEBUG
        #print("DataFeaturizer._featurize_mol")
        #print("mol, self.verbosity")
        #print(mol, self.verbosity)
        ###################################### DEBUG
        features.append(featurizer.featurize([mol], verbosity=self.verbosity))
    else:
      def featurize_wrapper(elem, dilled_featurizer):
        print("Featurizing %s" % elem)
        if field == "smiles":
          mol = Chem.MolFromSmiles(smiles)
        else:
          mol = elem
        featurizer = dill.loads(dilled_featurizer)
        feature = featurizer.featurize([mol], verbosity=self.verbosity)
        return feature

      features = worker_pool.map_sync(featurize_wrapper, 
                                      sample_elems)

    df[featurizer.__class__.__name__] = features

  def _add_user_specified_features(self, df, featurizer):
    """Merge user specified features. 

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
    log("Aggregating User-Specified Features", self.verbosity)
    features_data = []
    for ind, row in df.iterrows():
      # pandas rows are tuples (row_num, row_data)
      feature_list = []
      for feature_name in featurizer.feature_fields:
        feature_list.append(row[feature_name])
      features_data.append(np.array(feature_list))
    df[featurizer.__class__.__name__] = features_data
