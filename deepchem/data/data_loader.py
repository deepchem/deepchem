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
import tempfile
from rdkit import Chem
import time
import sys
from deepchem.utils.save import log
from deepchem.utils.save import save_to_disk
from deepchem.utils.save import load_pickle_from_disk
from deepchem.feat import Featurizer, ComplexFeaturizer
from deepchem.feat import UserDefinedFeaturizer
from deepchem.data import DiskDataset
from deepchem.utils.save import load_data
from deepchem.utils.save import get_input_type

class DataLoader(object):
  """
  Handles loading/featurizing of chemical samples (datapoints).

  Currently knows how to load csv-files/pandas-dataframes/SDF-files. Writes a
  dataframe object to disk as output.
  """

  def __init__(self, tasks, smiles_field=None,
               id_field=None, mol_field=None, featurizer=None,
               verbose=True, log_every_n=1000):
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
    """Featurize provided files and write to specified location."""
    log("Loading raw samples now.", self.verbose)
    log("shard_size: %d" % shard_size, self.verbose)

    # Allow users to specify a single file for featurization
    if not isinstance(input_files, list):
      input_files = [input_files]

    if data_dir is not None:
      if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    else:
      data_dir = tempfile.mkdtemp()

    if not len(input_files):
      return None
    input_type = get_input_type(input_files[0])

    metadata_rows = []
    for shard_num, elt in enumerate(load_data(input_files, shard_size)):
      time1 = time.time()
      metadata_row = self._featurize_shard(elt, data_dir, shard_num, input_type)
      if metadata_row is not None:
        metadata_rows.append(metadata_row)
      time2 = time.time()
      log("TIMING: shard %d took %0.3f s" % (shard_num, time2-time1),
          self.verbose)
    time1 = time.time()
    dataset = DiskDataset(data_dir=data_dir, metadata_rows=metadata_rows,
                          reload=True)
    time2 = time.time()
    print("TIMING: dataset construction took %0.3f s" % (time2-time1),
          self.verbose)
    return dataset 

  def _featurize_shard(self, df_shard, data_dir, shard_num, input_type):
    """Featurizes a shard of an input dataframe."""
    field = self.mol_field if input_type == "sdf" else self.smiles_field 
    field_type = "mol" if input_type == "sdf" else "smiles" 
    log("Currently featurizing feature_type: %s"
        % self.featurizer.__class__.__name__, self.verbose)
    if isinstance(self.featurizer, UserDefinedFeaturizer):
      self._add_user_specified_features(df_shard, self.featurizer)
    elif isinstance(self.featurizer, Featurizer):
      self._featurize_mol(df_shard, self.featurizer, field=field,
                          field_type=field_type)
    basename = "shard-%d" % shard_num 
    time1 = time.time()
    metadata_row = DiskDataset.write_dataframe(
        (basename, df_shard), data_dir=data_dir,
        featurizer=self.featurizer, tasks=self.tasks,
        mol_id_field=self.id_field)
    time2 = time.time()
    log("TIMING: writing metadata row took %0.3f s" % (time2-time1),
        self.verbose)
    return metadata_row

  def _featurize_mol(self, df, featurizer, parallel=True, field_type="mol",
                     field=None):    
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
    assert field_type in ["mol", "smiles"]
    assert field is not None
    sample_elems = df[field].tolist()

    features = []
    for ind, elem in enumerate(sample_elems):
      if field_type == "smiles":
        mol = Chem.MolFromSmiles(elem)
      else:
        mol = elem
      if ind % self.log_every_n == 0:
        log("Featurizing sample %d" % ind, self.verbose)
      features.append(featurizer.featurize([mol]))

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
    time1 = time.time()
    df[featurizer.feature_fields] = df[featurizer.feature_fields].apply(pd.to_numeric)
    X_shard = df.as_matrix(columns=featurizer.feature_fields)
    df[featurizer.__class__.__name__] = [np.array(elt) for elt in X_shard.tolist()]
    time2 = time.time()
    log("TIMING: user specified processing took %0.3f s" % (time2-time1),
        self.verbose)
