"""
PDBBind dataset loader.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd
import shutil
from rdkit import Chem
from deepchem.utils.save import load_from_disk
from deepchem.datasets import Dataset
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.transformers import BalancingTransformer
from deepchem.featurizers.nnscore import NNScoreComplexFeaturizer
from deepchem.featurizers.grid_featurizer import GridFeaturizer

def load_pdbbind_labels(labels_file):
  """Loads pdbbind labels as dataframe"""
  contents = []
  with open(labels_file) as f:
    for line in f:
      if line.startswith("#"):
        continue
      else:
        contents.append(line.split())
  contents_df = pd.DataFrame(
      contents,
      columns=("PDB code", "resolution", "release year", "-logKd/Ki", "Kd/Ki",
               "ignore-this-field", "reference", "ligand name"))
  return contents_df

def compute_pdbbind_grid_feature(compound_featurizers, complex_featurizers,
                                 pdb_subdir, pdb_code):
  """Compute features for a given complex"""
  protein_file = os.path.join(pdb_subdir, "%s_protein.pdb" % pdb_code)
  ligand_file = os.path.join(pdb_subdir, "%s_ligand.sdf" % pdb_code)
  #rdkit_mol = Chem.MolFromMol2File(str(ligand_file))
  rdkit_mol = Chem.SDMolSupplier(str(ligand_file)).next()

  all_features = []
  for complex_featurizer in complex_featurizers:
    features = complex_featurizer.featurize_complexes(
      [ligand_file], [protein_file])
    ################################################ DEBUG
    all_features.append(np.squeeze(features))
    print("type(features)")
    print(type(features))
    #all_features += features
    ################################################ DEBUG
  ################################################ DEBUG
  print("grid_featurizer outcome")
  print("all_features")
  print(all_features)
  ################################################ DEBUG
  
  for compound_featurizer in compound_featurizers:
    features = np.squeeze(compound_featurizer.featurize([rdkit_mol]))
    all_features.append(features)

  ################################################ DEBUG
  print("complex_featurizers, compound_featurizers")
  print(complex_featurizers, compound_featurizers)
  print("len(all_features)")
  print(len(all_features))
  print("[features.shape for features in all_features]")
  print([features.shape for features in all_features])
  ################################################ DEBUG
  features = np.concatenate(all_features)
  return features

def compute_pdbbind_atomic_coordinates(compound_featurizers, complex_featurizers,
                                       pdb_subdir, pdb_code):
  """Compute features for a given complex"""
  protein_file = os.path.join(pdb_subdir, "%s_protein.pdb" % pdb_code)
  ligand_file = os.path.join(pdb_subdir, "%s_ligand.sdf" % pdb_code)
  #rdkit_mol = Chem.MolFromMol2File(str(ligand_file))
  rdkit_mol = Chem.SDMolSupplier(str(ligand_file)).next()

  all_features = []
  for complex_featurizer in complex_featurizers:
    features = complex_featurizer.featurize_complexes(
      [ligand_file], [protein_file])
    all_features.append(features)
  
  for compound_featurizer in compound_featurizers:
    features = np.squeeze(compound_featurizer.featurize([rdkit_mol]))
    all_features.append(features)

  features = np.concatenate(all_features)
  return features
    
def load_core_pdbbind_grid(pdbbind_dir, base_dir, reload=True):
  """Load PDBBind datasets. Does not do train/test split"""
  # Set some global variables up top
  reload = True
  verbosity = "high"
  model = "logistic"
  regen = False

  # Create some directories for analysis
  # The base_dir holds the results of all analysis
  if not reload:
    if os.path.exists(base_dir):
      shutil.rmtree(base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  current_dir = os.path.dirname(os.path.realpath(__file__))
  #Make directories to store the raw and featurized datasets.
  data_dir = os.path.join(base_dir, "dataset")

  # Load PDBBind dataset
  labels_file = os.path.join(pdbbind_dir, "INDEX_core_data.2013")
  pdb_subdirs = os.path.join(pdbbind_dir, "website-core-set")
  tasks = ["-logKd/Ki"]
  print("About to load contents.")
  contents_df = load_pdbbind_labels(labels_file)
  ids = contents_df["PDB code"].values
  y = np.array([float(val) for val in contents_df["-logKd/Ki"].values])

  # Define featurizers
  grid_featurizer = GridFeaturizer(
      voxel_width=16.0, feature_types="voxel_combined",
      #voxel_feature_types=["ecfp", "splif", "hbond", "pi_stack", "cation_pi",
      #"salt_bridge"], ecfp_power=9, splif_power=9,
      voxel_feature_types=["ecfp", "splif", "hbond", 
      "salt_bridge"], ecfp_power=9, splif_power=9,
      parallel=True, flatten=True,
      verbosity=verbosity)
  compound_featurizers = [CircularFingerprint(size=1024)]
  complex_featurizers = [grid_featurizer]
  
  # Featurize Dataset
  features = []
  feature_len = None
  y_inds = []
  for ind, pdb_code in enumerate(ids):
    print("Processing %s" % str(pdb_code))
    pdb_subdir = os.path.join(pdb_subdirs, pdb_code)
    computed_feature = compute_pdbbind_grid_feature(
        compound_featurizers, complex_featurizers, pdb_subdir, pdb_code)
    if feature_len is None:
      feature_len = len(computed_feature)
    if len(computed_feature) != feature_len:
      print("Featurization failed for %s!" % pdb_code)
      continue
    y_inds.append(ind)
    features.append(computed_feature)
  ############################################################# DEBUG
  print("[feature.shape for feature in features]")
  print([feature.shape for feature in features])
  ############################################################# DEBUG
  ############################################################# DEBUG
  y = y[y_inds]
  ############################################################# DEBUG
  X = np.vstack(features)
  w = np.ones_like(y)
   
  dataset = Dataset.from_numpy(data_dir, X, y, w, ids)
  transformers = []
  
  return tasks, dataset, transformers

def load_core_pdbbind_atomic_coordinates(pdbbind_dir, base_dir, reload=True):
  """Load PDBBind datasets. Does not do train/test split"""
  # Set some global variables up top
  reload = True
  verbosity = "high"
  model = "logistic"
  regen = False

  # Create some directories for analysis
  # The base_dir holds the results of all analysis
  if not reload:
    if os.path.exists(base_dir):
      shutil.rmtree(base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  current_dir = os.path.dirname(os.path.realpath(__file__))
  #Make directories to store the raw and featurized datasets.
  data_dir = os.path.join(base_dir, "dataset")

  # Load PDBBind dataset
  labels_file = os.path.join(pdbbind_dir, "INDEX_core_data.2013")
  pdb_subdirs = os.path.join(pdbbind_dir, "website-core-set")
  tasks = ["-logKd/Ki"]
  print("About to load contents.")
  contents_df = load_pdbbind_labels(labels_file)
  ids = contents_df["PDB code"].values
  y = np.array([float(val) for val in contents_df["-logKd/Ki"].values])

  # Define featurizers
  grid_featurizer = GridFeaturizer(
      voxel_width=16.0, feature_types="voxel_combined",
      voxel_feature_types=["ecfp", "splif", "hbond", "pi_stack", "cation_pi",
      "salt_bridge"], ecfp_power=9, splif_power=9,
      parallel=True, flatten=True, verbosity=verbosity)
  compound_featurizers = [CircularFingerprint(size=1024)]
  complex_featurizers = [grid_featurizer]
  
  # Featurize Dataset
  features = []
  for pdb_code in ids:
    print("Processing %s" % str(pdb_code))
    pdb_subdir = os.path.join(pdb_subdirs, pdb_code)
    computed_feature = compute_pdbbind_atomic_coordinates(
        compound_featurizers, complex_featurizers, pdb_subdir, pdb_code)
    if len(computed_feature) == 0:
      computed_feature = np.zeros(1024)
    features.append(computed_feature)
  X = np.vstack(features)
  w = np.ones_like(y)
   
  dataset = Dataset.from_numpy(data_dir, X, y, w, ids)
  transformers = []
  
  return tasks, dataset, transformers
