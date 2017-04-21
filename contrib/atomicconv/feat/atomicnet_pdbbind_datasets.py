"""
PDBBind dataset loader.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd
from atomicnet_coordinates import ComplexNeighborListFragmentAtomicCoordinates


def load_pdbbind_labels(labels_file):
  """Loads pdbbind labels as dataframe

  Parameters
  ----------
  labels_file: str
    Location of PDBbind datafile.

  Returns
  -------
  contents_df: pd.DataFrame
    Dataframe containing contents of PDBbind datafile.

  """

  contents = []
  with open(labels_file) as f:
    for line in f:
      if line.startswith("#"):
        continue
      else:
        splitline = line.split()
        if len(splitline) == 8:
          contents.append(splitline)
        else:
          print("Incorrect data format")
          print(splitline)

  contents_df = pd.DataFrame(
      contents,
      columns=("PDB code", "resolution", "release year", "-logKd/Ki", "Kd/Ki",
               "ignore-this-field", "reference", "ligand name"))
  return contents_df


def compute_pdbbind_coordinate_features(complex_featurizer, pdb_subdir,
                                        pdb_code):
  """Compute features for a given complex

  Parameters
  ----------
  complex_featurizer: dc.feat.ComplexFeaturizer
    Complex featurizer.
  pdb_subdir: str
    Location of complex PDB files.
  pdb_core: str
    Complex PDB code.

  Returns
  -------
  feature: Tuple
    Complex features.

  """

  protein_file = os.path.join(pdb_subdir, "%s_pocket.pdb" % pdb_code)
  ligand_file = os.path.join(pdb_subdir, "%s_ligand.pdb" % pdb_code)
  feature = complex_featurizer._featurize_complex(
      str(ligand_file), str(protein_file))
  return feature


def load_pdbbind_fragment_coordinates(frag1_num_atoms,
                                      frag2_num_atoms,
                                      complex_num_atoms,
                                      max_num_neighbors,
                                      neighbor_cutoff,
                                      pdbbind_dir,
                                      base_dir,
                                      datafile="INDEX_core_data.2013"):
  """Featurize PDBBind dataset.

  Parameters
  ----------
  frag1_num_atoms: int
    Maximum number of atoms in fragment 1.
  frag2_num_atoms: int
    Maximum number of atoms in fragment 2.
  complex_num_atoms: int
    Maximum number of atoms in complex.
  max_num_neighbors: int
    Maximum number of neighbors per atom.
  neighbor_cutoff: float
    Interaction cutoff [Angstrom].
  pdbbind_dir: str
    Location of PDBbind datafile.
  base_dir: str
    Location for storing featurized dataset.
  datafile: str
    Name of PDBbind datafile, optional (Default "INDEX_core_data.2013").

  Returns
  -------
  tasks: list
    PDBbind tasks.
  dataset: dc.data.DiskDataset
    PDBbind featurized dataset.
  transformers: list
    dc.trans.Transformer objects.

  """

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
  labels_file = os.path.join(pdbbind_dir, datafile)
  tasks = ["-logKd/Ki"]
  print("About to load contents.")
  contents_df = load_pdbbind_labels(labels_file)
  ids = contents_df["PDB code"].values
  y = np.array([float(val) for val in contents_df["-logKd/Ki"].values])

  # Define featurizers
  featurizer = ComplexNeighborListFragmentAtomicCoordinates(
      frag1_num_atoms, frag2_num_atoms, complex_num_atoms, max_num_neighbors,
      neighbor_cutoff)

  w = np.ones_like(y)

  #Currently featurizes with shard_size=1
  #Dataset can be reshard: dataset = dataset.reshard(48) for example
  def shard_generator():
    for ind, pdb_code in enumerate(ids):
      print("Processing %s" % str(pdb_code))
      pdb_subdir = os.path.join(pdbbind_dir, pdb_code)
      computed_feature = compute_pdbbind_coordinate_features(
          featurizer, pdb_subdir, pdb_code)
      if computed_feature[0] is None:
        print("Bad featurization")
        continue
      else:
        X_b = np.reshape(np.array(computed_feature), (1, 9))
        y_b = y[ind]
        w_b = w[ind]
        y_b = np.reshape(y_b, (1, -1))
        w_b = np.reshape(w_b, (1, -1))
        yield (X_b, y_b, w_b, pdb_code)

  dataset = dc.data.DiskDataset.create_dataset(
      shard_generator(), data_dir=data_dir, tasks=tasks)
  transformers = []

  return tasks, dataset, transformers
