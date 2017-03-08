"""
qm9 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc


def load_qm9(featurizer=None, split='random'):
  """Load qm9 datasets."""
  # Featurize qm9 dataset
  print("About to featurize qm9 dataset.")
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"
  
  dataset_file = os.path.join(
      data_dir, "./gdb9.sdf")
  
  if not os.path.exists(dataset_file):
    os.system('wget -P ' + data_dir + 
    ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/gdb9.tar.gz ')
    os.system('tar -zxvf ' + os.path.join(data_dir, 'gdb9.tar.gz') + 
    ' -C ' + data_dir)
    

  qm9_tasks = [
      "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv",
      "u0_atom", "u298_atom", "h298_atom", "g298_atom"
  ]
  if featurizer is None:
    featurizer = dc.feat.CoulombMatrix(29)
  loader = dc.data.SDFLoader(
      tasks=qm9_tasks,
      smiles_field="smiles",
      mol_field="mol",
      featurizer=featurizer)
  dataset = loader.featurize(dataset_file)
  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'stratified': dc.splits.SingletaskStratifiedSplitter(task_number=11)
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)
  transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]
  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)
  return qm9_tasks, (train_dataset, valid_dataset, test_dataset), transformers
