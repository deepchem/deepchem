"""
qm9 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem


def load_qm9(featurizer='CoulombMatrix', split='random', reload=True):
  """Load qm9 datasets."""
  # Featurize qm9 dataset
  print("About to featurize qm9 dataset.")
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"
  if reload:
    save_dir = os.path.join(data_dir, "qm9/" + featurizer + "/" + split)

  if featurizer == 'CoulombMatrix':
    dataset_file = os.path.join(data_dir, "gdb9.sdf")

    if not os.path.exists(dataset_file):
      os.system(
          'wget -P ' + data_dir +
          ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz '
      )
      os.system('tar -zxvf ' + os.path.join(data_dir, 'gdb9.tar.gz') + ' -C ' +
                data_dir)
  else:
    dataset_file = os.path.join(data_dir, "qm9.csv")
    if not os.path.exists(dataset_file):
      os.system(
          'wget -P ' + data_dir +
          ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm9.csv '
      )

  qm9_tasks = [
      "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv",
      "u0_atom", "u298_atom", "h298_atom", "g298_atom"
  ]

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return qm9_tasks, all_dataset, transformers

  if featurizer == 'CoulombMatrix':
    featurizer = deepchem.feat.CoulombMatrix(29)
    loader = deepchem.data.SDFLoader(
        tasks=qm9_tasks,
        smiles_field="smiles",
        mol_field="mol",
        featurizer=featurizer)
  else:
    if featurizer == 'ECFP':
      featurizer = deepchem.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
      featurizer = deepchem.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
      featurizer = deepchem.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
      featurizer = deepchem.feat.RawFeaturizer()
    loader = deepchem.data.CSVLoader(
        tasks=qm9_tasks, smiles_field="smiles", featurizer=featurizer)

  dataset = loader.featurize(dataset_file)
  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter(task_number=11)
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)
  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]
  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(
        save_dir, train_dataset, valid_dataset, test_dataset, transformers)
  return qm9_tasks, (train_dataset, valid_dataset, test_dataset), transformers
