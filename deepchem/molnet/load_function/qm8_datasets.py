"""
qm8 dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem


def load_qm8(featurizer='CoulombMatrix',
             split='random',
             reload=True,
             move_mean=True):
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    if move_mean:
      dir_name = "qm8/" + featurizer + "/" + str(split)
    else:
      dir_name = "qm8/" + featurizer + "_mean_unmoved/" + str(split)
    save_dir = os.path.join(data_dir, dir_name)

  if featurizer in ['CoulombMatrix', 'BPSymmetryFunctionInput', 'MP', 'Raw']:
    dataset_file = os.path.join(data_dir, "qm8.sdf")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb8.tar.gz'
      )
      deepchem.utils.untargz_file(
          os.path.join(data_dir, 'gdb8.tar.gz'), data_dir)
  else:
    dataset_file = os.path.join(data_dir, "qm8.csv")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm8.csv'
      )

  qm8_tasks = [
      "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0",
      "f2-PBE0", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM",
      "f1-CAM", "f2-CAM"
  ]

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return qm8_tasks, all_dataset, transformers

  if featurizer in ['CoulombMatrix', 'BPSymmetryFunctionInput', 'MP', 'Raw']:
    if featurizer == 'CoulombMatrix':
      featurizer = deepchem.feat.CoulombMatrix(26)
    elif featurizer == 'BPSymmetryFunctionInput':
      featurizer = deepchem.feat.BPSymmetryFunctionInput(26)
    elif featurizer == 'Raw':
      featurizer = deepchem.feat.RawFeaturizer()
    elif featurizer == 'MP':
      featurizer = deepchem.feat.WeaveFeaturizer(
          graph_distance=False, explicit_H=True)
    loader = deepchem.data.SDFLoader(
        tasks=qm8_tasks,
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
    loader = deepchem.data.CSVLoader(
        tasks=qm8_tasks, smiles_field="smiles", featurizer=featurizer)

  dataset = loader.featurize(dataset_file)

  if split == None:
    raise ValueError()

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter(task_number=0)
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)
  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset, move_mean=move_mean)
  ]
  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)
  if reload:
    deepchem.utils.save.save_dataset_to_disk(
        save_dir, train_dataset, valid_dataset, test_dataset, transformers)
  return qm8_tasks, (train_dataset, valid_dataset, test_dataset), transformers
