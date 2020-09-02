import unittest
import os
import numpy as np
import tempfile

import pytest

import deepchem as dc
from deepchem.models import Smiles2Vec, ChemCeption
from deepchem.feat import create_char_to_idx, SmilesToSeq, SmilesToImage
from deepchem.molnet.load_function.chembl25_datasets import chembl25_tasks


@pytest.mark.skip(reason="Unknown")
class TestChemnetModel(unittest.TestCase):

  def setUp(self):
    self.max_seq_len = 20
    self.data_points = 10
    self.n_tasks = 5

  def get_dataset(self, mode="classification", featurizer="smiles2seq"):
    dataset_file = os.path.join(
        os.path.dirname(__file__), "chembl_25_small.csv")

    if featurizer == "smiles2seq":
      max_len = 250
      pad_len = 10
      self.char_to_idx = create_char_to_idx(
          dataset_file, max_len=max_len, smiles_field="smiles")
      featurizer = SmilesToSeq(
          char_to_idx=self.char_to_idx, max_len=max_len, pad_len=pad_len)

    elif featurizer == "smiles2img":
      img_size = 80
      img_spec = "engd"
      res = 0.5
      featurizer = SmilesToImage(img_size=img_size, img_spec=img_spec, res=res)

    loader = dc.data.CSVLoader(
        tasks=chembl25_tasks, smiles_field='smiles', featurizer=featurizer)
    dataset = loader.featurize(
        input_files=[dataset_file],
        shard_size=10000,
        data_dir=tempfile.mkdtemp())

    w = np.ones(shape=(self.data_points, self.n_tasks))

    if mode == 'classification':
      y = np.random.randint(0, 2, size=(self.data_points, self.n_tasks))
      metric = dc.metrics.Metric(
          dc.metrics.roc_auc_score, np.mean, mode="classification")
    else:
      y = np.random.normal(size=(self.data_points, self.n_tasks))
      metric = dc.metrics.Metric(
          dc.metrics.mean_absolute_error, mode="regression")

    if featurizer == "smiles2seq":
      dataset = dc.data.NumpyDataset(
          dataset.X[:self.data_points, :self.max_seq_len], y, w,
          dataset.ids[:self.data_points])
    else:
      dataset = dc.data.NumpyDataset(dataset.X[:self.data_points], y, w,
                                     dataset.ids[:self.data_points])

    return dataset, metric

  @pytest.mark.slow
  def test_smiles_to_vec_regression(self):
    dataset, metric = self.get_dataset(
        mode="regression", featurizer="smiles2seq")
    model = Smiles2Vec(
        char_to_idx=self.char_to_idx,
        max_seq_len=self.max_seq_len,
        use_conv=True,
        n_tasks=self.n_tasks,
        model_dir=None,
        mode="regression")
    model.fit(dataset, nb_epoch=500)
    scores = model.evaluate(dataset, [metric], [])
    assert all(s < 0.1 for s in scores['mean_absolute_error'])

  @pytest.mark.slow
  def test_smiles_to_vec_classification(self):
    dataset, metric = self.get_dataset(
        mode="classification", featurizer="smiles2seq")
    model = Smiles2Vec(
        char_to_idx=self.char_to_idx,
        max_seq_len=self.max_seq_len,
        use_conv=True,
        n_tasks=self.n_tasks,
        model_dir=None,
        mode="classification")
    model.fit(dataset, nb_epoch=500)
    scores = model.evaluate(dataset, [metric], [])
    assert scores['mean-roc_auc_score'] >= 0.9

  @pytest.mark.slow
  def test_chemception_regression(self):
    dataset, metric = self.get_dataset(
        mode="regression", featurizer="smiles2img")
    model = ChemCeption(
        n_tasks=self.n_tasks,
        img_spec="engd",
        model_dir=None,
        mode="regression")
    model.fit(dataset, nb_epoch=300)
    scores = model.evaluate(dataset, [metric], [])
    assert all(s < 0.1 for s in scores['mean_absolute_error'])

  @pytest.mark.slow
  def test_chemception_classification(self):
    dataset, metric = self.get_dataset(
        mode="classification", featurizer="smiles2img")
    model = ChemCeption(
        n_tasks=self.n_tasks,
        img_spec="engd",
        model_dir=None,
        mode="classification")
    model.fit(dataset, nb_epoch=300)
    scores = model.evaluate(dataset, [metric], [])
    assert scores['mean-roc_auc_score'] >= 0.9

  @pytest.mark.slow
  def test_chemception_fit_with_augmentation(self):
    dataset, metric = self.get_dataset(
        mode="classification", featurizer="smiles2img")
    model = ChemCeption(
        n_tasks=self.n_tasks,
        img_spec="engd",
        model_dir=None,
        augment=True,
        mode="classification")
    model.fit(dataset, nb_epoch=300)
    scores = model.evaluate(dataset, [metric], [])
    assert scores['mean-roc_auc_score'] >= 0.9
