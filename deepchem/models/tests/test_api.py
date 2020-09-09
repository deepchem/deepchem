"""
Integration tests for singletask vector feature models.
"""
import os
import deepchem as dc
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def test_singletask_sklearn_rf_ECFP_regression_API():
  """Test of singletask RF ECFP regression API."""
  X = np.random.rand(100, 5)
  y = np.random.rand(100,)
  dataset = dc.data.NumpyDataset(X, y)

  splitter = dc.splits.RandomSplitter()
  train_dataset, test_dataset = splitter.train_test_split(dataset)

  transformer = dc.trans.NormalizationTransformer(
      transform_y=True, dataset=train_dataset)
  train_dataset = transformer.transform(train_dataset)
  test_dataset = transformer.transform(test_dataset)

  regression_metrics = [
      dc.metrics.Metric(dc.metrics.r2_score),
      dc.metrics.Metric(dc.metrics.mean_squared_error),
      dc.metrics.Metric(dc.metrics.mean_absolute_error)
  ]

  sklearn_model = RandomForestRegressor()
  model = dc.models.SklearnModel(sklearn_model)

  # Fit trained model
  model.fit(train_dataset)
  model.save()

  # Eval model on train
  _ = model.evaluate(train_dataset, regression_metrics, [transformer])
  _ = model.evaluate(test_dataset, regression_metrics, [transformer])


def test_singletask_sklearn_rf_user_specified_regression_API():
  """Test of singletask RF USF regression API."""
  featurizer = dc.feat.UserDefinedFeaturizer(
      ["user-specified1", "user-specified2"])
  tasks = ["log-solubility"]
  current_dir = os.path.dirname(os.path.abspath(__file__))
  input_file = os.path.join(current_dir, "user_specified_example.csv")
  loader = dc.data.UserCSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(input_file)

  splitter = dc.splits.RandomSplitter()
  train_dataset, test_dataset = splitter.train_test_split(dataset)

  transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]
  for dataset in [train_dataset, test_dataset]:
    for transformer in transformers:
      dataset = transformer.transform(dataset)

  regression_metrics = [
      dc.metrics.Metric(dc.metrics.r2_score),
      dc.metrics.Metric(dc.metrics.mean_squared_error),
      dc.metrics.Metric(dc.metrics.mean_absolute_error)
  ]

  sklearn_model = RandomForestRegressor()
  model = dc.models.SklearnModel(sklearn_model)

  # Fit trained model
  model.fit(train_dataset)
  model.save()

  # Eval model on train/test
  _ = model.evaluate(train_dataset, regression_metrics, transformers)
  _ = model.evaluate(test_dataset, regression_metrics, transformers)


def test_singletask_sklearn_rf_RDKIT_descriptor_regression_API():
  """Test of singletask RF RDKIT-descriptor regression API."""
  splittype = "scaffold"
  featurizer = dc.feat.RDKitDescriptors()
  tasks = ["log-solubility"]

  current_dir = os.path.dirname(os.path.abspath(__file__))
  input_file = os.path.join(current_dir, "example.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(input_file)

  splitter = dc.splits.ScaffoldSplitter()
  train_dataset, test_dataset = splitter.train_test_split(dataset)

  transformers = [
      dc.trans.NormalizationTransformer(
          transform_X=True, dataset=train_dataset),
      dc.trans.ClippingTransformer(transform_X=True, dataset=train_dataset),
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]
  for dataset in [train_dataset, test_dataset]:
    for transformer in transformers:
      dataset = transformer.transform(dataset)

  regression_metrics = [
      dc.metrics.Metric(dc.metrics.r2_score),
      dc.metrics.Metric(dc.metrics.mean_squared_error),
      dc.metrics.Metric(dc.metrics.mean_absolute_error)
  ]

  sklearn_model = RandomForestRegressor()
  model = dc.models.SklearnModel(sklearn_model)

  # Fit trained model
  model.fit(train_dataset)
  model.save()

  # Eval model on train/test
  _ = model.evaluate(train_dataset, regression_metrics, transformers)
  _ = model.evaluate(test_dataset, regression_metrics, transformers)


def test_singletask_mlp_ECFP_classification_API():
  """Test of singletask MLP classification API."""
  np.random.seed(123)

  X = np.random.rand(100, 5)
  y = np.random.randint(2, size=(100,))
  dataset = dc.data.NumpyDataset(X, y)

  splitter = dc.splits.RandomSplitter()
  train_dataset, test_dataset = splitter.train_test_split(dataset)

  transformers = []

  classification_metrics = [
      dc.metrics.Metric(dc.metrics.roc_auc_score),
      dc.metrics.Metric(dc.metrics.prc_auc_score),
      dc.metrics.Metric(dc.metrics.matthews_corrcoef),
      dc.metrics.Metric(dc.metrics.recall_score),
      dc.metrics.Metric(dc.metrics.accuracy_score),
      dc.metrics.Metric(dc.metrics.balanced_accuracy_score),
      dc.metrics.Metric(dc.metrics.jaccard_score),
      dc.metrics.Metric(dc.metrics.f1_score),
      dc.metrics.Metric(dc.metrics.pixel_error),
      dc.metrics.Metric(dc.metrics.kappa_score),
      dc.metrics.Metric(dc.metrics.bedroc_score),
  ]

  model = dc.models.MultitaskClassifier(1, 5)

  # Fit trained model
  model.fit(train_dataset)

  # Eval model on train/test
  _ = model.evaluate(train_dataset, classification_metrics, transformers)
  _ = model.evaluate(test_dataset, classification_metrics, transformers)
