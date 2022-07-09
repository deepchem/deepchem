import pytest
try:
  import torch
  import deepchem as dc
  from deepchem.models import CNNModule, CNN
  import numpy as np
  has_pytorch = True
except:
  has_pytorch = False


@pytest.mark.torch
def test_cnn1d_torch():
  torch.manual_seed(0)

  n_tasks = 5
  n_features = 8
  n_classes = 7
  batch_size = 2
  mode = 'classification'
  model = CNNModule(n_tasks=n_tasks,
                    n_features=n_features,
                    dims=1,
                    dropouts=[0.5, 0.2, 0.4, 0.7],
                    layer_filters=[3, 8, 8, 16],
                    kernel_size=3,
                    n_classes=n_classes,
                    mode=mode,
                    uncertainty=False)

  x = torch.ones(batch_size, 8, 196)
  y = model(x)

  # output_type for classification = [output, logits]
  assert len(y) == 2
  assert y[0].shape == (batch_size, n_tasks, n_classes)
  assert y[0].shape == y[1].shape


@pytest.mark.torch
def test_cnn2d_torch():
  torch.manual_seed(0)

  n_tasks = 5
  n_features = 8
  n_classes = 7
  batch_size = 2
  mode = 'classification'
  model = CNNModule(n_tasks=n_tasks,
                    n_features=n_features,
                    dims=2,
                    dropouts=[0.5, 0.2, 0.4, 0.7],
                    layer_filters=[3, 8, 8, 16],
                    kernel_size=3,
                    n_classes=n_classes,
                    mode=mode,
                    uncertainty=False)

  x = torch.ones(batch_size, 8, 196, 196)
  y = model(x)

  # output_type for classification = [output, logits]
  assert len(y) == 2
  assert y[0].shape == (batch_size, n_tasks, n_classes)
  assert y[0].shape == y[1].shape


@pytest.mark.torch
def test_cnn3d_torch():
  torch.manual_seed(0)

  n_tasks = 5
  n_features = 8
  n_classes = 7
  batch_size = 2
  mode = 'classification'
  model = CNNModule(n_tasks=n_tasks,
                    n_features=n_features,
                    dims=3,
                    dropouts=[0.5, 0.2, 0.4, 0.7],
                    layer_filters=[3, 8, 8, 16],
                    kernel_size=3,
                    n_classes=n_classes,
                    mode=mode,
                    uncertainty=False)

  x = torch.ones(batch_size, 8, 196, 196, 196)
  y = model(x)

  assert len(y) == 2
  assert y[0].shape == (batch_size, n_tasks, n_classes)
  assert y[0].shape == y[1].shape


@pytest.mark.torch
def test_1d_cnn_regression():
  """Test that a 2D CNN can overfit simple regression datasets."""

  np.random.seed(1)

  n_samples = 10
  n_features = 3
  n_tasks = 1

  X = np.random.rand(n_samples, n_features, 10)
  y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)

  dataset = dc.data.NumpyDataset(X, y)

  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
  model = CNN(n_tasks,
              n_features,
              layer_filters=[100],
              dims=1,
              dropouts=0.,
              kernel_size=3,
              mode='regression',
              learning_rate=0.003)

  # Fit trained model
  model.fit(dataset, nb_epoch=200)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])

  assert scores[regression_metric.name] < 0.3


@pytest.mark.torch
def test_2d_cnn_classification():
  """Test that a 2D CNN can overfit simple classification datasets."""

  n_features = 8
  n_tasks = 5
  n_samples = 10

  np.random.seed(1)
  X = np.random.rand(n_samples, n_features, 196, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)

  dataset = dc.data.NumpyDataset(X, y)

  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
  model = CNN(n_tasks,
              n_features,
              dims=2,
              dropouts=0.,
              kernel_size=3,
              mode='classification')

  # Fit trained model
  model.fit(dataset, nb_epoch=200)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])

  assert scores[classification_metric.name] > 0.55
