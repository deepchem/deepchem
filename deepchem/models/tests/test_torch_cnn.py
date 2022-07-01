import pytest
try:
  import torch
  from deepchem.models import TorchCNN
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
  model = TorchCNN(n_tasks=n_tasks,
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
  model = TorchCNN(n_tasks=n_tasks,
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
  model = TorchCNN(n_tasks=n_tasks,
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

  # output_type for classification = [output, logits]
  assert len(y) == 2
  assert y[0].shape == (batch_size, n_tasks, n_classes)
  assert y[0].shape == y[1].shape
