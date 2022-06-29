import torch
import pytest


@pytest.mark.torch
def test_cnn_torch():

  from deepchem.models.torch_models import CNN
  torch.manual_seed(0)

  n_tasks = 5
  n_features = 8
  n_classes = 7
  batch_size = 2
  mode = 'classification'
  model = CNN(n_tasks=n_tasks,
              n_features=n_features,
              dims=2,
              dropouts=[0.5, 0.2, 0.4, 0.7],
              layer_filters=[3, 8, 8, 16],
              kernel_size=3,
              n_classes=n_classes,
              mode=mode,
              uncertainty=False)

  x = torch.Tensor(batch_size, 8, 224, 224)
  y = model(x)

  # output_type for classification = [output, logits]
  assert len(y) == 2
  assert y[0].shape == (batch_size, n_tasks, n_classes)
  assert y[0].shape == y[1].shape
