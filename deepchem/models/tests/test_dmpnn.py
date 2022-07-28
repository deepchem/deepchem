import pytest
import numpy as np
import torch
import deepchem as dc


@pytest.mark.torch
def test_dmpnn_regression():
  """
  """
  torch.manual_seed(0)
  from deepchem.models.torch_models.dmpnn import _MapperDMPNN, DMPNN

  # data
  input_smile = "CC"
  feat = dc.feat.DMPNNFeaturizer(features_generators=['morgan'])
  graph = feat.featurize(input_smile)

  mapper = _MapperDMPNN(graph[0])

  data = mapper.values

  number_of_tasks = 2
  number_of_molecules = 1
  morgan_feature_size = 2048
  # initialize the model
  model = DMPNN(mode='regression',
                global_features_size=morgan_feature_size,
                n_tasks=number_of_tasks,
                number_of_molecules=number_of_molecules)

  # get output
  output = model(data)
  assert output.shape == torch.Size([number_of_molecules, number_of_tasks])

  required_output = torch.tensor([[-0.0273, 0.0002]])
  assert torch.allclose(output[0], required_output, atol=1e-4)
