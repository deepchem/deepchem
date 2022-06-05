import pytest
from deepchem.models.torch_models.convode import ConvODEAutoEncoder
import deepchem.models.torch_models.layers as torch_layers
from deepchem.molnet import load_delaney
from deepchem.metrics import Metric, pearson_r2_score


@pytest.mark.torch
def test_ConvODEAutoEncoder():
  encoder = torch_layers.ConvODEEncoderLayer(dim=32)
  system_dynamics_net = torch_layers.SystemDynamics(dim=32)
  system_dynamics_block = torch_layers.ODEBlock(
      system_dynamics=system_dynamics_net)
  decoder = torch_layers.ConvODEDecoderLayer(dim=32)

  model = ConvODEAutoEncoder(encoder, system_dynamics_block, decoder)

  tasks, dataset, transformers = load_delaney()
  train_set, valid_set, test_set = dataset
  metric = Metric(pearson_r2_score)
