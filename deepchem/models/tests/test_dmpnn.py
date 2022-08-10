import pytest
import torch
import deepchem as dc


@pytest.mark.torch
def test_dmpnn_regression():
  """
  Test DMPNN class for regression mode
  """
  torch.manual_seed(0)
  from deepchem.models.torch_models.dmpnn import _MapperDMPNN, DMPNN

  # get data
  input_smile = "CC"
  feat = dc.feat.DMPNNFeaturizer(features_generators=['morgan'])
  graph = feat.featurize(input_smile)

  mapper = _MapperDMPNN(graph[0])
  atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features = mapper.values

  atom_features = torch.from_numpy(atom_features).float()
  f_ini_atoms_bonds = torch.from_numpy(f_ini_atoms_bonds).float()
  atom_to_incoming_bonds = torch.from_numpy(atom_to_incoming_bonds)
  mapping = torch.from_numpy(mapping)
  global_features = torch.from_numpy(global_features).float()

  data = [
      atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping,
      global_features
  ]

  # initialize the model
  number_of_tasks = 2
  number_of_molecules = 1
  morgan_feature_size = 2048
  model = DMPNN(mode='regression',
                global_features_size=morgan_feature_size,
                n_tasks=number_of_tasks,
                number_of_molecules=number_of_molecules)

  assert len(model.encoders) == 1
  assert model.encoders[0].__repr__(
  ) == 'DMPNNEncoderLayer(\n  (activation): ReLU()\n  (dropout): Dropout(p=0.0, inplace=False)\n  (W_i): Linear(in_features=147, out_features=300, bias=False)\n'\
       '  (W_h): Linear(in_features=300, out_features=300, bias=False)\n  (W_o): Linear(in_features=433, out_features=300, bias=True)\n)'
  assert model.ffn.__repr__(
  ) == 'PositionwiseFeedForward(\n  (activation): ReLU()\n  (linears): ModuleList(\n    (0): Linear(in_features=2348, out_features=300, bias=True)\n    '\
       '(1): Linear(in_features=300, out_features=300, bias=True)\n    (2): Linear(in_features=300, out_features=2, bias=True)\n  )\n  (dropout_p): ModuleList(\n    (0): Dropout(p=0.0, inplace=False)\n    '\
       '(1): Dropout(p=0.0, inplace=False)\n    (2): Dropout(p=0.0, inplace=False)\n  )\n)'

  # get output
  output = model(data)
  assert output.shape == torch.Size([number_of_molecules, number_of_tasks])

  required_output = torch.tensor([[0.0044, -0.0572]])
  assert torch.allclose(output[0], required_output, atol=1e-4)


def test_dmpnn_classification_single_task():
  """
  Test DMPNN class for classification mode with 1 task
  """
  torch.manual_seed(0)
  from deepchem.models.torch_models.dmpnn import _MapperDMPNN, DMPNN

  # get data
  input_smile = "CC"
  feat = dc.feat.DMPNNFeaturizer(features_generators=['morgan'])
  graph = feat.featurize(input_smile)

  mapper = _MapperDMPNN(graph[0])
  atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features = mapper.values

  atom_features = torch.from_numpy(atom_features).float()
  f_ini_atoms_bonds = torch.from_numpy(f_ini_atoms_bonds).float()
  atom_to_incoming_bonds = torch.from_numpy(atom_to_incoming_bonds)
  mapping = torch.from_numpy(mapping)
  global_features = torch.from_numpy(global_features).float()

  data = [
      atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping,
      global_features
  ]

  # initialize the model
  number_of_tasks = 1
  number_of_classes = 2
  number_of_molecules = 1
  morgan_feature_size = 2048
  model = DMPNN(mode='classification',
                n_classes=number_of_classes,
                global_features_size=morgan_feature_size,
                n_tasks=number_of_tasks,
                number_of_molecules=number_of_molecules)

  assert len(model.encoders) == 1
  assert model.encoders[0].__repr__(
  ) == 'DMPNNEncoderLayer(\n  (activation): ReLU()\n  (dropout): Dropout(p=0.0, inplace=False)\n  (W_i): Linear(in_features=147, out_features=300, bias=False)\n  (W_h): Linear(in_features=300, out_features=300, bias=False)\n'\
       '  (W_o): Linear(in_features=433, out_features=300, bias=True)\n)'
  assert model.ffn.__repr__(
  ) == 'PositionwiseFeedForward(\n  (activation): ReLU()\n  (linears): ModuleList(\n    (0): Linear(in_features=2348, out_features=300, bias=True)\n    (1): Linear(in_features=300, out_features=300, bias=True)\n    '\
       '(2): Linear(in_features=300, out_features=2, bias=True)\n  )\n  (dropout_p): ModuleList(\n    (0): Dropout(p=0.0, inplace=False)\n    (1): Dropout(p=0.0, inplace=False)\n    (2): Dropout(p=0.0, inplace=False)\n  )\n)'

  # get output
  output = model(data)
  assert output.shape == torch.Size([number_of_molecules, number_of_classes])

  required_output = torch.tensor([[0.5154, 0.4846]])
  assert torch.allclose(output[0], required_output, atol=1e-4)


def test_dmpnn_classification_multi_task():
  """
  Test DMPNN class for classification mode with more than 1 task
  """
  torch.manual_seed(0)
  from deepchem.models.torch_models.dmpnn import _MapperDMPNN, DMPNN

  # get data
  input_smile = "CC"
  feat = dc.feat.DMPNNFeaturizer(features_generators=['morgan'])
  graph = feat.featurize(input_smile)

  mapper = _MapperDMPNN(graph[0])
  atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features = mapper.values

  atom_features = torch.from_numpy(atom_features).float()
  f_ini_atoms_bonds = torch.from_numpy(f_ini_atoms_bonds).float()
  atom_to_incoming_bonds = torch.from_numpy(atom_to_incoming_bonds)
  mapping = torch.from_numpy(mapping)
  global_features = torch.from_numpy(global_features).float()

  data = [
      atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping,
      global_features
  ]

  # initialize the model
  number_of_tasks = 2
  number_of_classes = 2
  number_of_molecules = 1
  morgan_feature_size = 2048
  model = DMPNN(mode='classification',
                n_classes=number_of_classes,
                global_features_size=morgan_feature_size,
                n_tasks=number_of_tasks,
                number_of_molecules=number_of_molecules)

  assert len(model.encoders) == 1
  assert model.encoders[0].__repr__(
  ) == 'DMPNNEncoderLayer(\n  (activation): ReLU()\n  (dropout): Dropout(p=0.0, inplace=False)\n  (W_i): Linear(in_features=147, out_features=300, bias=False)\n  (W_h): Linear(in_features=300, out_features=300, bias=False)\n  (W_o): Linear(in_features=433, out_features=300, bias=True)\n)'
  assert model.ffn.__repr__(
  ) == 'PositionwiseFeedForward(\n  (activation): ReLU()\n  (linears): ModuleList(\n    (0): Linear(in_features=2348, out_features=300, bias=True)\n    (1): Linear(in_features=300, out_features=300, bias=True)\n    '\
       '(2): Linear(in_features=300, out_features=4, bias=True)\n  )\n  (dropout_p): ModuleList(\n    (0): Dropout(p=0.0, inplace=False)\n    (1): Dropout(p=0.0, inplace=False)\n    (2): Dropout(p=0.0, inplace=False)\n  )\n)'

  # get output
  output = model(data)
  assert output.shape == torch.Size(
      [number_of_molecules, number_of_tasks, number_of_classes])

  required_output = torch.tensor([[[0.5317, 0.4683], [0.4911, 0.5089]]])
  assert torch.allclose(output[0], required_output, atol=1e-4)
