import pytest
import deepchem as dc
import tempfile
import numpy as np
import os

try:
    import torch
    has_torch = True
except:
    has_torch = False


@pytest.mark.torch
def test_dmpnn_regression():
    """
  Test DMPNN class for regression mode
  """
    try:
        from torch_geometric.data import Data, Batch
    except ModuleNotFoundError:
        raise ImportError(
            "This test requires PyTorch Geometric to be installed.")

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
        Data(atom_features=atom_features,
             f_ini_atoms_bonds=f_ini_atoms_bonds,
             atom_to_incoming_bonds=atom_to_incoming_bonds,
             mapping=mapping,
             global_features=global_features)
    ]

    # prepare batch (size 1)
    pyg_batch = Batch()
    pyg_batch = pyg_batch.from_data_list(data)

    # initialize the model
    number_of_tasks = 2
    number_of_molecules = 1
    morgan_feature_size = 2048
    model = DMPNN(mode='regression',
                  global_features_size=morgan_feature_size,
                  n_tasks=number_of_tasks)

    assert model.encoder.__repr__(
    ) == 'DMPNNEncoderLayer(\n  (activation): ReLU()\n  (dropout): Dropout(p=0.0, inplace=False)\n  (W_i): Linear(in_features=147, out_features=300, bias=False)\n'\
         '  (W_h): Linear(in_features=300, out_features=300, bias=False)\n  (W_o): Linear(in_features=433, out_features=300, bias=True)\n)'
    assert model.ffn.__repr__(
    ) == 'PositionwiseFeedForward(\n  (activation): ReLU()\n  (linears): ModuleList(\n    (0): Linear(in_features=2348, out_features=300, bias=True)\n    '\
         '(1): Linear(in_features=300, out_features=300, bias=True)\n    (2): Linear(in_features=300, out_features=2, bias=True)\n  )\n  (dropout_p): ModuleList(\n    (0-2): 3 x Dropout(p=0.0, inplace=False)\n  )\n)'

    # get output
    output = model(pyg_batch)
    assert output.shape == torch.Size([number_of_molecules, number_of_tasks])

    required_output = torch.tensor([[0.0044, -0.0572]])
    assert torch.allclose(output[0], required_output, atol=1e-4)


@pytest.mark.torch
def test_dmpnn_classification_single_task():
    """
  Test DMPNN class for classification mode with 1 task
  """
    try:
        from torch_geometric.data import Data, Batch
    except ModuleNotFoundError:
        raise ImportError(
            "This test requires PyTorch Geometric to be installed.")

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
        Data(atom_features=atom_features,
             f_ini_atoms_bonds=f_ini_atoms_bonds,
             atom_to_incoming_bonds=atom_to_incoming_bonds,
             mapping=mapping,
             global_features=global_features)
    ]

    # prepare batch (size 1)
    pyg_batch = Batch()
    pyg_batch = pyg_batch.from_data_list(data)

    # initialize the model
    number_of_tasks = 1
    number_of_classes = 2
    number_of_molecules = 1
    morgan_feature_size = 2048
    model = DMPNN(mode='classification',
                  n_classes=number_of_classes,
                  global_features_size=morgan_feature_size,
                  n_tasks=number_of_tasks)

    assert model.encoder.__repr__(
    ) == 'DMPNNEncoderLayer(\n  (activation): ReLU()\n  (dropout): Dropout(p=0.0, inplace=False)\n  (W_i): Linear(in_features=147, out_features=300, bias=False)\n  (W_h): Linear(in_features=300, out_features=300, bias=False)\n'\
         '  (W_o): Linear(in_features=433, out_features=300, bias=True)\n)'
    assert model.ffn.__repr__(
    ) == 'PositionwiseFeedForward(\n  (activation): ReLU()\n  (linears): ModuleList(\n    (0): Linear(in_features=2348, out_features=300, bias=True)\n    (1): Linear(in_features=300, out_features=300, bias=True)\n    '\
         '(2): Linear(in_features=300, out_features=2, bias=True)\n  )\n  (dropout_p): ModuleList(\n    (0-2): 3 x Dropout(p=0.0, inplace=False)\n  )\n)'

    # get output
    output = model(pyg_batch)
    assert len(output) == 2
    assert output[0].shape == torch.Size(
        [number_of_molecules, number_of_classes])
    assert output[1].shape == torch.Size(
        [number_of_molecules, number_of_classes])

    required_output = torch.tensor([[0.5154,
                                     0.4846]]), torch.tensor([[0.0044,
                                                               -0.0572]])
    assert torch.allclose(output[0][0], required_output[0], atol=1e-4)
    assert torch.allclose(output[1][0], required_output[1], atol=1e-4)


@pytest.mark.torch
def test_dmpnn_classification_multi_task():
    """
  Test DMPNN class for classification mode with more than 1 task
  """
    try:
        from torch_geometric.data import Data, Batch
    except ModuleNotFoundError:
        raise ImportError(
            "This test requires PyTorch Geometric to be installed.")

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
        Data(atom_features=atom_features,
             f_ini_atoms_bonds=f_ini_atoms_bonds,
             atom_to_incoming_bonds=atom_to_incoming_bonds,
             mapping=mapping,
             global_features=global_features)
    ]

    # prepare batch (size 1)
    pyg_batch = Batch()
    pyg_batch = pyg_batch.from_data_list(data)

    # initialize the model
    number_of_tasks = 2
    number_of_classes = 2
    number_of_molecules = 1
    morgan_feature_size = 2048
    model = DMPNN(mode='classification',
                  n_classes=number_of_classes,
                  global_features_size=morgan_feature_size,
                  n_tasks=number_of_tasks)

    assert model.encoder.__repr__(
    ) == 'DMPNNEncoderLayer(\n  (activation): ReLU()\n  (dropout): Dropout(p=0.0, inplace=False)\n  (W_i): Linear(in_features=147, out_features=300, bias=False)\n  (W_h): Linear(in_features=300, out_features=300, bias=False)\n  (W_o): Linear(in_features=433, out_features=300, bias=True)\n)'
    assert model.ffn.__repr__(
    ) == 'PositionwiseFeedForward(\n  (activation): ReLU()\n  (linears): ModuleList(\n    (0): Linear(in_features=2348, out_features=300, bias=True)\n    (1): Linear(in_features=300, out_features=300, bias=True)\n    '\
         '(2): Linear(in_features=300, out_features=4, bias=True)\n  )\n  (dropout_p): ModuleList(\n    (0-2): 3 x Dropout(p=0.0, inplace=False)\n  )\n)'

    # get output
    output = model(pyg_batch)
    assert len(output) == 2
    assert output[0].shape == torch.Size(
        [number_of_molecules, number_of_tasks, number_of_classes])
    assert output[1].shape == torch.Size(
        [number_of_molecules, number_of_tasks, number_of_classes])

    required_output = torch.tensor([[[0.5317, 0.4683], [0.4911, 0.5089]]
                                   ]), torch.tensor([[[0.0545, -0.0724],
                                                      [0.0204, 0.0558]]])
    assert torch.allclose(output[0][0], required_output[0], atol=1e-4)
    assert torch.allclose(output[1][0], required_output[1], atol=1e-4)


@pytest.mark.torch
def test_dmpnn_model_regression():
    """
  Test DMPNNModel class for regression mode
  """
    torch.manual_seed(0)

    # load sample dataset
    dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(dir, 'assets/freesolv_sample_5.csv')
    loader = dc.data.CSVLoader(tasks=['y'],
                               feature_field='smiles',
                               featurizer=dc.feat.DMPNNFeaturizer())
    dataset = loader.create_dataset(input_file)

    # initialize the model
    from deepchem.models.torch_models.dmpnn import DMPNNModel
    model = DMPNNModel(batch_size=2)

    # overfit test
    model.fit(dataset, nb_epoch=30)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.5


@pytest.mark.torch
def test_dmpnn_model_classification():
    """
  Test DMPNNModel class for classification mode
  """
    torch.manual_seed(0)

    # load sample dataset
    dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(dir, 'assets/example_classification.csv')
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=dc.feat.DMPNNFeaturizer())
    dataset = loader.create_dataset(input_file)

    # initialize the model
    from deepchem.models.torch_models.dmpnn import DMPNNModel

    mode = 'classification'
    classes = 2
    tasks = 1
    model = DMPNNModel(mode=mode,
                       n_classes=classes,
                       n_tasks=tasks,
                       batch_size=2)

    # overfit test
    model.fit(dataset, nb_epoch=30)
    metric = dc.metrics.Metric(dc.metrics.accuracy_score, mode="classification")
    scores = model.evaluate(dataset, [metric], n_classes=classes)
    assert scores['accuracy_score'] > 0.9


@pytest.mark.torch
def test_dmpnn_model_reload():
    """
  Test DMPNNModel class for reloading the model
  """
    torch.manual_seed(0)

    # load sample dataset
    dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(dir, 'assets/freesolv_sample_5.csv')
    loader = dc.data.CSVLoader(tasks=['y'],
                               feature_field='smiles',
                               featurizer=dc.feat.DMPNNFeaturizer())
    dataset = loader.create_dataset(input_file)

    # initialize the model
    from deepchem.models.torch_models.dmpnn import DMPNNModel
    model_dir = tempfile.mkdtemp()
    model = DMPNNModel(model_dir=model_dir, batch_size=2)

    # fit the model
    model.fit(dataset, nb_epoch=10)

    # reload the model
    reloaded_model = DMPNNModel(model_dir=model_dir, batch_size=2)
    reloaded_model.restore()

    orig_predict = model.predict(dataset)
    reloaded_predict = reloaded_model.predict(dataset)
    assert np.all(orig_predict == reloaded_predict)
