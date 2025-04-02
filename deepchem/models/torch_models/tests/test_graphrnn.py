import pytest
import torch
import numpy as np
import deepchem as dc
from deepchem.feat.graph_data import GraphData
from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
from deepchem.models.torch_models.graphrnn import GraphRNN

@pytest.fixture
def synthetic_dataset():
    """
    Create a synthetic DeepChem dataset consisting of GraphData objects.

    Each GraphData object contains random node and edge features along with an 
    adjacency list and corresponding edge_index constructed from it.
    
    Returns
    -------
    dc.data.NumpyDataset
        A DeepChem NumpyDataset where X is a list of GraphData objects and y is
        a randomly generated target array.
    """
    batch_size = 4
    num_nodes = 10
    num_neighbors = 5
    node_feature_dim = 16
    edge_feature_dim = 8

    graphs = []
    for _ in range(batch_size):
        # Create synthetic node features: shape (num_nodes, node_feature_dim)
        node_features = np.random.randn(num_nodes, node_feature_dim)
        # Create synthetic edge features: shape (num_nodes, num_neighbors, edge_feature_dim)
        edge_features = np.random.randn(num_nodes, num_neighbors, edge_feature_dim)
        # Create a simple adjacency list: each node i lists its next num_neighbors nodes (with wrap-around)
        adjacency_list = {i: [(i + j + 1) % num_nodes for j in range(num_neighbors)] for i in range(num_nodes)}
        # Construct edge_index from the adjacency list
        edge_index_0, edge_index_1 = [], []
        for i in range(num_nodes):
            for neigh in adjacency_list[i]:
                edge_index_0.append(i)
                edge_index_1.append(neigh)
        edge_index = np.array([edge_index_0, edge_index_1])
        # Create the GraphData object (requires edge_index)
        graph = GraphData(node_features=node_features,
                          edge_features=edge_features,
                          adjacency_list=adjacency_list,
                          edge_index=edge_index)
        graphs.append(graph)
    
    # Create a NumpyDataset using these graphs as X and random targets as y
    dataset = dc.data.NumpyDataset(X=graphs, y=np.random.randn(batch_size, 1))
    return dataset

def test_model_initialization(synthetic_dataset):
    """
    Test that the GraphRNN model initializes correctly.

    This test creates a GraphRNN instance with specified dimensions and verifies that 
    the model is not None.
    """
    node_feature_dim = 16
    edge_feature_dim = 8
    num_neighbors = 5
    model = GraphRNN(node_input_dim=node_feature_dim, node_hidden_dim=32, node_feature_dim=node_feature_dim,
                     edge_input_dim=edge_feature_dim, edge_hidden_dim=32, edge_feature_dim=edge_feature_dim,
                     num_neighbors=num_neighbors, mode='regression')
    assert model is not None, "GraphRNN model should initialize without error."

def test_forward_pass(synthetic_dataset):
    """
    Test the forward pass of the GraphRNN model.

    This test uses the model's default_generator to fetch a batch of inputs, then 
    directly calls the model's forward function to produce node and edge predictions.
    It verifies that the output shapes match the expected dimensions.

    Expected shapes:
      - Node predictions: (batch_size, num_nodes, node_feature_dim)
      - Edge predictions: (batch_size, num_nodes, num_neighbors, edge_feature_dim)
    """
    node_feature_dim = 16
    edge_feature_dim = 8
    num_neighbors = 5
    model = GraphRNN(node_input_dim=node_feature_dim, node_hidden_dim=32, node_feature_dim=node_feature_dim,
                     edge_input_dim=edge_feature_dim, edge_hidden_dim=32, edge_feature_dim=edge_feature_dim,
                     num_neighbors=num_neighbors, mode='regression')
    
    # Retrieve one batch from the default generator
    generator = model.default_generator(synthetic_dataset, epochs=1)
    inputs, targets, weights = next(generator)
    
    node_preds, edge_preds = model.model.forward(inputs)
    
    # Verify node predictions shape
    assert node_preds.shape[0] == 4, "Batch size should be 4."
    assert node_preds.shape[1] == 10, "Number of nodes should be 10."
    assert node_preds.shape[2] == node_feature_dim, f"Node feature dim should be {node_feature_dim}."
    
    # Verify edge predictions shape
    assert edge_preds.shape[0] == 4, "Batch size should be 4."
    assert edge_preds.shape[1] == 10, "Number of nodes should be 10."
    assert edge_preds.shape[2] == num_neighbors, f"Number of neighbors should be {num_neighbors}."
    assert edge_preds.shape[3] == edge_feature_dim, f"Edge feature dim should be {edge_feature_dim}."

def test_training(synthetic_dataset):
    """
    Test the training process of the GraphRNN model.

    This test evaluates the initial loss using a DeepChem metric (Mean Absolute Error),
    trains the model for a couple of epochs, and checks that the loss decreases.
    """
    node_feature_dim = 16
    edge_feature_dim = 8
    num_neighbors = 5
    model = GraphRNN(node_input_dim=node_feature_dim, node_hidden_dim=32, node_feature_dim=node_feature_dim,
                     edge_input_dim=edge_feature_dim, edge_hidden_dim=32, edge_feature_dim=edge_feature_dim,
                     num_neighbors=num_neighbors, mode='regression')
    
    # Define a metric for evaluation
    mae_metric = dc.metrics.Metric(dc.metrics.mean_absolute_error, np.mean)
    initial_scores = model.evaluate(synthetic_dataset, [mae_metric])
    initial_loss = initial_scores.get('mean_absolute_error', None)
    assert initial_loss is not None, "Initial loss should be computed."
    
    # Train the model for two epochs
    model.fit(synthetic_dataset, nb_epoch=2)
    
    final_scores = model.evaluate(synthetic_dataset, [mae_metric])
    final_loss = final_scores.get('mean_absolute_error', None)
    assert final_loss is not None, "Final loss should be computed."
    assert final_loss < initial_loss, "Loss should decrease after training."

def test_prediction_shape(synthetic_dataset):
    """
    Test that the predictions from the GraphRNN model have the expected shape.

    After training, the model should return node predictions of shape 
    (batch_size, num_nodes, node_feature_dim).
    """
    node_feature_dim = 16
    edge_feature_dim = 8
    num_neighbors = 5
    model = GraphRNN(node_input_dim=node_feature_dim, node_hidden_dim=32, node_feature_dim=node_feature_dim,
                     edge_input_dim=edge_feature_dim, edge_hidden_dim=32, edge_feature_dim=edge_feature_dim,
                     num_neighbors=num_neighbors, mode='regression')
    
    model.fit(synthetic_dataset, nb_epoch=1)
    preds = model.predict(synthetic_dataset)
    expected_shape = (4, 10, node_feature_dim)
    assert preds.shape == expected_shape, f"Expected prediction shape {expected_shape}, got {preds.shape}"

def test_metric_evaluation(synthetic_dataset):
    """
    Test the compatibility of the GraphRNN model with DeepChem's metric evaluation.

    This test ensures that the model can evaluate metrics (e.g., mean absolute error) correctly,
    and that the metric value returned is a float.
    """
    node_feature_dim = 16
    edge_feature_dim = 8
    num_neighbors = 5
    model = GraphRNN(node_input_dim=node_feature_dim, node_hidden_dim=32, node_feature_dim=node_feature_dim,
                     edge_input_dim=edge_feature_dim, edge_hidden_dim=32, edge_feature_dim=edge_feature_dim,
                     num_neighbors=num_neighbors, mode='regression')
    
    model.fit(synthetic_dataset, nb_epoch=1)
    mae_metric = dc.metrics.Metric(dc.metrics.mean_absolute_error, np.mean)
    scores = model.evaluate(synthetic_dataset, [mae_metric], transformers=[])
    assert 'mean_absolute_error' in scores, "Metric evaluation should return 'mean_absolute_error'."
    assert isinstance(scores['mean_absolute_error'], float), "Metric value should be a float."
