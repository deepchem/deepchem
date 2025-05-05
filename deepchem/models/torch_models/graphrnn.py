import torch
import torch.nn as nn
from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy


class NodeFeatureMLP(nn.Module):
    """
    MLP to generate node features sequentially.

    This module applies a two-layer feed-forward neural network with a ReLU activation
    between the layers to transform input node features into a desired output feature space.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Number of units in the hidden layer.
    output_dim : int
        Dimensionality of the output features.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NodeFeatureMLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, h):
        """
        Compute the output features from the input.

        Parameters
        ----------
        h : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        return self.mlp(h)


class EdgeRNN(nn.Module):
    """
    Edge-level RNN to generate edge features sequentially.

    This module uses a GRU to process a sequence of edge feature vectors and then applies
    a linear transformation to produce the final edge features.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input edge features.
    hidden_dim : int
        Number of hidden units in the GRU.
    edge_feature_dim : int
        Dimensionality of the output edge features.
    """

    def __init__(self, input_dim, hidden_dim, edge_feature_dim):
        super(EdgeRNN, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.edge_mlp = nn.Linear(hidden_dim, edge_feature_dim)

    def forward(self, edge_seq, h_prev):
        """
        Process the edge sequence to produce edge features.

        Parameters
        ----------
        edge_seq : torch.Tensor
            Input edge sequence tensor of shape (batch_size, num_neighbors, input_dim).
        h_prev : torch.Tensor
            Initial hidden state for the GRU, shape (1, batch_size, hidden_dim).

        Returns
        -------
        tuple of torch.Tensor
            A tuple (edge_features, h_next) where:
            - edge_features is a tensor of shape (batch_size, num_neighbors, edge_feature_dim).
            - h_next is the updated hidden state tensor of shape (1, batch_size, hidden_dim).
        """
        output, h_next = self.rnn(edge_seq, h_prev)
        edge_features = self.edge_mlp(output)
        return edge_features, h_next


class _GraphRNNTorchModel(nn.Module):
    """
    Core GraphRNN model with support for node and edge features.

    This model sequentially processes node features using an MLP and edge features using a GRU,
    generating a stack of node and edge representations.

    Parameters
    ----------
    node_input_dim : int
        Dimensionality of the node input features.
    node_hidden_dim : int
        Number of hidden units in the node MLP.
    node_feature_dim : int
        Dimensionality of the node output features.
    edge_input_dim : int
        Dimensionality of the edge input features.
    edge_hidden_dim : int
        Number of hidden units in the edge RNN.
    edge_feature_dim : int
        Dimensionality of the edge output features.
    num_neighbors : int
        Maximum number of neighbors (edge features) per node.
    """

    def __init__(self, node_input_dim, node_hidden_dim, node_feature_dim,
                 edge_input_dim, edge_hidden_dim, edge_feature_dim,
                 num_neighbors):
        super(_GraphRNNTorchModel, self).__init__()
        self.node_mlp = NodeFeatureMLP(node_input_dim, node_hidden_dim,
                                       node_feature_dim)
        self.edge_rnn = EdgeRNN(edge_input_dim, edge_hidden_dim,
                                edge_feature_dim)
        self.edge_hidden_dim = edge_hidden_dim
        self.num_neighbors = num_neighbors

    def forward(self, inputs):
        """
        Forward pass of the GraphRNN model.

        Parameters
        ----------
        inputs : tuple of torch.Tensor
            A tuple (node_seq, edge_seq) where:
            - node_seq has shape (batch_size, num_nodes, node_input_dim)
            - edge_seq has shape (batch_size, num_nodes, num_neighbors, edge_input_dim)

        Returns
        -------
        tuple of torch.Tensor
            A tuple (node_predictions, edge_predictions) where:
            - node_predictions is of shape (batch_size, num_nodes, node_feature_dim)
            - edge_predictions is of shape (batch_size, num_nodes, num_neighbors, edge_feature_dim)
        """
        node_seq, edge_seq = inputs
        batch_size, num_nodes, _ = node_seq.shape
        generated_nodes = []
        generated_edges = []
        h_edge = torch.zeros(1,
                             batch_size,
                             self.edge_hidden_dim,
                             device=node_seq.device)

        for i in range(num_nodes):
            # Process node features for the i-th node
            node_feat = self.node_mlp(node_seq[:, i, :])
            generated_nodes.append(node_feat)
            # Process edge features for the i-th node
            edge_feat, h_edge = self.edge_rnn(edge_seq[:, i, :, :], h_edge)
            generated_edges.append(edge_feat)

        return torch.stack(generated_nodes, dim=1), torch.stack(generated_edges,
                                                                dim=1)


class GraphRNN(TorchModel):
    """
    GraphRNN DeepChem Model Wrapper.

    This class wraps the core GraphRNN model for integration with DeepChem, providing
    a default data generator and selecting an appropriate loss function based on the mode.

    Parameters
    ----------
    node_input_dim : int
        Dimensionality of the node input features.
    node_hidden_dim : int
        Number of hidden units in the node MLP.
    node_feature_dim : int
        Dimensionality of the node output features.
    edge_input_dim : int
        Dimensionality of the edge input features.
    edge_hidden_dim : int
        Number of hidden units in the edge RNN.
    edge_feature_dim : int
        Dimensionality of the edge output features.
    learning_rate : float, optional (default=0.001)
        Learning rate for training.
    num_neighbors : int, optional (default=5)
        Maximum number of neighbors (edge features) per node.
    mode : str, optional (default='regression')
        Mode of the model; either 'regression' or 'classification'.
    n_classes : int, optional (default=2)
        Number of classes (used only in classification mode).

    Raises
    ------
    ValueError
        If an unsupported mode is provided.
    """

    def __init__(self,
                 node_input_dim,
                 node_hidden_dim,
                 node_feature_dim,
                 edge_input_dim,
                 edge_hidden_dim,
                 edge_feature_dim,
                 learning_rate=0.001,
                 num_neighbors=5,
                 mode='regression',
                 n_classes=2):
        self.mode = mode
        self.n_classes = n_classes
        model = _GraphRNNTorchModel(node_input_dim, node_hidden_dim,
                                    node_feature_dim, edge_input_dim,
                                    edge_hidden_dim, edge_feature_dim,
                                    num_neighbors)
        loss = L2Loss() if mode == 'regression' else SoftmaxCrossEntropy()
        super(GraphRNN, self).__init__(model,
                                       loss=loss,
                                       learning_rate=learning_rate)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        """
        Default data generator for training or evaluating the model.

        This generator iterates over a DeepChem dataset containing GraphData objects,
        pads the node and edge features to have uniform dimensions within each batch, and
        yields a tuple of inputs, targets, and sample weights.

        Parameters
        ----------
        dataset : dc.data.Dataset
            A DeepChem dataset where each sample is a GraphData object.
        epochs : int, optional (default=1)
            Number of epochs to iterate over the dataset.
        mode : str, optional (default='fit')
            Mode of operation; 'fit' for training or 'predict' for inference.
        deterministic : bool, optional (default=True)
            Whether to iterate over the dataset deterministically.
        pad_batches : bool, optional (default=True)
            Whether to pad batches to the model's preferred batch size.

        Yields
        ------
        tuple
            A tuple ([node_seq, edge_seq], y_tensor, w_tensor) where:
            - node_seq is a tensor of shape (batch_size, max_num_nodes, node_input_dim)
            - edge_seq is a tensor of shape (batch_size, max_num_nodes, num_neighbors, edge_input_dim)
            - y_tensor is a tensor of targets.
            - w_tensor is a tensor of sample weights (or None if not provided).
        """
        for epoch in range(epochs):
            for g_b, y_b, w_b, _ in dataset.iterbatches(
                    batch_size=self.batch_size,
                    deterministic=deterministic,
                    pad_batches=pad_batches):
                # g_b is a list of GraphData objects.
                # Extract node features; each g.node_features is (num_nodes, node_input_dim)
                node_seq_list = [
                    torch.tensor(g.node_features, dtype=torch.float32)
                    for g in g_b
                ]
                # Determine the maximum number of nodes in this batch.
                max_num_nodes = max([ns.shape[0] for ns in node_seq_list])
                # Pad node feature matrices to shape (max_num_nodes, node_input_dim)
                node_seq_padded = []
                for ns in node_seq_list:
                    if ns.shape[0] < max_num_nodes:
                        pad = torch.zeros(max_num_nodes - ns.shape[0],
                                          ns.shape[1],
                                          dtype=torch.float32)
                        ns = torch.cat([ns, pad], dim=0)
                    node_seq_padded.append(ns)
                node_seq = torch.stack(
                    node_seq_padded,
                    dim=0)  # (B, max_num_nodes, node_input_dim)

                # Extract edge features; assume each g.edge_features is (num_nodes, num_neighbors, edge_input_dim)
                edge_seq_list = [
                    torch.tensor(g.edge_features, dtype=torch.float32)
                    for g in g_b
                ]
                edge_seq_padded = []
                for es in edge_seq_list:
                    if es.shape[0] < max_num_nodes:
                        pad = torch.zeros(max_num_nodes - es.shape[0],
                                          es.shape[1],
                                          es.shape[2],
                                          dtype=torch.float32)
                        es = torch.cat([es, pad], dim=0)
                    edge_seq_padded.append(es)
                edge_seq = torch.stack(
                    edge_seq_padded,
                    dim=0)  # (B, max_num_nodes, num_neighbors, edge_input_dim)

                # Handle weights if provided
                w_tensor = torch.tensor(
                    w_b, dtype=torch.float32) if w_b is not None else None
                # Convert targets
                y_tensor = torch.tensor(y_b, dtype=torch.float32)

                yield ([node_seq, edge_seq], y_tensor, w_tensor)
