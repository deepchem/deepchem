import torch
import numpy as np
from deepchem.feat.graph_data import GraphData
from torch_geometric.nn import global_mean_pool
from deepchem.models.losses import GraphInfomaxLoss
torch.manual_seed(123)
x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
edge_index = np.array([[0, 1, 2, 0, 3], [1, 0, 1, 3, 2]])
graph_index = np.array([0, 0, 1, 1])
data = GraphData(node_features=x, edge_index=edge_index, graph_index=graph_index).numpy_to_torch()

graph_infomax_loss = GraphInfomaxLoss()._create_pytorch_loss()

# Initialize node_emb randomly without using a GCN
num_nodes = data.num_nodes
embedding_dim = 8
node_emb = torch.randn(num_nodes, embedding_dim)

# Compute the global graph representation
summary_emb = global_mean_pool(node_emb, data.graph_index)

# Compute positive and negative scores
positive_score = torch.matmul(node_emb, summary_emb.t())
negative_score = torch.matmul(node_emb, summary_emb.roll(1, dims=0).t())

# Compute the loss
loss = graph_infomax_loss(positive_score, negative_score)

print("Loss:", loss)