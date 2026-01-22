import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.torch_models import TorchModel
from typing import Optional, List, Dict

try:
    from torch_geometric.nn import GATv2Conv, SAGEConv, global_mean_pool
    from torch_geometric.data import Batch
except ImportError:
    raise ImportError("This module requires PyTorch Geometric. Install with `pip install torch_geometric`.")

class GATv2Net(nn.Module):
    """GATv2 network for molecular graphs using dynamic attention."""
    def __init__(
        self,
        node_dim: int = 30,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_tasks: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
        # Input layer
        self.convs.append(GATv2Conv(node_dim, hidden_dim, heads=num_heads, dropout=dropout))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                )
            )
        # Output layer
        self.output = nn.Linear(hidden_dim * num_heads, num_tasks)

    def forward(self, graphs: Batch) -> torch.Tensor:
        x, edge_index, batch = graphs.x, graphs.edge_index, graphs.batch
        # Apply GATv2 layers with skip connections
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Global pooling and output
        x = global_mean_pool(x, batch)
        return self.output(x)

class GraphSAGENet(nn.Module):
    """GraphSAGE network for inductive molecular learning."""
    def __init__(
        self,
        node_dim: int = 30,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_tasks: int = 1,
        aggregator: str = 'mean',  # 'mean', 'pool', 'lstm'
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(node_dim, hidden_dim, aggr=aggregator))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        # Output layer
        self.output = nn.Linear(hidden_dim, num_tasks)

    def forward(self, graphs: Batch) -> torch.Tensor:
        x, edge_index, batch = graphs.x, graphs.edge_index, graphs.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.output(x)

class GATv2Model(TorchModel):
    """GATv2Model for molecular property prediction."""
    def __init__(
        self,
        n_tasks: int,
        mode: str = 'regression',
        node_dim: int = 30,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ):
        model = GATv2Net(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_tasks=n_tasks,
            dropout=dropout,
        )
        super().__init__(
            model=model,
            loss=torch.nn.MSELoss() if mode == 'regression' else torch.nn.BCEWithLogitsLoss(),
            output_types=['prediction'],
            **kwargs,
        )

class GraphSAGEModel(TorchModel):
    """GraphSAGEModel for scalable inductive learning."""
    def __init__(
        self,
        n_tasks: int,
        mode: str = 'regression',
        node_dim: int = 30,
        hidden_dim: int = 64,
        num_layers: int = 2,
        aggregator: str = 'mean',
        **kwargs,
    ):
        model = GraphSAGENet(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_tasks=n_tasks,
            aggregator=aggregator,
        )
        super().__init__(
            model=model,
            loss=torch.nn.MSELoss() if mode == 'regression' else torch.nn.BCEWithLogitsLoss(),
            output_types=['prediction'],
            **kwargs,
        )
