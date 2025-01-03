import torch
from torch import nn
from torch import Tensor
from typing import Sequence
from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.utils import scatter_reduce


class WeightedAttentionPooling(nn.Module):
    """
    Weighted attention pooling layer.

    Parameters
    ----------
    gate_nn: nn.Module
        Neural network to calculate attention scalars.
    message_nn: nn.Module
        Neural network to evaluate message updates.
    """

    def __init__(self, gate_nn: nn.Module, message_nn: nn.Module) -> None:
        """Initialize softmax attention layer.

        Parameters
        ----------
        gate_nn: nn.Module
            Neural network to calculate attention scalars.
        message_nn: nn.Module
            Neural network to evaluate message updates.
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor, index: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input features for nodes
        index: torch.Tensor
            The indices for scatter operation over nodes
        weights: torch.Tensor
            The weights to assign to nodes

        Returns
        -------
        torch.Tensor
            Output features for nodes
        """
        gate = self.gate_nn(x)

        gate -= scatter_reduce(gate, index, dim=0, reduce="amax")[index]
        gate = (weights**self.pow) * gate.exp()
        gate /= scatter_reduce(gate, index, dim=0, reduce="sum")[index] + 1e-10

        x = self.message_nn(x)
        return scatter_reduce(gate * x, index, dim=0, reduce="sum")


class MessageLayer(nn.Module):

    def __init__(
        self,
        msg_feature_len: int,
        num_msg_heads: int,
        msg_gate_layers: Sequence[int],
        msg_net_layers: Sequence[int],
    ) -> None:
        """
        Initialize the MessageLayer, consisting of a weighted attention pooling layer with MLPs for the gate and message networks.

        Parameters
        ----------
        msg_feature_len: int
            The length of the message features
        num_msg_heads: int
            The number of attention heads
        msg_gate_layers: Sequence[int]
            The number of hidden units in the message gate network
        msg_net_layers: Sequence[int]
            The number of hidden units in the message network
        """
        self.msg_feature_len = msg_feature_len
        self.num_msg_heads = num_msg_heads
        self.msg_gate_layers = msg_gate_layers
        self.msg_net_layers = msg_net_layers

        self.pooling = nn.ModuleList(
            WeightedAttentionPooling(
                gate_nn=MultilayerPerceptron(d_input=2 * msg_feature_len,
                                             d_output=1,
                                             d_hidden=msg_gate_layers),
                message_nn=MultilayerPerceptron(d_input=2 * msg_feature_len,
                                                d_output=msg_feature_len,
                                                d_hidden=msg_net_layers)))

    def forward(self, node_weights: Tensor, node_prev_features: Tensor,
                self_idx: torch.LongTensor,
                neighbor_idx: torch.LongTensor) -> Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        node_weights: torch.Tensor
            The fractional weights of elements in their materials
        node_prev_features: torch.Tensor
            Node hidden features before message passing
        self_idx: torch.LongTensor
            Indices of the 1st element in each of the node pairs
        neighbor_idx: torch.LongTensor
            Indices of the 2nd element in each of the node pairs
        """

        # construct the total features for passing
        node_neighbor_weights = node_weights[neighbor_idx, :]
        msg_neighbor_features = node_prev_features[neighbor_idx, :]
        msg_self_features = node_prev_features[self_idx, :]
        message = torch.cat([msg_self_features, msg_neighbor_features], dim=1)

        # sum selectivity over the neighbors to get node updates
        head_features = []
        for attn_head in self.pooling:
            output_msg = attn_head(message,
                                   index=self_idx,
                                   weights=node_neighbor_weights)
            head_features.append(output_msg)

        # average the attention heads
        node_update = torch.stack(head_features).mean(dim=0)

        return node_update + node_prev_features
