import torch
import torch.nn as nn
from deepchem.data import NumpyDataset
from deepchem.feat import CircularFingerprint
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L1Loss


class SCScore(nn.Module):
    """
    SCScore paper - https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00622

    SCScore is a FeedForward neural network with 5 hidden layers. Each hidden layer
    contains 300 neurons. ReLU is applied for all the hidden layers. Sigmoid is applied
    for the output layer. In the original paper, they didn't use Dropout. But in our
    implementation, you can set the dropout for all the hidden layers at once.
    """
    
    def __init__(self, n_features: int, layer_sizes: list, dropout: int, **kwargs) -> None:
        """
        Arguments
        """
        super().__init__()
        self.dropout = dropout
        self.input_layer = nn.Linear(n_features, layer_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for layer_size in layer_sizes[1:]:
            self.hidden_layers.append(nn.Linear(layer_size, layer_size))
        self.output_layer = nn.Linear(layer_sizes[-1],1)
        
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            if self.dropout > 0:
                x = nn.Dropout(p=self.dropout)(x)
            x = nn.ReLU()(x)
        output = self.output_layer(x)
        output = nn.Sigmoid()(output)
        return output
        
        
class SCScoreModel(TorchModel):
    """
    
    """
    
    def __init__(self, n_features=1024, layer_sizes=[300,300,300,300,300], dropout = 0, **kwargs):
        self.n_features = n_features
        model = SCScore(n_features, layer_sizes, dropout)
        output_types = ['prediction']
        super(SCScoreModel, self).__init__(
             model, L1Loss(), output_types=output_types, **kwargs)
