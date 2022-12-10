import torch
import torch.nn as nn
from deepchem.data import NumpyDataset
from deepchem.feat import CircularFingerprint
from deepchem.models.torch_models.torch_model import TorchModel


class SCScore(nn.Module):
    """
    SCScore is a Feed-Forward Neural network with 5 layers.
    Each layer consists of only 300 neurons.
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
        #input tensor is of shape - (batch_size, 2, n_features)
        x = torch.cat((x[:,0],x[:,1]),dim=0)
        x = nn.ReLU()(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            if self.dropout > 0:
                x = nn.Dropout(p=self.dropout)(x)
            x = nn.ReLU()(x)
        output = self.output_layer(x)
        output = nn.Sigmoid()(output) * 4 + 1
        
        #output shape - (batch_size*2, 1)
        selector = output.shape[0] // 2
        output = [output[:selector], output[selector:]]
        #output type is a list
        #output[0] is reactant, output[1] is product
        return output

    
#we'll define our own loss function, since it doesn't require labels, weights, only just outputs.
#we're not also inheriting from deepchem Loss fucntion, since it requires ouputs, labels and weights.
class SCScoreLoss(object):
    """
    Defines the Loss for the SCScore Model.
    The Loss for the SCScore is a HingeLoss or a shifted relu loss.
    """
    def __init__(self, offset):
        self.offset = offset

    def _create_pytorch_loss(self):

        def loss(output, labels, weights):
            #calculates loss on the batch
            #doesn't make use of labels, weights
            #output[0] is reactant, output[1] is product
            loss_item = nn.ReLU() (self.offset + output[0] - output[1])
            loss_item = torch.sum(loss_item)
            return loss_item
        
        return loss
        

class SCScoreModel(TorchModel):
    """
    SCScore paper link - https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00622
    """
    
    def __init__(self, n_features=1024, layer_sizes=[300,300,300,300,300], dropout = 0, offset=0.25, **kwargs):
        self.n_features = n_features
        model = SCScore(n_features, layer_sizes, dropout)
        loss = SCScoreLoss(offset)._create_pytorch_loss()
        output_types = ["prediction", "prediction"]
        super(SCScoreModel, self).__init__(
             model, loss, output_types=output_types, **kwargs)

    def default_generator(self, dataset, epochs=1, mode='fit', deterministic=True, pad_batches=True):

        for epoch in range(epochs):
            for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
                batch_size=self.batch_size,
                deterministic=deterministic,
                pad_batches=pad_batches):
                #yield ([X_b[:, 0], X_b[:, 1]], [y_b], [w_b])
                yield ([X_b], [y_b], [w_b])  
    
    def predict_mols(self, mols):
        pass
        
