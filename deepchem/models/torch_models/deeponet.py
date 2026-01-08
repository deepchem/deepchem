import torch
import torch.nn as nn
from typing import Tuple, List, Iterable
from deepchem.data import Dataset
from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss


class DeepONet(nn.Module):
    """Deep Operator Network for learning nonlinear operators.

    DeepONet is a neural network architecture designed to learn mappings
    between infinite-dimensional function spaces. It consists of two
    sub-networks: a branch network that encodes the input function, and
    a trunk network that encodes the query locations. The final output
    is computed as the dot product of the branch and trunk outputs plus
    a learnable bias.

    Parameters
    ----------
    branch_input_dim : int
        Dimension of the input to the branch network (discretized input function).
    trunk_input_dim : int
        Dimension of the input to the trunk network (query coordinates).
    branch_hidden : Tuple[int, ...], default (64, 64)
        Hidden layer dimensions for the branch network.
    trunk_hidden : Tuple[int, ...], default (64, 64)
        Hidden layer dimensions for the trunk network.
    output_dim : int, default 64
        Output dimension of both branch and trunk networks before dot product.
    activation_fn : str, default 'tanh'
        Activation function for hidden layers.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.deeponet import DeepONet
    >>> model = DeepONet(branch_input_dim=10, trunk_input_dim=3)
    >>> branch_input = torch.randn(5, 10)
    >>> trunk_input = torch.randn(5, 3)
    >>> output = model([branch_input, trunk_input])
    >>> output.shape
    torch.Size([5, 1])

    References
    ----------
    .. [1] Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021).
        "Learning nonlinear operators via DeepONet based on the universal
        approximation theorem of operators."
        Nature Machine Intelligence, 3(3), 218-229.
        https://arxiv.org/abs/1910.03193

    """

    def __init__(self,
                 branch_input_dim: int,
                 trunk_input_dim: int,
                 branch_hidden: Tuple[int, ...] = (64, 64),
                 trunk_hidden: Tuple[int, ...] = (64, 64),
                 output_dim: int = 64,
                 activation_fn: str = 'tanh') -> None:
        """Initialize the DeepONet model."""
        super().__init__()
        self.output_dim = output_dim

        self.branch_net = MultilayerPerceptron(d_input=branch_input_dim,
                                               d_hidden=branch_hidden,
                                               d_output=output_dim,
                                               activation_fn=activation_fn)

        self.trunk_net = MultilayerPerceptron(d_input=trunk_input_dim,
                                              d_hidden=trunk_hidden,
                                              d_output=output_dim,
                                              activation_fn=activation_fn)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the DeepONet.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            A list containing two tensors:
            - inputs[0]: Branch input of shape (batch_size, branch_input_dim)
            - inputs[1]: Trunk input of shape (batch_size, trunk_input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1).

        """
        branch_input, trunk_input = inputs[0], inputs[1]
        branch_out = self.branch_net(branch_input)
        trunk_out = self.trunk_net(trunk_input)
        output = torch.sum(branch_out * trunk_out, dim=-1,
                           keepdim=True) + self.bias
        return output


class DeepONetModel(TorchModel):
    """DeepONet model wrapper for DeepChem.

    This class wraps the DeepONet base model and provides a DeepChem-compatible
    interface for training and prediction. It includes a custom default_generator
    that splits the input data into branch and trunk components.

    The input data X should be a concatenation of branch and trunk inputs:
    X = [branch_data | trunk_data] with shape (n_samples, branch_input_dim + trunk_input_dim)

    Parameters
    ----------
    branch_input_dim : int
        Dimension of the input to the branch network.
    trunk_input_dim : int
        Dimension of the input to the trunk network.
    branch_hidden : Tuple[int, ...], default (64, 64)
        Hidden layer dimensions for the branch network.
    trunk_hidden : Tuple[int, ...], default (64, 64)
        Hidden layer dimensions for the trunk network.
    output_dim : int, default 64
        Output dimension of both networks before dot product.
    activation_fn : str, default 'tanh'
        Activation function for hidden layers.
    **kwargs : dict
        Additional arguments passed to TorchModel constructor.

    Examples
    --------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models import DeepONetModel
    >>> branch_data = np.random.randn(100, 10).astype(np.float32)
    >>> trunk_data = np.random.randn(100, 3).astype(np.float32)
    >>> X = np.concatenate([branch_data, trunk_data], axis=1)
    >>> y = np.random.randn(100, 1).astype(np.float32)
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> model = DeepONetModel(branch_input_dim=10, trunk_input_dim=3, batch_size=32)
    >>> loss = model.fit(dataset, nb_epoch=10)

    References
    ----------
    .. [1] Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021).
        "Learning nonlinear operators via DeepONet based on the universal
        approximation theorem of operators."
        Nature Machine Intelligence, 3(3), 218-229.
        https://arxiv.org/abs/1910.03193

    """

    def __init__(self,
                 branch_input_dim: int,
                 trunk_input_dim: int,
                 branch_hidden: Tuple[int, ...] = (64, 64),
                 trunk_hidden: Tuple[int, ...] = (64, 64),
                 output_dim: int = 64,
                 activation_fn: str = 'tanh',
                 **kwargs) -> None:
        """Initialize the DeepONetModel."""
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        model = DeepONet(branch_input_dim=branch_input_dim,
                         trunk_input_dim=trunk_input_dim,
                         branch_hidden=branch_hidden,
                         trunk_hidden=trunk_hidden,
                         output_dim=output_dim,
                         activation_fn=activation_fn)
        super().__init__(model, loss=L2Loss(), **kwargs)

    def default_generator(
            self,
            dataset: Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        """Create a generator that splits input data into branch and trunk components.

        This generator overrides the default TorchModel generator to handle
        the two-input architecture of DeepONet. It expects X to be a concatenation
        of branch and trunk inputs.

        Parameters
        ----------
        dataset : Dataset
            The dataset to iterate over.
        epochs : int, default 1
            Number of epochs to iterate.
        mode : str, default 'fit'
            The mode of iteration ('fit', 'predict', or 'uncertainty').
        deterministic : bool, default True
            Whether to iterate in a deterministic order.
        pad_batches : bool, default True
            Whether to pad batches to the batch size.

        Yields
        ------
        Tuple[List, List, List]
            A tuple of ([branch_input, trunk_input], [labels], [weights]).

        """
        for epoch in range(epochs):
            for X, y, w, ids in dataset.iterbatches(batch_size=self.batch_size,
                                                    deterministic=deterministic,
                                                    pad_batches=pad_batches):
                branch_input = X[:, :self.branch_input_dim]
                trunk_input = X[:, self.branch_input_dim:]
                yield ([branch_input, trunk_input], [y], [w])
