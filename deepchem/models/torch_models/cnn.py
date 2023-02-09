import torch
import numpy as np
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.torch_models.layers import CNNModule
from deepchem.models.losses import L2Loss
from deepchem.metrics import to_one_hot

from typing import List, Union, Callable, Optional
from deepchem.utils.typing import OneOrMany, ActivationFn, LossFn


class CNN(TorchModel):
    """A 1, 2, or 3 dimensional convolutional network for either regression or classification.

    The network consists of the following sequence of layers:

    - A configurable number of convolutional layers
    - A global pooling layer (either max pool or average pool)
    - A final fully connected layer to compute the output

    It optionally can compose the model from pre-activation residual blocks, as
    described in https://arxiv.org/abs/1603.05027, rather than a simple stack of
    convolution layers.  This often leads to easier training, especially when using a
    large number of layers.  Note that residual blocks can only be used when
    successive layers have the same output shape.  Wherever the output shape changes, a
    simple convolution layer will be used even if residual=True.

    Examples
    --------
    >>> n_samples = 10
    >>> n_features = 3
    >>> n_tasks = 1
    >>> np.random.seed(123)
    >>> X = np.random.rand(n_samples, 10, n_features)
    >>> y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
    >>> dataset: dc.data.Dataset = dc.data.NumpyDataset(X, y)
    >>> regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
    >>> model = CNN(n_tasks, n_features, dims=1, kernel_size=3, mode='regression')
    >>> avg_loss = model.fit(dataset, nb_epoch=10)

    """

    def __init__(self,
                 n_tasks: int,
                 n_features: int,
                 dims: int,
                 layer_filters: List[int] = [100],
                 kernel_size: OneOrMany[int] = 5,
                 strides: OneOrMany[int] = 1,
                 weight_init_stddevs: OneOrMany[float] = 0.02,
                 bias_init_consts: OneOrMany[float] = 1.0,
                 weight_decay_penalty: float = 0.0,
                 weight_decay_penalty_type: str = 'l2',
                 dropouts: OneOrMany[float] = 0.5,
                 activation_fns: OneOrMany[ActivationFn] = 'relu',
                 pool_type: str = 'max',
                 mode: str = 'classification',
                 n_classes: int = 2,
                 uncertainty: bool = False,
                 residual: bool = False,
                 padding: Union[int, str] = 'valid',
                 **kwargs) -> None:
        """TorchModel wrapper for CNN

        Parameters
        ----------
        n_tasks: int
            number of tasks
        n_features: int
            number of features
        dims: int
            the number of dimensions to apply convolutions over (1, 2, or 3)
        layer_filters: list
            the number of output filters for each convolutional layer in the network.
            The length of this list determines the number of layers.
        kernel_size: int, tuple, or list
            a list giving the shape of the convolutional kernel for each layer.  Each
            element may be either an int (use the same kernel width for every dimension)
            or a tuple (the kernel width along each dimension).  Alternatively this may
            be a single int or tuple instead of a list, in which case the same kernel
            shape is used for every layer.
        strides: int, tuple, or list
            a list giving the stride between applications of the  kernel for each layer.
            Each element may be either an int (use the same stride for every dimension)
            or a tuple (the stride along each dimension).  Alternatively this may be a
            single int or tuple instead of a list, in which case the same stride is
            used for every layer.
        weight_init_stddevs: list or float
            the standard deviation of the distribution to use for weight initialization
            of each layer.  The length of this list should equal len(layer_filters)+1,
            where the final element corresponds to the dense layer.  Alternatively this
            may be a single value instead of a list, in which case the same value is used
            for every layer.
        bias_init_consts: list or float
            the value to initialize the biases in each layer to.  The length of this
            list should equal len(layer_filters)+1, where the final element corresponds
            to the dense layer.  Alternatively this may be a single value instead of a
            list, in which case the same value is used for every layer.
        weight_decay_penalty: float
            the magnitude of the weight decay penalty to use
        weight_decay_penalty_type: str
            the type of penalty to use for weight decay, either 'l1' or 'l2'
        dropouts: list or float
            the dropout probability to use for each layer.  The length of this list should equal len(layer_filters).
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer
        activation_fns: str or list
            the torch activation function to apply to each layer. The length of this list should equal
            len(layer_filters).  Alternatively this may be a single value instead of a list, in which case the
            same value is used for every layer, 'relu' by default
        pool_type: str
            the type of pooling layer to use, either 'max' or 'average'
        mode: str
            Either 'classification' or 'regression'
        n_classes: int
            the number of classes to predict (only used in classification mode)
        uncertainty: bool
            if True, include extra outputs and loss terms to enable the uncertainty
            in outputs to be predicted
        residual: bool
            if True, the model will be composed of pre-activation residual blocks instead
            of a simple stack of convolutional layers.
        padding: str, int or tuple
            the padding to use for convolutional layers, either 'valid' or 'same'
        """
        self.mode = mode
        self.n_classes = n_classes
        self.n_tasks = n_tasks

        self.model = CNNModule(n_tasks=n_tasks,
                               n_features=n_features,
                               dims=dims,
                               layer_filters=layer_filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               weight_init_stddevs=weight_init_stddevs,
                               bias_init_consts=bias_init_consts,
                               dropouts=dropouts,
                               activation_fns=activation_fns,
                               pool_type=pool_type,
                               mode=mode,
                               n_classes=n_classes,
                               uncertainty=uncertainty,
                               residual=residual,
                               padding=padding)

        regularization_loss: Optional[Callable]

        if weight_decay_penalty != 0:
            weights = [layer.weight for layer in self.model.layers]
            if weight_decay_penalty_type == 'l1':
                regularization_loss = lambda: weight_decay_penalty * torch.sum(  # noqa: E731
                    torch.stack([torch.abs(w).sum() for w in weights]))
            else:
                regularization_loss = lambda: weight_decay_penalty * torch.sum(  # noqa: E731
                    torch.stack([torch.square(w).sum() for w in weights]))
        else:
            regularization_loss = None

        loss: Union[L2Loss, LossFn]

        if uncertainty:

            def loss(outputs, labels, weights):
                diff = labels[0] - outputs[0]
                return torch.mean(diff**2 / torch.exp(outputs[1]) + outputs[1])

        else:
            loss = L2Loss()

        if self.mode == 'classification':
            output_types = ['prediction', 'loss']
        else:
            if uncertainty:
                output_types = ['prediction', 'variance', 'loss', 'loss']
            else:
                output_types = ["prediction"]

        super(CNN, self).__init__(self.model,
                                  loss=loss,
                                  output_types=output_types,
                                  regularization_loss=regularization_loss,
                                  **kwargs)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):

        for epoch in range(epochs):

            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):

                if self.mode == 'classification':
                    if y_b is not None:
                        y_b = to_one_hot(y_b.flatten(), self.n_classes)\
                            .reshape(-1, self.n_tasks, self.n_classes)

                dropout = np.array(0.) if mode == 'predict' else np.array(1.)

                yield ([X_b, dropout], [y_b], [w_b])
