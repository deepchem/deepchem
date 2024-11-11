from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import SigmoidCrossEntropy
import torch
import torch.nn as nn
from typing import Callable, Optional

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class IRVLayer(nn.Module):
    """
    Implements Influence Relevance Voter (IRV), a novel machine learning model designed for virtual
    high-throughput screening (vHTS).vHTS predicts the biological activity of chemical compounds
    sing computational methods, reducing the need for expensive experimental screening.

    The IRV model extends the k-Nearest Neighbors (kNN) algorithm by improving how neighbors
    influence predictions. Instead of treating all neighbors equally, IRV assigns each neighbor
    a relevance score based on its similarity to the query compound.This similarity is calculated
    using molecular fingerprint comparisons between the query compound and its neighbors,allowing
    more relevant neighbors to have a greater impact on the prediction.

    Example
    -------

    >>> import deepchem as dc
    >>> import numpy as np
    >>> n_tasks = 5
    >>> n_samples = 10
    >>> n_features = 128
    >>> K=5

    >>> # Generate dummy dataset.
    >>> ids = np.arange(n_samples)

    >>> # Features in ECFP Fingerprints representation
    >>> X = np.random.randint(2, size=(n_samples, n_features))

    >>> # Labels for each task in each column.
    >>> # Either 1 or 0 depending on whether the samples are active or inactive in that application(task)
    >>> y = np.ones((n_samples, n_tasks))

    >>> # Weights for each task in each column
    >>> w = np.ones((n_samples, n_tasks))

    >>> dataset = dc.data.NumpyDataset(X, y, w, ids)

    >>> # Transforms ECFP Fingerprints to IRV features(Similarity values of top K nearest neighbors).
    >>> IRV_transformer = dc.trans.IRVTransformer(K, n_tasks, dataset)
    >>> dataset_trans = IRV_transformer.transform(dataset)

    >>> # Instantiate the model
    >>> model = dc.models.torch_models.MultitaskIRVClassifier(n_tasks, K = 5, learning_rate = 0.01, batch_size = n_samples)
    >>> # Train the model
    >>> loss = model.fit(dataset_trans)

    >>> # Prediction and Evaluation
    >>> output = model.predict(dataset_trans)
    >>> classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score,task_averager=np.mean)
    >>> score = model.evaluate(dataset_trans, [classification_metric])

    References:
    -----------
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2750043/
    """

    def __init__(self, n_tasks: int, K, penalty: int):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        K: int
            Number of nearest neighbours used in classification
        penalty: float
            Amount of penalty (l2 or l1 applied)
        """
        super(IRVLayer, self).__init__()
        self.n_tasks = n_tasks
        self.K = K
        self.penalty = penalty

        # Initialize weights and biases
        self.V = nn.Parameter(torch.tensor([0.01, 1.], dtype=torch.float32))
        self.W = nn.Parameter(torch.tensor([1., 1.], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))
        self.b2 = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))

    def forward(self, inputs):
        K = self.K
        predictions = []
        for count in range(self.n_tasks):

            # Similarity values
            similarity = inputs[:, 2 * K * count:(2 * K * count + K)]

            # Labels for all top K similar samples
            ys = inputs[:, (2 * K * count + K):2 * K * (count + 1)].int()

            # Relevance
            R = self.b + self.W[0] * similarity + self.W[1] * torch.arange(
                1, K + 1, dtype=torch.float32, device=device)
            R = torch.sigmoid(R)

            # Influence = Relevance * Vote
            z = torch.sum(R * self.V[ys], dim=1) + self.b2
            predictions.append(z.view(-1, 1))

        predictions = torch.cat(predictions, dim=1)

        logits = []
        outputs = []
        for task in range(self.n_tasks):
            task_output = Slice(task, 1)(predictions)
            sigmoid = torch.sigmoid(task_output)
            logits.append(task_output)
            outputs.append(sigmoid)
        outputs = torch.stack(outputs, dim=1)
        outputs2 = 1 - outputs
        outputs = [
            torch.cat([outputs2, outputs], dim=2),
            logits[0] if len(logits) == 1 else torch.cat(logits, dim=1)
        ]
        return outputs


class Slice(nn.Module):
    """ Choose a slice of input on the last axis given order,
    Suppose input x has two dimensions,
    output f(x) = x[:, slice_num:slice_num+1]
    """

    def __init__(self, slice_num: int, axis=1):
        """
        Parameters
        ----------
        slice_num: int
            index of slice number
        axis: int
            axis id
        """
        super(Slice, self).__init__()
        self.slice_num = slice_num
        self.axis = axis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        slice_num = self.slice_num
        axis = self.axis
        return inputs.index_select(axis, torch.tensor([slice_num],
                                                      device=device))


class MultitaskIRVClassifier(TorchModel):

    def __init__(self,
                 n_tasks: int,
                 K=10,
                 penalty=0.0,
                 mode="classification",
                 **kwargs):
        """Initialize MultitaskIRVClassifier

        Parameters
        ----------
        n_tasks: int
            Number of tasks
        K: int
            Number of nearest neighbours used in classification
        penalty: float
            Amount of penalty (l2 or l1 applied)
        """
        self.n_tasks = n_tasks
        self.K = K
        self.n_features = 2 * self.K * self.n_tasks
        self.penalty = penalty

        # Define the IRVLayer
        self.model = IRVLayer(self.n_tasks, self.K, self.penalty).to(device)

        regularization_loss: Optional[Callable]
        if self.penalty != 0.0:
            weights = list(self.model.parameters())
            regularization_loss = lambda: self.penalty * torch.sum(  # noqa: E731
                torch.stack([torch.square(w).sum() for w in weights]))
        else:
            regularization_loss = None

        super(MultitaskIRVClassifier,
              self).__init__(self.model,
                             loss=SigmoidCrossEntropy(),
                             output_types=['prediction', 'loss'],
                             regularization_loss=regularization_loss,
                             **kwargs)
