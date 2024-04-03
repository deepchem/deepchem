from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import HingeLoss
import torch.nn as nn
import torch.nn.functional as F


class ScScore(nn.Module):
    """
    Builder class for ScScore Model.

    The SCScore model is a neural network model based on the work of Coley et al. [1]_ that predicts the synthetic complexity score (SCScore) of molecules and correlates it with the expected number of reaction steps required to produce the given target molecule.
    It is trained on a dataset of over 12 million reactions from the Reaxys database to impose a pairwise inequality constraint enforcing that on average the products of published chemical reactions should be more synthetically complex than their corresponding reactants.
    The learned metric (SCScore) exhibits highly desirable nonlinear behavior, particularly in recognizing increases in synthetic complexity throughout a number of linear synthetic routes.
    The SCScore model can accurately predict the synthetic complexity of a variety of molecules, including both drug-like and natural product molecules.
    SCScore has the potential to be a valuable tool for chemists who are working on drug discovery and other areas of chemistry.

    The learned metric (SCScore) exhibits highly desirable nonlinear behavior, particularly in recognizing increases in synthetic complexity throughout a number of linear synthetic routes.

    Our model uses hingeloss instead of the shifted relu loss as in the supplementary material [2]_ provided by the author.
    This could cause differentiation issues with compounds that are "close" to each other in "complexity".

    The default values for the model are the same as the ones used in the original paper [1]_.

    Examples
    --------
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models import ScScoreModel
    >>> # preparing dataset
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> labels = [0., 1.]
    >>> featurizer = dc.feat.CircularFingerprint(size=1024, radius=2, chiral=True)
    >>> X = featurizer.featurize(smiles)
    >>> dataset = dc.data.NumpyDataset(X=X, y=labels)
    >>> # training model
    >>> model = ScScoreModel(n_features=1024, layer_sizes=[300, 300, 300, 300, 300],
    ...                      dropout=0.2, score_scale=5)
    >>> loss = model.fit(dataset, nb_epoch=5)

    References
    ----------
    .. [1] Coley, C. W., Rogers, L., Green, W., & Jensen, K. F. (2018). "SCScore: Synthetic Complexity Learned from a Reaction Corpus". Journal of Chemical Information and Modeling, 58(2), 252-261. https://doi.org/10.1021/acs.jcim.7b00622

    .. [2] Coley, C. W., Rogers, L., Green, W., & Jensen, K. F. (2018). Supplementary material to "SCScore: Synthetic Complexity Learned from a Reaction Corpus". Journal of Chemical Information and Modeling, 58(2), 252-261. https://github.com/connorcoley/scscore
    """

    def __init__(self,
                 n_features: int = 1024,
                 layer_sizes: list = [300, 300, 300, 300, 300],
                 dropout: float = 0.0,
                 score_scale: int = 5):
        """
        Parameters
        ----------
        n_features: int (default 1024)
            Number of features per molecule.
        layer_sizes: list of int (default [300, 300, 300, 300, 300])
            Size of each hidden layer, length of the list is the number of hidden layers.
        dropouts: float (default 0.0)
            Droupout to apply to each hidden layer.
        score_scale: int (default 5)
            Scale of the output score.
        """
        super(ScScore, self).__init__()
        self.n_features = n_features
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.score_scale = score_scale

        input_size = self.layer_sizes[0]
        self.input_layer = nn.Linear(self.n_features, input_size)

        self.hidden_layers = nn.ModuleList()
        for layer_size in self.layer_sizes[1:]:
            self.hidden_layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size

        self.output_layer = nn.Linear(self.layer_sizes[-1], 1)

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs: list of Tensor
            data inputs for the model

        Returns
        -------
        output: Tensor
            Synthetic complexity score of the input molecule.
        """

        x = F.relu(self.input_layer(inputs))
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout)

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout)

        output = F.sigmoid(self.output_layer(x))

        output = 1 + (self.score_scale - 1) * output
        return output


class ScScoreModel(TorchModel):
    """
    The SCScore model is a neural network model based on the work of Coley et al. [1]_ that predicts the synthetic complexity score (SCScore) of molecules and correlates it with the expected number of reaction steps required to produce the given target molecule.
    It is trained on a dataset of over 12 million reactions from the Reaxys database to impose a pairwise inequality constraint enforcing that on average the products of published chemical reactions should be more synthetically complex than their corresponding reactants.
    The learned metric (SCScore) exhibits highly desirable nonlinear behavior, particularly in recognizing increases in synthetic complexity throughout a number of linear synthetic routes.
    The SCScore model can accurately predict the synthetic complexity of a variety of molecules, including both drug-like and natural product molecules.
    SCScore has the potential to be a valuable tool for chemists who are working on drug discovery and other areas of chemistry.

    The learned metric (SCScore) exhibits highly desirable nonlinear behavior, particularly in recognizing increases in synthetic complexity throughout a number of linear synthetic routes.

    Our model uses hingeloss instead of the shifted relu loss as in the supplementary material [2]_ provided by the author.
    This could cause differentiation issues with compounds that are "close" to each other in "complexity".

    The default values for the model are the same as the ones used in the original paper [1]_.

    Examples
    --------
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models import ScScoreModel
    >>> # preparing dataset
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> labels = [0., 1.]
    >>> featurizer = dc.feat.CircularFingerprint(size=1024, radius=2, chiral=True)
    >>> X = featurizer.featurize(smiles)
    >>> dataset = dc.data.NumpyDataset(X=X, y=labels)
    >>> # training model
    >>> model = ScScoreModel(n_features=1024, layer_sizes=[300, 300, 300, 300, 300],
    ...                      dropout=0.2, score_scale=5)
    >>> loss = model.fit(dataset, nb_epoch=5)

    References
    ----------
    .. [1] Coley, C. W., Rogers, L., Green, W., & Jensen, K. F. (2018). "SCScore: Synthetic Complexity Learned from a Reaction Corpus". Journal of Chemical Information and Modeling, 58(2), 252-261. https://doi.org/10.1021/acs.jcim.7b00622

    .. [2] Coley, C. W., Rogers, L., Green, W., & Jensen, K. F. (2018). Supplementary material to "SCScore: Synthetic Complexity Learned from a Reaction Corpus". Journal of Chemical Information and Modeling, 58(2), 252-261. https://github.com/connorcoley/scscore
    """

    def __init__(self,
                 n_features: int = 1024,
                 layer_sizes: list = [300, 300, 300, 300, 300],
                 dropout: float = 0.0,
                 score_scale: int = 5,
                 **kwargs):
        """
        Parameters
        ----------
        n_features: int (default 1024)
            Number of features per molecule.
        layer_sizes: list of int (default [300, 300, 300, 300, 300])
            Size of each hidden layer, length of the list is the number of hidden layers.
        dropouts: float (default 0.0)
            Droupout to apply to each hidden layer.
        score_scale: int (default 5)
            Max output score, range of output = (1, score_scale).
        """

        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(
                f"Dropout must be between 0.0 and 1.0, but got {dropout}")

        if score_scale < 1:
            raise ValueError(
                f"Score scale must be greater than 1, but got {score_scale}")

        model = ScScore(n_features=n_features,
                        layer_sizes=layer_sizes,
                        dropout=dropout,
                        score_scale=score_scale)

        if 'loss' not in kwargs:
            kwargs['loss'] = HingeLoss()

        super(ScScoreModel, self).__init__(model, **kwargs)
