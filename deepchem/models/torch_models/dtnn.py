from typing import List
import torch.nn as nn
import torch
from deepchem.models.torch_models import layers
from deepchem.models.torch_models import TorchModel
from deepchem.data.datasets import Dataset
from deepchem.utils import batch_coulomb_matrix_features
from deepchem.models.optimizers import Optimizer
from deepchem.models.losses import Loss
from deepchem.models.optimizers import (AdaGrad, Adam, SparseAdam, AdamW,
                                        RMSProp, ExponentialDecay,
                                        LambdaLRWithWarmup, PolynomialDecay,
                                        LinearCosineDecay,
                                        PiecewiseConstantSchedule, KFAC, Lamb)
from deepchem.models.losses import L1Loss, L2Loss, HuberLoss, HingeLoss, SquaredHingeLoss

optimizers_map = {
    "AdaGrad": AdaGrad,
    "Adam": Adam,
    "SparseAdam": SparseAdam,
    "AdamW": AdamW,
    "RMSProp": RMSProp,
    "ExponentialDecay": ExponentialDecay,
    "LambdaLRWithWarmup": LambdaLRWithWarmup,
    "PolynomialDecay": PolynomialDecay,
    "LinearCosineDecay": LinearCosineDecay,
    "PiecewiseConstantSchedule": PiecewiseConstantSchedule,
    "KFAC": KFAC,
    "Lamb": Lamb
}

losses_map = {
    "L1Loss": L1Loss,
    "L2Loss": L2Loss,
    "HuberLoss": HuberLoss,
    "HingeLoss": HingeLoss,
    "SquaredHingeLoss": SquaredHingeLoss
}


class DTNN(nn.Module):
    """Deep Tensor Neural Networks

    DTNN is based on the many-body Hamiltonian concept, which is a fundamental principle in quantum mechanics.
    The DTNN recieves a molecule's distance matrix and membership of its atom from its Coulomb Matrix representation.
    Then, it iteratively refines the representation of each atom by considering its interactions with neighboring atoms.
    Finally, it predicts the energy of the molecule by summing up the energies of the individual atoms.

    In this class, we establish a sequential model for the Deep Tensor Neural Network (DTNN) [1]_.

    Examples
    --------
    >>> import os
    >>> import torch
    >>> from deepchem.models.torch_models import DTNN
    >>> from deepchem.data import SDFLoader
    >>> from deepchem.feat import CoulombMatrix
    >>> from deepchem.utils import batch_coulomb_matrix_features
    >>> # Get Data
    >>> model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    >>> dataset_file = os.path.join(model_dir, 'tests/assets/qm9_mini.sdf')
    >>> TASKS = ["alpha", "homo"]
    >>> loader = SDFLoader(tasks=TASKS, featurizer=CoulombMatrix(29), sanitize=True)
    >>> data = loader.create_dataset(dataset_file, shard_size=100)
    >>> inputs = batch_coulomb_matrix_features(data.X)
    >>> atom_number, distance, atom_membership, distance_membership_i, distance_membership_j = inputs
    >>> inputs = [torch.tensor(atom_number).to(torch.int64),
    ...           torch.tensor(distance).to(torch.float32),
    ...           torch.tensor(atom_membership).to(torch.int64),
    ...           torch.tensor(distance_membership_i).to(torch.int64),
    ...           torch.tensor(distance_membership_j).to(torch.int64)]
    >>> n_tasks = data.y.shape[0]
    >>> model = DTNN(n_tasks)
    >>> pred = model(inputs)

    References
    ----------
    .. [1] Schütt, Kristof T., et al. "Quantum-chemical insights from deep
        tensor neural networks." Nature communications 8.1 (2017): 1-8.

    """

    def __init__(self,
                 n_tasks: int,
                 n_embedding: int = 30,
                 n_hidden: int = 100,
                 n_distance: int = 100,
                 distance_min: float = -1,
                 distance_max: float = 18,
                 output_activation: bool = True,
                 mode: str = "regression",
                 dropout: float = 0.0,
                 n_steps: int = 2):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_embedding: int (default 30)
            Number of features per atom.
        n_hidden: int (default 100)
            Number of features for each molecule after DTNNStep
        n_distance: int (default 100)
            granularity of distance matrix
            step size will be (distance_max-distance_min)/n_distance
        distance_min: float (default -1)
            minimum distance of atom pairs (in Angstrom)
        distance_max: float (default 18)
            maximum distance of atom pairs (in Angstrom)
        output_activation: bool (default True)
            determines whether an activation function should be apply  to its output.
        mode: str (default "regression")
            Only "regression" is currently supported.
        dropout: float (default 0.0)
            the dropout probablity to use.
        n_steps: int (default 2)
            Number of DTNNStep Layers to use.

        """
        super(DTNN, self).__init__()
        self.n_tasks = n_tasks
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_distance = n_distance
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.output_activation = output_activation
        self.mode = mode
        self.dropout = dropout
        self.n_steps = n_steps

        # get DTNNEmbedding
        self.dtnn_embedding = layers.DTNNEmbedding(n_embedding=self.n_embedding)

        # get DTNNSteps
        self.dtnn_step = nn.ModuleList()
        for i in range(self.n_steps):
            self.dtnn_step.append(
                layers.DTNNStep(n_embedding=self.n_embedding,
                                n_distance=self.n_distance))

        # get DTNNGather
        self.dtnn_gather = layers.DTNNGather(
            n_embedding=self.n_embedding,
            layer_sizes=[self.n_hidden],
            n_outputs=self.n_tasks,
            output_activation=self.output_activation)

        # get Final Linear Layer
        self.linear = nn.LazyLinear(self.n_tasks)

    def forward(self, inputs: List[torch.Tensor]):
        """
        Parameters
        ----------
        inputs: List
            A list of tensors containing atom_number, distance,
            atom_membership, distance_membership_i, and distance_membership_j.

        Returns
        -------
        output: torch.Tensor
            Predictions of the Molecular Energy.

        """
        dtnn_embedding = self.dtnn_embedding(inputs[0])

        for i in range(self.n_steps):
            dtnn_embedding = nn.Dropout(self.dropout)(dtnn_embedding)
            dtnn_embedding = self.dtnn_step[i](
                [dtnn_embedding, inputs[1], inputs[3], inputs[4]])

        dtnn_step = nn.Dropout(self.dropout)(dtnn_embedding)
        dtnn_gather = self.dtnn_gather([dtnn_step, inputs[2]])

        dtnn_gather = nn.Dropout(self.dropout)(dtnn_gather)
        output = self.linear(dtnn_gather)
        return output


class DTNNModel(TorchModel):
    """Implements DTNN models for regression.

    DTNN is based on the many-body Hamiltonian concept, which is a fundamental principle in quantum mechanics.
    DTNN recieves a molecule's distance matrix and membership of its atom from its Coulomb Matrix representation.
    Then, it iteratively refines the representation of each atom by considering its interactions with neighboring atoms.
    Finally, it predicts the energy of the molecule by summing up the energies of the individual atoms.

    This class implements the Deep Tensor Neural Network (DTNN) [1]_.

    Examples
    --------
    >>> import os
    >>> from deepchem.data import SDFLoader
    >>> from deepchem.feat import CoulombMatrix
    >>> from deepchem.models.torch_models import DTNNModel
    >>> model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    >>> dataset_file = os.path.join(model_dir, 'tests/assets/qm9_mini.sdf')
    >>> TASKS = ["alpha", "homo"]
    >>> loader = SDFLoader(tasks=TASKS, featurizer=CoulombMatrix(29), sanitize=True)
    >>> data = loader.create_dataset(dataset_file, shard_size=100)
    >>> n_tasks = data.y.shape[1]
    >>> model = DTNNModel(n_tasks,
    ...                   n_embedding=20,
    ...                   n_distance=100,
    ...                   learning_rate=1.0,
    ...                   mode="regression")
    >>> loss = model.fit(data, nb_epoch=250)
    >>> pred = model.predict(data)

    References
    ----------
    .. [1] Schütt, Kristof T., et al. "Quantum-chemical insights from deep
        tensor neural networks." Nature communications 8.1 (2017): 1-8.

    """

    def __init__(self,
                 n_tasks: int,
                 n_embedding: int = 30,
                 n_hidden: int = 100,
                 n_distance: int = 100,
                 distance_min: float = -1,
                 distance_max: float = 18,
                 output_activation: bool = True,
                 mode: str = "regression",
                 dropout: float = 0.0,
                 n_steps: int = 2,
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_embedding: int (default 30)
            Number of features per atom.
        n_hidden: int (default 100)
            Number of features for each molecule after DTNNStep
        n_distance: int (default 100)
            granularity of distance matrix
            step size will be (distance_max-distance_min)/n_distance
        distance_min: float (default -1)
            minimum distance of atom pairs (in Angstrom)
        distance_max: float (default = 18)
            maximum distance of atom pairs (in Angstrom)
        output_activation: bool (default True)
            determines whether an activation function should be apply  to its output.
        mode: str (default "regression")
            Only "regression" is currently supported.
        dropout: float (default 0.0)
            the dropout probablity to use.
        n_steps: int (default 2)
            Number of DTNNStep Layers to use.

        """
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(dropout))
        model = DTNN(n_tasks=n_tasks,
                     n_embedding=n_embedding,
                     n_hidden=n_hidden,
                     n_distance=n_distance,
                     distance_min=distance_min,
                     distance_max=distance_max,
                     output_activation=output_activation,
                     mode=mode,
                     dropout=dropout,
                     n_steps=n_steps)
        if mode not in ['regression']:
            raise ValueError("Only 'regression' mode is currently supported")
        super(DTNNModel, self).__init__(model, L2Loss(), ["prediction"],
                                        **kwargs)
        self.name = "DTNN"
        self.param_dict = {
            "name": "DTNN",
            "n_tasks": n_tasks,
            "n_embedding": n_embedding,
            "n_hidden": n_hidden,
            "n_distance": n_distance,
            "distance_min": distance_min,
            "distance_max": distance_max,
            "output_activation": output_activation,
            "mode": mode,
            "dropout": dropout,
            "n_steps": n_steps
        }
        self.dict_kwargs = kwargs
        self.param_dict.update(self.dict_kwargs)

    def default_generator(self,
                          dataset: Dataset,
                          epochs: int = 1,
                          mode: str = 'fit',
                          deterministic: bool = True,
                          pad_batches: bool = True):
        """Create a generator that iterates batches for a dataset.
        It processes inputs through the _compute_features_on_batch function to calculate required features of input.

        Parameters
        ----------
        dataset: Dataset
            the data to iterate
        epochs: int
            the number of times to iterate over the full dataset
        mode: str
            allowed values are 'fit' (called during training), 'predict' (called
            during prediction), and 'uncertainty' (called during uncertainty
            prediction)
        deterministic: bool
            whether to iterate over the dataset in order, or randomly shuffle the
            data for each epoch
        pad_batches: bool
            whether to pad each batch up to this model's preferred batch size

        Returns
        -------
        a generator that iterates batches, each represented as a tuple of lists:
        ([inputs], [outputs], [weights])

        """
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                yield (batch_coulomb_matrix_features(X_b,
                                                     self.model.distance_max,
                                                     self.model.distance_min,
                                                     self.model.n_distance),
                       [y_b], [w_b])

    @staticmethod
    def serialize(obj):
        """
        Serializes a supported object into a JSON-serializable dictionary.

        Supported types:
        - Basic types: str, int, float, bool, list, None
        - torch.device
        - torch.optim.Optimizer subclasses
        - torch.nn.Loss subclasses

        Parameters:
            obj (Any): The object to serialize.

        Returns:
            dict: A JSON-serializable dictionary representation of the object.

        Raises:
            TypeError: If the object type is unsupported.
        """
        if isinstance(obj, (str, int, list, float, bool, type(None))):
            return {"__type__": type(obj).__name__, "__value__": obj}
        if isinstance(obj, torch.device):
            return {"__type__": "torch_device", "__value__": str(obj)}
        if isinstance(obj, Optimizer):
            return {
                "__type__": "deepchem_optimizer",
                "__name__": obj.__class__.__name__,
                "params": obj.__dict__
            }
        if isinstance(obj, Loss):
            return {
                "__type__": "deepchem_loss",
                "__name__": obj.__class__.__name__,
                "params": obj.__dict__
            }
        raise TypeError(
            f"Object of type {type(obj).__name__} is not JSON serializable")

    @staticmethod
    def deserialize(obj):
        """
        Deserializes an object from a dictionary previously created by `serialize()`.

        Parameters:
            obj (dict): A dictionary representation of the serialized object.

        Returns:
            Any: The deserialized object.

        Raises:
            ValueError: If the dictionary contains an unsupported object type.
        """
        obj_type = obj["__type__"]
        normal_types = {"int", "list", "str", "float", "bool", "NoneType"}

        if obj_type in normal_types:
            return obj["__value__"]

        if obj_type == "torch_device":
            return torch.device(obj["__value__"])

        if obj_type == "deepchem_optimizer":
            optimizer_class = optimizers_map[obj["__name__"]]
            return optimizer_class(**obj["params"])

        if obj_type == "deepchem_loss":
            loss_class = losses_map[obj["__name__"]]
            return loss_class(**obj["params"])

        raise ValueError(f"Unsupported type: {obj_type}")
