from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy, Loss
import deepchem as dc
import numpy as np
from deepchem.metrics import to_one_hot
from typing import List, Tuple, Optional, Any, Iterable, Union
from deepchem.utils.typing import LossFn
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from deepchem.models.losses import _make_pytorch_shapes_consistent
    from deepchem.models.torch_models import DAGLayer, DAGGather
    from deepchem.models.torch_models.torch_model import TorchModel
except (ModuleNotFoundError, ImportError):
    pass


class _DAG(nn.Module):
    """
    Directed Acyclic Graph models for molecular property prediction.

    Examples
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> from deepchem.models.torch_models.dag import _DAG, DAGModel
    >>> from deepchem.molnet import load_bace_classification
    >>> from deepchem.data import NumpyDataset
    >>> from deepchem.trans import DAGTransformer
    >>> n_tasks = 1
    >>> n_features = 75
    >>> n_classes = 2
    >>> n_samples = 10
    >>> tasks, all_dataset, transformers = load_bace_classification("GraphConv", reload=False)
    >>> train_dataset, valid_dataset, test_dataset = all_dataset
    >>> dataset = NumpyDataset(train_dataset.X[:n_samples], train_dataset.y[:n_samples], train_dataset.w[:n_samples], train_dataset.ids[:n_samples])
    >>> max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    >>> transformer = DAGTransformer(max_atoms=max_atoms)
    >>> dataset = transformer.transform(dataset)
    >>> base_mod = DAGModel(n_tasks=n_tasks, max_atoms=max_atoms, mode='classification', n_classes=n_classes, batch_size=2, device='cpu')
    >>> X,_,_ = base_mod._prepare_batch(next(base_mod.default_generator(dataset)))
    >>> model = _DAG(n_tasks=n_tasks, max_atoms=max_atoms, mode='classification', n_classes=n_classes, device='cpu', batch_size=2)
    >>> # forward pass
    >>> _ = model(X)

    References
    ----------
    .. [1] Lusci Alessandro, Gianluca Pollastri, and Pierre Baldi."Deep architectures and deep learning in chemoinformatics: the prediction of aqueous solubility for drug-like molecules."Journal of chemoinformatics and modeling 53.7 (2013):1563-1575.https://pmc.ncbi.nlm.nih.gov/articles/PMC3739985
    """

    def __init__(self,
                 n_tasks: int,
                 max_atoms: int = 50,
                 n_atom_feat: int = 75,
                 n_graph_feat: int = 30,
                 n_outputs: int = 30,
                 layer_sizes: List[int] = [100],
                 layer_sizes_gather: List[int] = [100],
                 dropout: Optional[float] = None,
                 mode: str = "classification",
                 n_classes: int = 2,
                 uncertainty: Optional[bool] = False,
                 batch_size: int = 100,
                 device: Optional[torch.device] = None,
                 **kwargs: Any) -> None:
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        max_atoms: int, optional
            Maximum number of atoms in a molecule, should be defined based on dataset.
        n_atom_feat: int, optional
            Number of features per atom.
        n_graph_feat: int, optional
            Number of features for atom in the graph.
        n_outputs: int, optional
            Number of features for each molecule.
        layer_sizes: list of int, optional
            List of hidden layer size(s) in the propagation step.
        layer_sizes_gather: list of int, optional
            List of hidden layer size(s) in the gather step.
        dropout: None or float, optional
            Dropout probability.
        mode: str, optional
            Either "classification" or "regression" for type of model.
        n_classes: int
            the number of classes to predict (only used in classification mode)
        uncertainty: bool
            if True, include extra outputs to enable uncertainty prediction
        batch_size: int, optional
            the batch size to use during training
        device: str, optional
            the device to run the model on
        """
        super(_DAG, self).__init__()

        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        if uncertainty and mode != "regression":
            raise ValueError("Uncertainty is only supported in regression mode")
        if uncertainty and (dropout is None or dropout == 0.0):
            raise ValueError('Dropout must be included to predict uncertainty')

        self.n_tasks = n_tasks
        self.max_atoms = max_atoms
        self.n_atom_feat = n_atom_feat
        self.n_graph_feat = n_graph_feat
        self.n_outputs = n_outputs
        self.mode = mode
        self.n_classes = n_classes
        self.uncertainty = uncertainty

        # DAG layers
        self.dag_layer = DAGLayer(n_graph_feat=self.n_graph_feat,
                                  n_atom_feat=self.n_atom_feat,
                                  max_atoms=self.max_atoms,
                                  layer_sizes=layer_sizes,
                                  dropout=dropout,
                                  batch_size=batch_size,
                                  device=device)

        # Gather layer
        self.dag_gather = DAGGather(n_graph_feat=self.n_graph_feat,
                                    n_outputs=self.n_outputs,
                                    max_atoms=self.max_atoms,
                                    layer_sizes=layer_sizes_gather,
                                    dropout=dropout,
                                    device=device)

        # Output layers
        if self.mode == 'classification':
            self.dense = nn.Linear(n_outputs, n_tasks * n_classes)
        else:
            self.dense = nn.Linear(n_outputs, n_tasks)
            if uncertainty:
                self.log_var = nn.Linear(n_outputs, n_tasks)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through the model.

        Parameters
        ----------
        inputs: list of torch.Tensor
            Contains the following tensors:
            - atoms_all: torch.Tensor of shape (n_atoms, n_atom_feat)
                Contains the atom features.
            - parents_all: torch.Tensor of shape (n_atoms, max_atoms, 2)
                Contains the parent indices for each atom.
            - calculation_orders: torch.Tensor of shape (n_atoms, max_atoms)
                Contains the order of calculation for each atom.
            - calculation_masks: torch.Tensor of shape (n_atoms, max_atoms)
                Contains the mask for the calculation order.
            - membership: torch.Tensor of shape (n_atoms)
                Contains the membership of each atom to a molecule.
            - n_atoms
                Contains the number of atoms in the molecule.
        """
        atoms_all, parents_all, calculation_orders, calculation_masks, membership, n_atoms = inputs
        # Propagate information through the graph
        daglayer = self.dag_layer([
            atoms_all, parents_all, calculation_orders, calculation_masks,
            n_atoms
        ])
        membership = membership.long()
        # Gather information from the graph
        dagather = self.dag_gather([daglayer, membership])

        # Output layer
        output = self.dense(dagather)

        if self.mode == 'classification':
            logits = output.view(-1, self.n_tasks, self.n_classes)
            output = F.softmax(logits, dim=-1)
            return [output, logits]
        else:
            if self.uncertainty:
                log_var = self.log_var(dagather)
                var = torch.exp(log_var)
                return [output, var, output, log_var]
            else:
                return [output]


class DAGModel(TorchModel):
    """
    Directed Acyclic Graph models for molecular property prediction.

    The basic idea for this paper is that a molecule is usually
    viewed as an undirected graph. However, you can convert it to
    a series of directed graphs. The idea is that for each atom,
    you make a DAG using that atom as the vertex of the DAG and
    edges pointing "inwards" to it. This transformation is
    implemented in
    `dc.trans.transformers.DAGTransformer.UG_to_DAG`.

    This model accepts ConvMols as input, just as GraphConvModel
    does, but these ConvMol objects must be transformed by
    dc.trans.DAGTransformer.

    As a note, performance of this model can be a little
    sensitive to initialization. It might be worth training a few
    different instantiations to get a stable set of parameters.

    Examples
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> from deepchem.models.torch_models import DAGModel
    >>> from deepchem.molnet import load_bace_classification
    >>> from deepchem.data import NumpyDataset
    >>> from deepchem.trans import DAGTransformer
    >>> n_tasks = 1
    >>> n_features = 75
    >>> n_classes = 2
    >>> n_samples = 10
    >>> tasks, all_dataset, transformers = load_bace_classification("GraphConv", reload=False)
    >>> train_dataset, valid_dataset, test_dataset = all_dataset
    >>> dataset = NumpyDataset(train_dataset.X[:n_samples], train_dataset.y[:n_samples], train_dataset.w[:n_samples], train_dataset.ids[:n_samples])
    >>> max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    >>> transformer = DAGTransformer(max_atoms=max_atoms)
    >>> dataset = transformer.transform(dataset)
    >>> model = DAGModel(n_tasks=n_tasks, max_atoms=max_atoms, mode='classification', n_classes=n_classes)
    >>> # train a model
    >>> _ = model.fit(dataset)
    >>> # inferencing
    >>> _ = model.predict(dataset)

    References
    ----------
    .. [1] Lusci Alessandro, Gianluca Pollastri, and Pierre Baldi."Deep architectures and deep learning in chemoinformatics: the prediction of aqueous solubility for drug-like molecules."Journal of chemoinformatics and modeling 53.7 (2013):1563-1575.https://pmc.ncbi.nlm.nih.gov/articles/PMC3739985
    """

    def __init__(self,
                 n_tasks: int,
                 max_atoms: int = 50,
                 n_atom_feat: int = 75,
                 n_graph_feat: int = 30,
                 n_outputs: int = 30,
                 layer_sizes: List[int] = [100],
                 layer_sizes_gather: List[int] = [100],
                 dropout: Optional[float] = None,
                 mode: str = "classification",
                 n_classes: int = 2,
                 uncertainty: bool = False,
                 batch_size: int = 100,
                 device: Optional[torch.device] = None,
                 **kwargs: Any) -> None:
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        max_atoms: int, optional
            Maximum number of atoms in a molecule, should be defined based on dataset.
        n_atom_feat: int, optional
            Number of features per atom.
        n_graph_feat: int, optional
            Number of features for atom in the graph.
        n_outputs: int, optional
            Number of features for each molecule.
        layer_sizes: list of int, optional
            List of hidden layer size(s) in the propagation step.
        layer_sizes_gather: list of int, optional
            List of hidden layer size(s) in the gather step.
        dropout: None or float, optional
            Dropout probability.
        mode: str, optional
            Either "classification" or "regression" for type of model.
        n_classes: int, optional
            the number of classes to predict (only used in classification mode)
        uncertainty: bool, optional
            if True, include extra outputs to enable uncertainty prediction
        batch_size: int, optional
            the batch size to use during training
        device: str, optional
            the device to run the model on
        """
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = _DAG(n_tasks=n_tasks,
                          max_atoms=max_atoms,
                          n_atom_feat=n_atom_feat,
                          n_graph_feat=n_graph_feat,
                          n_outputs=n_outputs,
                          layer_sizes=layer_sizes,
                          layer_sizes_gather=layer_sizes_gather,
                          dropout=dropout,
                          mode=mode,
                          n_classes=n_classes,
                          uncertainty=uncertainty,
                          batch_size=batch_size,
                          device=device)
        self.n_tasks = n_tasks
        self.mode = mode
        self.max_atoms = max_atoms
        self.n_outputs = n_outputs
        self.n_classes = n_classes
        if mode == 'classification':
            self.output_types = ['prediction', 'loss']
            self.loss: Union[Loss, LossFn] = SoftmaxCrossEntropy()
        else:
            if uncertainty:
                self.output_types = ['prediction', 'variance', 'loss', 'loss']

                def loss(outputs: List, labels: List,
                         weights: List) -> torch.Tensor:
                    # Ensure outputs and labels are shape-consistent
                    output, labels = _make_pytorch_shapes_consistent(
                        outputs[0], labels[0])
                    # Compute the losses
                    losses = (output - labels)**2 / torch.exp(
                        outputs[1]) + outputs[1]

                    # Handle weights reshaping if necessary
                    w = weights[0]
                    if len(w.shape) < len(losses.shape):
                        shape = tuple(w.shape)
                        shape = tuple(-1 if x is None else x for x in shape)
                        w = w.view(*shape,
                                   *([1] * (len(losses.shape) - len(w.shape))))

                    # Compute the weighted mean loss
                    return torch.mean(losses * w)

                self.loss = loss
            else:
                self.output_types = ['prediction']
                self.loss = L2Loss()
        super(DAGModel, self).__init__(self.model,
                                       loss=self.loss,
                                       output_types=self.output_types,
                                       batch_size=batch_size,
                                       device=device,
                                       **kwargs)

    def default_generator(
            self,
            dataset: dc.data.Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        """
        Convert a dataset into the tensors needed for learning.

        Parameters
        ----------
        dataset : object
            The dataset to iterate over
        epochs : int, optional
            Number of epochs to iterate
        mode : str, optional
            The mode of the generator. One of 'fit', 'predict', 'uncertainty'
        deterministic : bool, optional
            Whether to iterate over the dataset deterministically
        pad_batches : bool, optional
            Whether to pad the batches to the same size

        Yields
        ------
        tuple
            A tuple of (inputs, labels, weights) as expected by TorchModel.fit()
        """
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):

                # Convert labels for classification
                if y_b is not None and self.mode == 'classification':
                    y_b = np.array(y_b.flatten())
                    y_b = to_one_hot(y_b, self.n_classes)
                    y_b = np.reshape(y_b, (-1, self.n_tasks, self.n_classes))

                # Process molecular graphs
                atoms_per_mol = [mol.get_num_atoms() for mol in X_b]
                n_atoms = sum(atoms_per_mol)
                start_index = [0] + list(np.cumsum(atoms_per_mol)[:-1])

                atoms_all = []
                parents_all = []
                calculation_orders = []
                calculation_masks = []
                membership = []

                for idm, mol in enumerate(X_b):
                    atoms_all.append(mol.get_atom_features())
                    parents = mol.parents
                    parents_all.extend(parents)
                    calculation_index = np.array(parents)[:, :, 0]
                    mask = np.array(calculation_index - self.max_atoms,
                                    dtype=bool)
                    calculation_orders.append(calculation_index +
                                              start_index[idm])
                    calculation_masks.append(mask)
                    membership.extend([idm] * atoms_per_mol[idm])

                # Create inputs tuple
                inputs = [
                    np.concatenate(atoms_all, axis=0),
                    np.stack(parents_all, axis=0),
                    np.concatenate(calculation_orders, axis=0),
                    np.concatenate(calculation_masks, axis=0),
                    np.array(membership),
                    np.array(n_atoms)
                ]

                yield inputs, [y_b], [w_b]
