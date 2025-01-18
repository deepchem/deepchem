import torch.nn as nn
from deepchem.models.torch_models import layers
import torch
import torch.nn.functional as F
import deepchem as dc
from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
import numpy as np
from deepchem.metrics import to_one_hot



class DAGModel(TorchModel):
    """Directed Acyclic Graph models for molecular property prediction.

    This model is based on the following paper:

    Lusci, Alessandro, Gianluca Pollastri, and Pierre Baldi. "Deep architectures and deep learning in chemoinformatics: the prediction of aqueous solubility for drug-like molecules." Journal of chemical information and modeling 53.7 (2013): 1563-1575.

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
    """

    def __init__(self,
                 n_tasks,
                 max_atoms=50,
                 n_atom_feat=75,
                 n_graph_feat=30,
                 n_outputs=30,
                 layer_sizes=[100],
                 layer_sizes_gather=[100],
                 dropout=None,
                 mode="classification",
                 n_classes=2,
                 uncertainty=False,
                 batch_size=100,
                 **kwargs):
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
            List of hidden layer size(s) in the propagation step:
            length of this list represents the number of hidden layers,
            and each element is the width of corresponding hidden layer.
        layer_sizes_gather: list of int, optional
            List of hidden layer size(s) in the gather step.
        dropout: None or float, optional
            Dropout probability, applied after each propagation step and gather step.
        mode: str, optional
            Either "classification" or "regression" for type of model.
        n_classes: int
            the number of classes to predict (only used in classification mode)
        uncertainty: bool
            if True, include extra outputs and loss terms to enable the uncertainty
            in outputs to be predicted
        """
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")
        self.n_tasks = n_tasks
        self.max_atoms = max_atoms
        self.n_atom_feat = n_atom_feat
        self.n_graph_feat = n_graph_feat
        self.n_outputs = n_outputs
        self.layer_sizes = layer_sizes
        self.layer_sizes_gather = layer_sizes_gather
        self.dropout = dropout
        self.mode = mode
        self.n_classes = n_classes
        self.uncertainty = uncertainty
        if uncertainty:
            if mode != "regression":
                raise ValueError(
                    "Uncertainty is only supported in regression mode")
            if dropout is None or dropout == 0.0:
                raise ValueError(
                    'Dropout must be included to predict uncertainty')

        # Build the model.
        model = DAG(n_graph_feat=self.n_graph_feat,
                                    mode=self.mode,
                                    n_tasks=self.n_tasks,   
                                     n_atom_feat=self.n_atom_feat,
                                     max_atoms=self.max_atoms,
                                     layer_sizes=self.layer_sizes,
                                     dropout=self.dropout,
                                     batch_size=batch_size,
                                     n_outputs=self.n_outputs,
                                     layer_sizes_gather=self.layer_sizes_gather)
        n_tasks = self.n_tasks
        if self.mode == 'classification':
            output_types = ['prediction', 'loss']
            loss = SoftmaxCrossEntropy()
        else:

            if self.uncertainty:
                output_types = ['prediction', 'variance', 'loss', 'loss']

                def loss(outputs, labels, weights):
                    output, labels = dc.models.losses._make_pytorch_shapes_consistent(
                        outputs[0], labels[0])
                    losses = torch.square(output - labels) / torch.exp(
                        outputs[1]) + outputs[1]
                    w = weights[0]
                    if len(w.shape) < len(losses.shape):
                        shape = tuple(w.shape)
                        shape = tuple(-1 if x is None else x for x in shape)
                        w = torch.reshape(
                            w,
                            shape + (1,) * (len(losses.shape) - len(w.shape)))
                    return torch.mean(losses * w)
            
            else:
                output_types = ['prediction']
                loss = L2Loss()
        super(DAGModel, self).__init__(model,
                                       loss,
                                       output_types=output_types,
                                       batch_size=batch_size,
                                       **kwargs)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
    
        """Convert a dataset into the tensors needed for learning"""
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):

                if y_b is not None and self.mode == 'classification':
                    y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                        -1, self.n_tasks, self.n_classes)

                atoms_per_mol = [mol.get_num_atoms() for mol in X_b]
                n_atoms = sum(atoms_per_mol)
                start_index = [0] + list(np.cumsum(atoms_per_mol)[:-1])

                atoms_all = []
                # calculation orders for a batch of molecules
                parents_all = []
                calculation_orders = []
                calculation_masks = []
                membership = []
                for idm, mol in enumerate(X_b):
                    # padding atom features vector of each molecule with 0
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
                if mode == 'predict':
                    dropout = np.array(0.0)
                else:
                    dropout = np.array(1.0)
                X = [
                    np.concatenate(atoms_all, axis=0),
                    np.stack(parents_all, axis=0),
                    np.concatenate(calculation_orders, axis=0),
                    np.concatenate(calculation_masks, axis=0),
                    np.array(membership),
                    np.array(n_atoms), dropout
                ]
                yield (X, [y_b], [w_b])
   

           
class DAG(nn.Module):
    def __init__(self,
                 n_tasks,
                 max_atoms=50,
                 n_atom_feat=75,
                 n_graph_feat=30,
                 n_outputs=30,
                 layer_sizes=[100],
                 layer_sizes_gather=[100],
                 dropout=None,
                 mode="classification",
                 n_classes=2,
                 uncertainty=False,
                 batch_size=100,
                 **kwargs):
        super(DAG, self).__init__()

        self.n_tasks = n_tasks
        self.max_atoms = max_atoms
        self.n_atom_feat = n_atom_feat
        self.n_graph_feat = n_graph_feat
        self.n_outputs = n_outputs
        self.layer_sizes = layer_sizes
        self.layer_sizes_gather = layer_sizes_gather
        self.dropout = dropout
        self.mode = mode
        self.n_classes = n_classes
        self.uncertainty = uncertainty
        
        self.dag_layer = layers.DAGLayer(n_graph_feat=self.n_graph_feat,
                                     n_atom_feat=self.n_atom_feat,
                                     max_atoms=self.max_atoms,
                                     layer_sizes=self.layer_sizes,
                                     dropout=self.dropout,
                                     batch_size=batch_size)
        self.dag_gather = layers.DAGGather(
            n_graph_feat=self.n_graph_feat,
            n_outputs=self.n_outputs,
            max_atoms=self.max_atoms,
            layer_sizes=self.layer_sizes_gather,
            dropout=self.dropout)
        
        self.dense1 = nn.Linear(n_outputs, n_tasks * n_classes)
        self.dense2 = nn.Linear(n_outputs, n_tasks)


    def forward(self, inputs):
        dag_layer_output = self.dag_layer([inputs[0], inputs[1], inputs[2], inputs[3],inputs[5]])
        dag_gather_output = self.dag_gather([dag_layer_output, inputs[4]])
        n_tasks = self.n_tasks
    
        if self.mode == 'classification':
            n_classes = self.n_classes

            x = self.dense1(dag_gather_output)
            logits = x.view(-1, n_tasks, n_classes)
            
            output = F.softmax(logits, dim=2)
            outputs = [output, logits]

        else:
            output = self.dense2(dag_gather_output)
            if self.uncertainty:
                log_var = self.dense2(dag_gather_output)
                var = torch.exp(log_var)
                outputs = [output, var, output, log_var]
            else:
                outputs = [output]
        return outputs

