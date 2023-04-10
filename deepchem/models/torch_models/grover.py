import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Sequence, Optional, Any, Tuple
from rdkit import Chem
from deepchem.feat.graph_data import BatchGraphData
from deepchem.models.torch_models.modular import ModularTorchModel
from deepchem.models.torch_models.grover_layers import (
    GroverEmbedding, GroverBondVocabPredictor, GroverAtomVocabPredictor,
    GroverFunctionalGroupPredictor)
from deepchem.models.torch_models.readout import GroverReadout
from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder
from deepchem.utils.grover import extract_grover_attributes


class GroverPretrain(nn.Module):
    """The Grover Pretrain module.

    The GroverPretrain module is used for training an embedding based on the Grover Pretraining task.
    Grover pretraining is a self-supervised task where an embedding is trained to learn the contextual
    information of atoms and bonds along with graph-level properties, which are functional groups
    in case of molecular graphs.

    Parameters
    ----------
    embedding: nn.Module
        An embedding layer to generate embedding from input molecular graph
    atom_vocab_task_atom: nn.Module
        A layer used for predicting atom vocabulary from atom features generated via atom hidden states.
    atom_vocab_task_bond: nn.Module
        A layer used for predicting atom vocabulary from atom features generated via bond hidden states.
    bond_vocab_task_atom: nn.Module
        A layer used for predicting bond vocabulary from bond features generated via atom hidden states.
    bond_vocab_task_bond: nn.Module
        A layer used for predicting bond vocabulary from bond features generated via bond hidden states.

    Returns
    -------
    prediction_logits: Tuple
        A tuple of prediction logits containing prediction logits of atom vocabulary task from atom hidden state,
    prediction logits for atom vocabulary task from bond hidden states, prediction logits for bond vocabulary task
    from atom hidden states, prediction logits for bond vocabulary task from bond hidden states, functional
    group prediction logits from atom embedding generated from atom and bond hidden states, functional group
    prediction logits from bond embedding generated from atom and bond hidden states.

    Example
    -------
    >>> import deepchem as dc
    >>> from deepchem.feat.graph_data import BatchGraphData
    >>> from deepchem.utils.grover import extract_grover_attributes
    >>> from deepchem.models.torch_models.grover import GroverPretrain
    >>> from deepchem.models.torch_models.grover_layers import GroverEmbedding, GroverAtomVocabPredictor, GroverBondVocabPredictor, GroverFunctionalGroupPredictor
    >>> smiles = ['CC', 'CCC', 'CC(=O)C']

    >>> fg = dc.feat.CircularFingerprint()
    >>> featurizer = dc.feat.GroverFeaturizer(features_generator=fg)

    >>> graphs = featurizer.featurize(smiles)
    >>> batched_graph = BatchGraphData(graphs)
    >>> grover_graph_attributes = extract_grover_attributes(batched_graph)
    >>> f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, _, _ = grover_graph_attributes
    >>> components = {}
    >>> components['embedding'] = GroverEmbedding(node_fdim=f_atoms.shape[1], edge_fdim=f_bonds.shape[1])
    >>> components['atom_vocab_task_atom'] = GroverAtomVocabPredictor(vocab_size=10, in_features=128)
    >>> components['atom_vocab_task_bond'] = GroverAtomVocabPredictor(vocab_size=10, in_features=128)
    >>> components['bond_vocab_task_atom'] = GroverBondVocabPredictor(vocab_size=10, in_features=128)
    >>> components['bond_vocab_task_bond'] = GroverBondVocabPredictor(vocab_size=10, in_features=128)
    >>> components['functional_group_predictor'] = GroverFunctionalGroupPredictor(10)
    >>> model = GroverPretrain(**components)

    >>> inputs = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
    >>> output = model(inputs)

    Reference
    ---------
    .. Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." Advances in Neural Information Processing Systems 33 (2020): 12559-12571.
    """

    def __init__(self, embedding: nn.Module, atom_vocab_task_atom: nn.Module,
                 atom_vocab_task_bond: nn.Module,
                 bond_vocab_task_atom: nn.Module,
                 bond_vocab_task_bond: nn.Module,
                 functional_group_predictor: nn.Module):
        super(GroverPretrain, self).__init__()
        self.embedding = embedding
        self.atom_vocab_task_atom = atom_vocab_task_atom
        self.atom_vocab_task_bond = atom_vocab_task_bond
        self.bond_vocab_task_atom = bond_vocab_task_atom
        self.bond_vocab_task_bond = bond_vocab_task_bond
        self.functional_group_predictor = functional_group_predictor

    def forward(self, graph_batch):
        """Forward function

        Parameters
        ----------
        graph_batch: List[torch.Tensor]
            A list containing grover graph attributes
        """
        _, _, _, _, _, atom_scope, bond_scope, _ = graph_batch
        atom_scope = atom_scope.data.cpu().numpy().tolist()
        bond_scope = bond_scope.data.cpu().numpy().tolist()

        embeddings = self.embedding(graph_batch)
        av_task_atom_pred = self.atom_vocab_task_atom(
            embeddings["atom_from_atom"])
        av_task_bond_pred = self.atom_vocab_task_bond(
            embeddings["atom_from_bond"])

        bv_task_atom_pred = self.bond_vocab_task_atom(
            embeddings["bond_from_atom"])
        bv_task_bond_pred = self.bond_vocab_task_bond(
            embeddings["bond_from_bond"])

        fg_prediction = self.functional_group_predictor(embeddings, atom_scope,
                                                        bond_scope)

        return av_task_atom_pred, av_task_bond_pred, bv_task_atom_pred, bv_task_bond_pred, fg_prediction[
            'atom_from_atom'], fg_prediction['atom_from_bond'], fg_prediction[
                'bond_from_atom'], fg_prediction['bond_from_bond']


class GroverFinetune(nn.Module):
    """Grover Finetune model.

    For a graph level prediction task, the GroverFinetune model uses node/edge embeddings
    output by the GroverEmbeddong layer and applies a readout function on it to get
    graph embeddings and use additional MLP layers to predict the property of the molecular graph.

    Parameters
    ----------
    embedding: nn.Module
        An embedding layer to generate embedding from input molecular graph
    readout: nn.Module
        A readout layer to perform readout atom and bond hidden states
    mol_atom_from_atom_ffn: nn.Module
        A feed forward network which learns representation from atom messages generated via atom hidden states of a molecular graph
    mol_atom_from_bond_ffn: nn.Module
        A feed forward network which learns representation from atom messages generated via bond hidden states of a molecular graph
    mode: str
        classification or regression

    Returns
    -------
    prediction_logits: torch.Tensor
        prediction logits

    Example
    -------
    >>> import deepchem as dc
    >>> from deepchem.feat.graph_data import BatchGraphData
    >>> from deepchem.utils.grover import extract_grover_attributes
    >>> from deepchem.models.torch_models.grover_layers import GroverEmbedding
    >>> from deepchem.models.torch_models.readout import GroverReadout
    >>> from deepchem.models.torch_models.grover import GroverFinetune
    >>> smiles = ['CC', 'CCC', 'CC(=O)C']
    >>> fg = dc.feat.CircularFingerprint()
    >>> featurizer = dc.feat.GroverFeaturizer(features_generator=fg)
    >>> graphs = featurizer.featurize(smiles)
    >>> batched_graph = BatchGraphData(graphs)
    >>> attributes = extract_grover_attributes(batched_graph)
    >>> components = {}
    >>> f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, fg_labels, additional_features = attributes
    >>> inputs = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
    >>> components = {}
    >>> components['embedding'] = GroverEmbedding(node_fdim=f_atoms.shape[1], edge_fdim=f_bonds.shape[1])
    >>> components['readout'] = GroverReadout(rtype="mean", in_features=128)
    >>> components['mol_atom_from_atom_ffn'] = nn.Linear(in_features=additional_features.shape[1]+ 128, out_features=128)
    >>> components['mol_atom_from_bond_ffn'] = nn.Linear(in_features=additional_features.shape[1] + 128, out_features=128)
    >>> model = GroverFinetune(**components, mode='regression', hidden_size=128)
    >>> model.training = False
    >>> output = model((inputs, additional_features))

    Reference
    ---------
    .. Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." Advances in Neural Information Processing Systems 33 (2020): 12559-12571.
    """

    def __init__(self,
                 embedding: nn.Module,
                 readout: nn.Module,
                 mol_atom_from_atom_ffn: nn.Module,
                 mol_atom_from_bond_ffn: nn.Module,
                 hidden_size: int = 128,
                 mode: str = 'regression',
                 n_tasks: int = 1,
                 n_classes: Optional[int] = None):
        super().__init__()
        self.embedding = embedding
        self.readout = readout
        self.mol_atom_from_atom_ffn = mol_atom_from_atom_ffn
        self.mol_atom_from_bond_ffn = mol_atom_from_bond_ffn
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.mode = mode
        # the hidden size here is the output size of last layer in mol_atom_from_atom_ffn and mol_atom_from_bond_ffn components.
        # it is necessary that aforementioned components produces output tensor of same size.
        if self.mode == 'classification':
            assert n_classes is not None
            self.linear = nn.Linear(hidden_size,
                                    out_features=n_tasks * n_classes)
        elif self.mode == 'regression':
            self.linear = nn.Linear(hidden_size, out_features=n_tasks)

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs: Tuple
            grover batch graph attributes
        """
        graphbatch, additional_features = inputs
        _, _, _, _, _, atom_scope, bond_scope, _ = graphbatch
        output = self.embedding(graphbatch)

        mol_atom_from_bond_output = self.readout(output["atom_from_bond"],
                                                 atom_scope)
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"],
                                                 atom_scope)

        if additional_features[0] is not None:
            additional_features = torch.from_numpy(
                np.stack(additional_features)).float()
            additional_features.to(output["atom_from_bond"])
            if len(additional_features.shape) == 1:
                additional_features = additional_features.view(
                    1, additional_features.shape[0])
            mol_atom_from_atom_output = torch.cat(
                [mol_atom_from_atom_output, additional_features], 1)
            mol_atom_from_bond_output = torch.cat(
                [mol_atom_from_bond_output, additional_features], 1)

        atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
        bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
        if self.training:
            # In training mode, we return atom level aggregated output and bond level aggregated output.
            # The training has an additional objective which ensures that atom and bond level aggregated outputs
            # are similar to each other apart from the objective of making the aggregated output closer to each
            # other.
            return atom_ffn_output, bond_ffn_output
        else:
            if self.mode == 'classification':
                atom_ffn_output = torch.sigmoid(atom_ffn_output)
                bond_ffn_output = torch.sigmoid(bond_ffn_output)
            output = (atom_ffn_output + bond_ffn_output) / 2
            output = self.linear(output)

            if self.mode == 'classification':
                if self.n_tasks == 1:
                    logits = output.view(-1, self.n_classes)
                    softmax_dim = 1
                else:
                    logits = output.view(-1, self.n_tasks, self.n_classes)
                    softmax_dim = 2
                proba = F.softmax(logits, dim=softmax_dim)
                return proba, logits
            elif self.mode == 'regression':
                return output


class GroverModel(ModularTorchModel):
    """GROVER model

    The GROVER model employs a self-supervised message passing transformer architecutre
    for learning molecular representation. The pretraining task can learn rich structural
    and semantic information of molecules from unlabelled molecular data, which can
    be leveraged by finetuning for downstream applications. To this end, GROVER integrates message
    passing networks into a transformer style architecture.

    Parameters
    ----------
    node_fdim: int
        the dimension of additional feature for node/atom.
    edge_fdim: int
        the dimension of additional feature for edge/bond.
    atom_vocab: GroverAtomVocabularyBuilder
        Grover atom vocabulary builder required during pretraining.
    bond_vocab: GroverBondVocabularyBuilder
        Grover bond vocabulary builder required during pretraining.
    hidden_size: int
        Size of hidden layers
    features_only: bool
        Uses only additional features in the feed-forward network, no graph network
    self_attention: bool, default False
        When set to True, a self-attention layer is used during graph readout operation.
    functional_group_size: int (default: 85)
        Size of functional group used in grover.
    features_dim: int
        Size of additional molecular features, like fingerprints.
    ffn_num_layers: int (default: 1)
        Number of linear layers to use for feature extraction from embeddings
    task: str (pretraining or finetuning)
        Pretraining or finetuning tasks.
    mode: str (classification or regression)
        Training mode (used only for finetuning)
    n_tasks: int, optional (default: 1)
        Number of tasks
    n_classes: int, optiona (default: 2)
        Number of target classes in classification mode
    model_dir: str
        Directory to save model checkpoints
    dropout: float, optional (default: 0.2)
        dropout value
    actionvation: str, optional (default: 'relu')
        supported activation function

    Example
    -------
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.grover import GroverModel
    >>> from deepchem.feat.vocabulary_builders import (GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder)
    >>> import pandas as pd
    >>> import os
    >>> import tempfile
    >>> tmpdir = tempfile.mkdtemp()
    >>> df = pd.DataFrame({'smiles': ['CC', 'CCC'], 'preds': [0, 0]})
    >>> filepath = os.path.join(tmpdir, 'example.csv')
    >>> df.to_csv(filepath, index=False)
    >>> dataset_path = os.path.join(filepath)
    >>> loader = dc.data.CSVLoader(tasks=['preds'], featurizer=dc.feat.DummyFeaturizer(), feature_field=['smiles'])
    >>> dataset = loader.create_dataset(filepath)
    >>> av = GroverAtomVocabularyBuilder()
    >>> av.build(dataset)
    >>> bv = GroverBondVocabularyBuilder()
    >>> bv.build(dataset)
    >>> fg = dc.feat.CircularFingerprint()
    >>> loader2 = dc.data.CSVLoader(tasks=['preds'], featurizer=dc.feat.GroverFeaturizer(features_generator=fg), feature_field='smiles')
    >>> graph_data = loader2.create_dataset(filepath)
    >>> model = GroverModel(node_fdim=151, edge_fdim=165, atom_vocab=av, bond_vocab=bv, features_dim=2048, hidden_size=128, functional_group_size=85, mode='regression', task='finetuning', model_dir='gm')
    >>> loss = model.fit(graph_data, nb_epoch=1)

    Reference
    ---------
    .. Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." Advances in Neural Information Processing Systems 33 (2020): 12559-12571.
    """

    def __init__(self,
                 node_fdim: int,
                 edge_fdim: int,
                 atom_vocab: GroverAtomVocabularyBuilder,
                 bond_vocab: GroverBondVocabularyBuilder,
                 hidden_size: int,
                 self_attention=False,
                 features_only=False,
                 functional_group_size: int = 85,
                 features_dim=128,
                 dropout=0.2,
                 activation='relu',
                 task='pretraining',
                 ffn_num_layers=1,
                 mode: Optional[str] = None,
                 model_dir=None,
                 n_tasks: int = 1,
                 n_classes: Optional[int] = None,
                 **kwargs):
        assert task in ['pretraining', 'finetuning']
        self.ffn_num_layers = ffn_num_layers
        self.activation = activation
        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.atom_vocab_size = atom_vocab.size
        self.bond_vocab_size = bond_vocab.size
        self.task = task
        self.model_dir = model_dir
        self.hidden_size = hidden_size
        self.attn_hidden_size = hidden_size
        self.attn_out_size = hidden_size
        self.functional_group_size = functional_group_size
        self.self_attention = self_attention
        self.features_only = features_only
        self.features_dim = features_dim
        self.dropout = dropout
        self.mode = mode
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.components = self.build_components()
        self.model = self.build_model()
        super().__init__(self.model,
                         self.components,
                         model_dir=self.model_dir,
                         **kwargs)
        # FIXME In the above step, we initialize modular torch model but
        # something is missing here. The attribute loss from TorchModel gets assigned `loss_func`
        # by super class initialization in ModularTorchModel but here we reinitialize it.
        self.loss = self.get_loss_func()

    def build_components(self):
        """Builds components for grover pretraining and finetuning model.

        .. list-table:: Components of pretraining model
           :widths: 25 25 50
           :header-rows: 1

           * - Component name
             - Type
             - Description
           * - `embedding`
             - Graph message passing network
             - A layer which accepts a molecular graph and produces an embedding for grover pretraining task
           * - `atom_vocab_task_atom`
             - Feed forward layer
             - A layer which accepts an embedding generated from atom hidden states and predicts atom vocabulary for grover pretraining task
           * - `atom_vocab_task_bond`
             - Feed forward layer
             - A layer which accepts an embedding generated from bond hidden states and predicts atom vocabulary for grover pretraining task
           * - `bond_vocab_task_atom`
             - Feed forward layer
             - A layer which accepts an embedding generated from atom hidden states and predicts bond vocabulary for grover pretraining task
           * - `bond_vocab_task_bond`
             - Feed forward layer
             - A layer which accepts an embedding generated from bond hidden states and predicts bond vocabulary for grover pretraining task
           * - `functional_group_predictor`
             - Feed forward layer
             - A layer which accepts an embedding generated from a graph readout and predicts functional group for grover pretraining task

        .. list-table:: Components of finetuning model

           * - Component name
             - Type
             - Description
           * - `embedding`
             - Graph message passing network
             - An embedding layer to generate embedding from input molecular graph
           * - `readout`
             - Feed forward layer
             - A readout layer to perform readout atom and bond hidden states
           * - `mol_atom_from_atom_ffn`
             - Feed forward layer
             - A feed forward network which learns representation from atom messages generated via atom hidden states of a molecular graph
           * - `mol_atom_from_bond_ffn`
             - Feed forward layer
             - A feed forward network which learns representation from atom messages generated via bond hidden states of a molecular graph
        """
        if self.task == 'pretraining':
            components = self._get_pretraining_components()
        elif self.task == 'finetuning':
            components = self._get_finetuning_components()
        return components

    def build_model(self):
        """Builds grover pretrain or finetune model based on task"""
        if self.task == 'pretraining':
            return GroverPretrain(**self.components)
        elif self.task == 'finetuning':
            return GroverFinetune(**self.components,
                                  mode=self.mode,
                                  hidden_size=self.hidden_size,
                                  n_tasks=self.n_tasks,
                                  n_classes=self.n_classes)

    def get_loss_func(self):
        """Returns loss function based on task"""
        if self.task == 'pretraining':
            from deepchem.models.losses import GroverPretrainLoss
            return GroverPretrainLoss()._create_pytorch_loss()
        elif self.task == 'finetuning':
            return self._finetuning_loss

    def loss_func(self, inputs, labels, weights):
        """Returns loss function which performs forward iteration based on task type"""
        if self.task == 'pretraining':
            return self._pretraining_loss(inputs, labels, weights)
        elif self.task == 'finetuning':
            return self._finetuning_loss(inputs, labels, weights)

    def _get_pretraining_components(self):
        """Return pretraining components.

        The component names are described in GroverModel.build_components method.
        """
        components = {}
        components['embedding'] = GroverEmbedding(node_fdim=self.node_fdim,
                                                  edge_fdim=self.edge_fdim)
        components['atom_vocab_task_atom'] = GroverAtomVocabPredictor(
            self.atom_vocab_size, self.hidden_size)
        components['atom_vocab_task_bond'] = GroverAtomVocabPredictor(
            self.atom_vocab_size, self.hidden_size)
        components['bond_vocab_task_atom'] = GroverBondVocabPredictor(
            self.bond_vocab_size, self.hidden_size)
        components['bond_vocab_task_bond'] = GroverBondVocabPredictor(
            self.bond_vocab_size, self.hidden_size)
        components[
            'functional_group_predictor'] = GroverFunctionalGroupPredictor(
                self.functional_group_size)
        return components

    def _get_finetuning_components(self):
        """Return finetuning components.

        The component names are described in GroverModel.build_components method.
        """
        components = {}
        components['embedding'] = GroverEmbedding(node_fdim=self.node_fdim,
                                                  edge_fdim=self.edge_fdim)
        if self.self_attention:
            components['readout'] = GroverReadout(
                rtype="self_attention",
                in_features=self.hidden_size,
                attn_hidden=self.attn_hidden_size,
                attn_out=self.attn_out_size)
        else:
            components['readout'] = GroverReadout(rtype="mean",
                                                  in_features=self.hidden_size)
        components['mol_atom_from_atom_ffn'] = self._create_ffn()
        components['mol_atom_from_bond_ffn'] = self._create_ffn()

        return components

    def _prepare_batch(self, batch):
        """Prepare batch method for preprating batch of data for finetuning and pretraining tasks"""
        if self.task == 'pretraining':
            return self._prepare_batch_for_pretraining(batch)
        elif self.task == 'finetuning':
            return self._prepare_batch_for_finetuning(batch)

    def _prepare_batch_for_pretraining(self, batch: Tuple[Any, Any, Any]):
        """Prepare batch for pretraining

        This method is used for batching a sequence of graph data objects using BatchGraphData method.
        It batches a graph and extracts attributes from the batched graph and return it as inputs for
        the model. It also performs generates labels for atom vocab prediction and bonc vocab
        prediction tasks in grover.

        Parameters
        ----------
        batch: Tuple[Sequence[GraphData], Any, Any]
            A batch of data containing grover molecular graphs, target prediction values and weights for datapoint.

        Returns
        -------
        inputs: Tuple
            Inputs for grover pretraining model
        labels: Dict[str, torch.Tensor]
            Labels for grover pretraining self-supervised task
        w: Any
            Weights of data point
        """
        X, y, w = batch
        batchgraph = BatchGraphData(X[0])
        fgroup_label = getattr(batchgraph, 'fg_labels')
        smiles_batch = getattr(batchgraph, 'smiles').reshape(-1).tolist()

        f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, _, _ = extract_grover_attributes(
            batchgraph)

        atom_vocab_label = torch.Tensor(
            self.atom_vocab_random_mask(self.atom_vocab, smiles_batch)).long()
        bond_vocab_label = torch.Tensor(
            self.bond_vocab_random_mask(self.bond_vocab, smiles_batch)).long()
        labels = {
            "av_task": atom_vocab_label,
            "bv_task": bond_vocab_label,
            "fg_task": torch.Tensor(fgroup_label)
        }
        inputs = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
        return inputs, labels, w

    def _prepare_batch_for_finetuning(self, batch: Tuple[Any, Any, Any]):
        """Prepare batch for finetuning task

        The method batches a sequence of grover graph data objects using BatchGraphData utility
        and extracts attributes from the batched graph. The extracted attributes are fed to
        the grover model as inputs.

        Parameters
        ----------
        batch: Tuple[Sequence[GraphData], Any, Any]
            A batch of data containing grover molecular graphs, target prediction values and weights for datapoint.

        Returns
        -------
        inputs: Tuple
            Inputs for grover finetuning model
        labels: Dict[str, torch.Tensor]
            Labels for grover finetuning task
        w: Any
            Weights of data point
        """
        X, y, w = batch
        batchgraph = BatchGraphData(X[0])
        if y is not None:
            labels = torch.FloatTensor(y[0])
        else:
            labels = None
        f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, _, additional_features = extract_grover_attributes(
            batchgraph)
        inputs = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope,
                  a2a), additional_features
        return inputs, labels, w

    def _pretraining_loss(self,
                          inputs,
                          labels,
                          weights: Optional[List[Sequence]] = None,
                          dist_coff: float = 0.1):
        """Grover pretraining loss

        The Grover pretraining loss function performs a forward iteration and returns
        the loss value.

        Parameters
        ----------
        inputs: Tuple[torch.Tensor]
            extracted grover graph attributed
        labels: Dict[str, torch.Tensor]
            Target predictions
        weights: List[Sequence]
            Weight to assign to each datapoint
        dist_coff: float, default: 0.1
            Loss term weight for weighting closeness between embedding generated from atom hidden state and bond hidden state in atom vocabulary and bond vocabulary prediction tasks.

        Returns
        -------
        loss: torch.Tensor
            loss value
        """
        _, _, _, _, _, atom_scope, bond_scope, _ = inputs
        av_task_atom_pred, av_task_bond_pred, bv_task_atom_pred, bv_task_bond_pred, fg_prediction_atom_from_atom, fg_prediction_atom_from_bond, fg_prediction_bond_from_atom, fg_prediction_bond_from_bond = self.model(
            inputs)

        loss = self.loss(av_task_atom_pred,
                         av_task_bond_pred,
                         bv_task_atom_pred,
                         bv_task_bond_pred,
                         fg_prediction_atom_from_atom,
                         fg_prediction_atom_from_bond,
                         fg_prediction_bond_from_atom,
                         fg_prediction_bond_from_bond,
                         labels['av_task'],
                         labels['bv_task'],
                         labels['fg_task'],
                         weights=weights,
                         dist_coff=dist_coff)  # type: ignore
        return loss

    def _finetuning_loss(self, inputs, labels, weights, dist_coff=0.1):
        """Loss function for finetuning task

        The finetuning loss is a binary cross entropy loss for classification mode and
        mean squared error loss for regression mode. During training of the model, apart from
        learning the data distribution, the loss function is also used to make the embedding
        generated from atom hidden state and bond hidden state close to each other.

        Parameters
        ----------
        inputs: Tuple[torch.Tensor]
            extracted grover graph attributed
        labels: Dict[str, torch.Tensor]
            Target predictions
        weights: List[Sequence]
            Weight to assign to each datapoint
        dist_coff: float, default: 0.1
            Loss term weight for weighting closeness between embedding generated from atom hidden state and bond hidden state

        Returns
        -------
        loss: torch.Tensor
            loss value
        """
        if self.mode == 'classification':
            pred_loss = nn.BCEWithLogitsLoss()
        elif self.mode == 'regression':
            pred_loss = nn.MSELoss()

        preds = self.model(inputs)

        if not self.model.training:
            # in eval mode.
            return pred_loss(preds, labels)
        elif self.model.training:
            dist_loss = nn.MSELoss()
            dist = dist_loss(preds[0], preds[1])
            pred_loss1 = pred_loss(preds[0].mean(axis=-1).unsqueeze(-1), labels)
            pred_loss2 = pred_loss(preds[1].mean(axis=-1).unsqueeze(-1), labels)
            return pred_loss1 + pred_loss2 + dist_coff * dist

    def _create_ffn(self):
        """Creates feed-forward network for the finetune task"""
        if self.features_only:
            first_linear_dim = self.features_size + self.features_dim
        else:
            if self.self_attention:
                first_linear_dim = self.hidden_size * self.attn_out_size
                # Also adding features, this is optional
                first_linear_dim += self.features_dim
            else:
                first_linear_dim = self.hidden_size + self.features_dim
        dropout = nn.Dropout(self.dropout)

        if self.activation == 'relu':
            activation = nn.ReLU()

        ffn = [dropout, nn.Linear(first_linear_dim, self.hidden_size)]
        for i in range(self.ffn_num_layers - 1):
            ffn.extend([
                activation, dropout,
                nn.Linear(self.hidden_size, self.hidden_size)
            ])

        return nn.Sequential(*ffn)

    @staticmethod
    def atom_vocab_random_mask(atom_vocab: GroverAtomVocabularyBuilder,
                               smiles: List[str]) -> List[int]:
        """Random masking of atom labels from vocabulary

        For every atom in the list of SMILES string, the algorithm fetches the atoms
        context (vocab label) from the vocabulary provided and returns the vocabulary
        labels with a random masking (probability of masking = 0.15).

        Parameters
        ----------
        atom_vocab: GroverAtomVocabularyBuilder
            atom vocabulary
        smiles: List[str]
            a list of smiles string

        Returns
        -------
        vocab_label: List[int]
            atom vocab label with random masking

        Example
        -------
        >>> import deepchem as dc
        >>> from deepchem.models.torch_models.grover import GroverModel
        >>> from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder
        >>> smiles = np.array(['CC', 'CCC'])
        >>> dataset = dc.data.NumpyDataset(X=smiles)
        >>> atom_vocab = GroverAtomVocabularyBuilder()
        >>> atom_vocab.build(dataset)
        >>> vocab_labels = GroverModel.atom_vocab_random_mask(atom_vocab, smiles)
        """
        vocab_label = []
        percent = 0.15
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            mlabel = [0] * mol.GetNumAtoms()
            n_mask = math.ceil(mol.GetNumAtoms() * percent)
            perm = np.random.permutation(mol.GetNumAtoms())[:n_mask]
            for p in perm:
                atom = mol.GetAtomWithIdx(int(p))
                mlabel[p] = atom_vocab.stoi.get(
                    GroverAtomVocabularyBuilder.atom_to_vocab(mol, atom),
                    atom_vocab.other_index)

            vocab_label.extend(mlabel)
        return vocab_label

    @staticmethod
    def bond_vocab_random_mask(bond_vocab: GroverBondVocabularyBuilder,
                               smiles: List[str]) -> List[int]:
        """Random masking of bond labels from bond vocabulary

        For every bond in the list of SMILES string, the algorithm fetches the bond
        context (vocab label) from the vocabulary provided and returns the vocabulary
        labels with a random masking (probability of masking = 0.15).

        Parameters
        ----------
        bond_vocab: GroverBondVocabularyBuilder
            bond vocabulary
        smiles: List[str]
            a list of smiles string

        Returns
        -------
        vocab_label: List[int]
            bond vocab label with random masking

        Example
        -------
        >>> import deepchem as dc
        >>> from deepchem.models.torch_models.grover import GroverModel
        >>> from deepchem.feat.vocabulary_builders import GroverBondVocabularyBuilder
        >>> smiles = np.array(['CC', 'CCC'])
        >>> dataset = dc.data.NumpyDataset(X=smiles)
        >>> bond_vocab = GroverBondVocabularyBuilder()
        >>> bond_vocab.build(dataset)
        >>> vocab_labels = GroverModel.bond_vocab_random_mask(bond_vocab, smiles)
        """
        vocab_label = []
        percent = 0.15
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            nm_atoms = mol.GetNumAtoms()
            nm_bonds = mol.GetNumBonds()
            mlabel = []
            n_mask = math.ceil(nm_bonds * percent)
            perm = np.random.permutation(nm_bonds)[:n_mask]
            virtual_bond_id = 0
            for a1 in range(nm_atoms):
                for a2 in range(a1 + 1, nm_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue
                    if virtual_bond_id in perm:
                        label = bond_vocab.stoi.get(
                            GroverBondVocabularyBuilder.bond_to_vocab(
                                mol, bond), bond_vocab.other_index)
                        mlabel.extend([label])
                    else:
                        mlabel.extend([0])

                    virtual_bond_id += 1
            vocab_label.extend(mlabel)
        return vocab_label

    def restore(  # type: ignore
            self,
            checkpoint: Optional[str] = None,
            model_dir: Optional[str] = None) -> None:  # type: ignore
        """Reload the values of all variables from a checkpoint file.

        Parameters
        ----------
        checkpoint: str
            the path to the checkpoint file to load.  If this is None, the most recent
            checkpoint will be chosen automatically.  Call get_checkpoints() to get a
            list of all available checkpoints.
        model_dir: str, default None
            Directory to restore checkpoint from. If None, use self.model_dir.  If
            checkpoint is not None, this is ignored.
        """
        # FIXME I am rewriting restore because the restore method in parent class
        # does not restore layers which are not components. This restore method
        # can restore an full model.
        self._ensure_built()
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if len(checkpoints) == 0:
                raise ValueError('No checkpoint found')
            checkpoint = checkpoints[0]
        data = torch.load(checkpoint)
        self.model.load_state_dict(data['model'])
        self._pytorch_optimizer.load_state_dict(data['optimizer_state_dict'])
        self._global_step = data['global_step']
