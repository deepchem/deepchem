import unittest
import tempfile

import numpy as np

import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.models import Pagtn, PagtnModel
from deepchem.models.tests.test_graph_models import get_dataset

import deepchem as dc
import dgl

smiles = ["C1CCC1", "C1=CC=CN=C1"]
featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
graphs = featurizer.featurize(smiles)
dgl_graphs = [
    graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))
]
batch_dgl_graph = dgl.batch(dgl_graphs)

model = Pagtn(
    n_tasks=1,
    number_atom_features=30,
    number_bond_features=11,
    ouput_node_features=64)
preds = model(batch_dgl_graph)
print(preds)
