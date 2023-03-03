import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
import pytest
import os


@pytest.mark.torch
def gen_dataset():
    # load sample dataset
    dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(dir, 'assets/example_classification.csv')
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=MolGraphConvFeaturizer())
    dataset = loader.create_dataset(input_file)
    return dataset


@pytest.mark.torch
def test_GINConv():
    from deepchem.models.torch_models.mp_layers import GINConv
    layer = GINConv(emb_dim=10)


@pytest.mark.torch
def test_GCNConv():
    from deepchem.models.torch_models.mp_layers import GCNConv
    layer = GCNConv(emb_dim=10)


@pytest.mark.torch
def test_GATConv():
    from deepchem.models.torch_models.mp_layers import GATConv
    layer = GATConv(emb_dim=10)


@pytest.mark.torch
def test_GraphSAGEConv():
    from deepchem.models.torch_models.mp_layers import GraphSAGEConv
    layer = GraphSAGEConv(emb_dim=10)
