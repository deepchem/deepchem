import pytest


@pytest.fixture
def grover_graph_attributes():
    import deepchem as dc
    from deepchem.feat.graph_data import BatchGraphData
    from deepchem.utils.grover import extract_grover_attributes
    smiles = ['CC', 'CCC', 'CC(=O)C']

    fg = dc.feat.CircularFingerprint()
    featurizer = dc.feat.GroverFeaturizer(features_generator=fg)

    graphs = featurizer.featurize(smiles)
    batched_graph = BatchGraphData(graphs)
    attributes = extract_grover_attributes(batched_graph)
    return attributes
