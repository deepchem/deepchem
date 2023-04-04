import os
import pytest
import numpy as np
import deepchem as dc
from deepchem.feat.molecule_featurizers import SNAPFeaturizer


def get_regression_dataset():
    np.random.seed(123)
    featurizer = SNAPFeaturizer()
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/example_regression.csv')
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")

    return dataset, metric


@pytest.mark.torch
def test_GNN_context_pred():
    # import torch
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, metric = get_regression_dataset()
    model = GNNModular("gin", 3, 64, 1, "attention", 0, "concat", "edge_pred")
    model.fit(dataset, nb_epoch=1)


test_GNN_context_pred()
