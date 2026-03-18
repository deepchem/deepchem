import os
import numpy as np

import deepchem as dc


def test_DAG_transformer():
    """Tests the DAG transformer."""
    np.random.seed(123)

    # Load mini log-solubility dataset.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    featurizer = dc.feat.ConvMolFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(
        current_dir, "../../models/tests/assets/example_regression.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    transformer = dc.trans.DAGTransformer(max_atoms=50)
    dataset = transformer.transform(dataset)
    # The transformer generates n DAGs for a molecule with n
    # atoms. These are denoted the "parents"
    for idm, mol in enumerate(dataset.X):
        assert dataset.X[idm].get_num_atoms() == len(dataset.X[idm].parents)
