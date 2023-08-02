import os
import numpy as np
import pytest

from deepchem.data import SDFLoader
from deepchem.feat import CoulombMatrix

try:
    from deepchem.models.torch_models import DTNNModel
except ModuleNotFoundError:
    pass


@pytest.mark.torch
def test_dtnn():
    """Tests DTNN Model for Shape and prediction.

    - Used dataset files: qm9_mini.sdf, qm9_mini.sdf.csv (A subset of qm9 dataset.)
    - Tasks selected are only of regression type.

    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(current_dir, "assets/qm9_mini.sdf")
    TASKS = ["alpha", "homo"]
    loader = SDFLoader(tasks=TASKS, featurizer=CoulombMatrix(29), sanitize=True)
    data = loader.create_dataset(dataset_file, shard_size=100)

    model = DTNNModel(data.y.shape[1],
                      n_embedding=40,
                      n_distance=100,
                      learning_rate=0.7,
                      mode="regression")
    model.fit(data, nb_epoch=1000)

    # Eval model on train
    pred = model.predict(data)

    mean_rel_error = np.mean(np.abs(1 - pred / (data.y)))

    assert mean_rel_error < 0.15
    assert pred.shape == data.y.shape
