import pytest
import tempfile
import numpy as np
import deepchem as dc


@pytest.mark.torch
def test_restore_scscore():
    from deepchem.models.torch_models.scscore import ScScoreModel
    n_features = 1024
    layer_sizes = [300, 300, 300, 300, 300]

    X = np.random.rand(100, n_features).astype(np.float32)
    y = np.random.uniform(1, 5, size=(100)).astype(np.float32)
    np_dataset = dc.data.NumpyDataset(X, y)

    model_dir = tempfile.mkdtemp()
    model = ScScoreModel(n_features,
                         layer_sizes,
                         dropout=0.0,
                         score_scale=5,
                         model_dir=model_dir)
    model.fit(np_dataset, nb_epoch=5)
    pred = model.predict(np_dataset)

    reloaded_model = ScScoreModel(n_features,
                                  layer_sizes,
                                  dropout=0.0,
                                  score_scale=5,
                                  model_dir=model_dir)
    reloaded_model.restore()

    pred = model.predict(np_dataset)
    reloaded_pred = reloaded_model.predict(np_dataset)

    assert len(pred) == len(reloaded_pred)
    assert np.allclose(pred, reloaded_pred, atol=1e-04)
