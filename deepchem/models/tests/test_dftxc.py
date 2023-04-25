import deepchem as dc
from deepchem.models.dft.dftxc import XCModel
from deepchem.data.data_loader import DFTYamlLoader
import pytest
import numpy as np
import tempfile


@pytest.mark.dqc
def test_dftxc_eval():
    inputs = 'deepchem/models/tests/assets/test_dftxcdata.yaml'
    data = DFTYamlLoader()
    model_dir = tempfile.mkdtemp()
    dataset = (data.create_dataset(inputs))
    model = XCModel("lda_x",
                    batch_size=1,
                    log_frequency=1,
                    mode="classification",
                    n_tasks=2,
                    model_dir=model_dir)
    loss = model.fit(dataset, nb_epoch=1, checkpoint_interval=1)
    assert loss < 0.001
    reload_model = XCModel("lda_x",
                           batch_size=1,
                           log_frequency=1,
                           mode="classification",
                           n_tasks=2,
                           model_dir=model_dir)
    reload_model.restore()
    predict = model.predict(dataset)
    r_predict = reload_model.predict(dataset)
    assert np.all(predict == r_predict)
    metric = dc.metrics.Metric(dc.metrics.mae_score)
    scores = model.evaluate(dataset, [metric])
    assert scores['mae_score'] < 0.3
    model2 = XCModel("lda_x",
                     batch_size=2,
                     log_frequency=1,
                     mode="classification")
    loss2 = model2.fit(dataset, nb_epoch=2, checkpoint_interval=2)
    assert loss2 < 0.2


def test_dm_predict():
    inputs = 'deepchem/models/tests/assets/test_dm.yaml'
    data = DFTYamlLoader()
    dataset = (data.create_dataset(inputs))
    model = XCModel("lda_x", batch_size=1)
    loss = model.fit(dataset, nb_epoch=1, checkpoint_interval=1)
    assert loss < 0.008
    predict = model.predict(dataset)
    assert predict.shape == (57, 57)
