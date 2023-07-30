import deepchem as dc
try:
    from deepchem.models.dft.dftxc import XCModel
    from deepchem.data.data_loader import DFTYamlLoader
    has_dqc = True
except ModuleNotFoundError:
    has_dqc = False
import pytest
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
                    model_dir=model_dir)
    loss = model.fit(dataset, nb_epoch=2, checkpoint_interval=1)
    assert loss < 0.001
    reload_model = XCModel("lda_x",
                           batch_size=1,
                           log_frequency=1,
                           mode="classification",
                           model_dir=model_dir)
    reload_model.restore()
    inputs1 = 'deepchem/models/tests/assets/test_ieLi.yaml'
    predict_dataset = data.create_dataset(inputs1)
    predict = reload_model.predict(predict_dataset)
    assert predict < 0.199
    metric = dc.metrics.Metric(dc.metrics.mae_score)
    scores = model.evaluate(dataset, [metric])
    assert scores['mae_score'] < 0.3
    # testing batch size > 1
    model2 = XCModel("lda_x",
                     batch_size=2,
                     log_frequency=1,
                     mode="classification")
    loss2 = model2.fit(dataset, nb_epoch=2, checkpoint_interval=2)
    assert loss2 < 0.2
    # testing true values
    assert dataset.y[0] != dataset.y[1]


@pytest.mark.dqc
def test_dm():
    inputs = 'deepchem/models/tests/assets/test_dm.yaml'
    data = DFTYamlLoader()
    dataset = (data.create_dataset(inputs))
    model = XCModel("lda_x", batch_size=1)
    loss = model.fit(dataset, nb_epoch=1, checkpoint_interval=1)
    assert loss < 0.008
