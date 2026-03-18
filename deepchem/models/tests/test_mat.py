import pytest
import tempfile
import numpy as np
import deepchem as dc


@pytest.mark.torch
def test_mat_regression():
    # load datasets
    task, datasets, trans = dc.molnet.load_freesolv()
    train, valid, test = datasets

    # initialize model
    model = dc.models.torch_models.MATModel(n_encoders=2,
                                            sa_hsize=128,
                                            d_input=128,
                                            d_hidden=128,
                                            d_output=128,
                                            encoder_hsize=128,
                                            embed_input_hsize=36,
                                            gen_attn_hidden=32)
    # overfit test
    model.fit(valid, nb_epoch=100)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")
    scores = model.evaluate(valid, [metric], trans)
    assert scores['mean_absolute_error'] < 1.0


@pytest.mark.torch
def test_mat_reload():
    from deepchem.models.torch_models import MATModel
    model_dir = tempfile.mkdtemp()

    tasks, datasets, trans = dc.molnet.load_freesolv()
    train, valid, test = datasets
    model = MATModel(n_encoders=2,
                     sa_hsize=128,
                     d_input=128,
                     d_hidden=128,
                     d_output=128,
                     encoder_hsize=128,
                     embed_input_hsize=36,
                     gen_attn_hidden=32,
                     model_dir=model_dir)
    model.fit(train, nb_epoch=1)

    reloaded_model = MATModel(n_encoders=2,
                              sa_hsize=128,
                              d_input=128,
                              d_hidden=128,
                              d_output=128,
                              encoder_hsize=128,
                              embed_input_hsize=36,
                              gen_attn_hidden=32,
                              model_dir=model_dir)
    reloaded_model.restore()

    original_pred = model.predict(valid)
    reload_pred = reloaded_model.predict(valid)
    assert np.all(original_pred == reload_pred)
