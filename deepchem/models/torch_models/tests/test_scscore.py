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

    assert len(pred) == len(
        reloaded_pred
    ), "Number of reloaded predictions do not match original predictions"
    assert np.allclose(
        pred, reloaded_pred,
        atol=1e-04), "Predictions do not match reloaded predictions"


@pytest.mark.torch
def test_loaded_pretrained_scscore():
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

    pretrained_model = ScScoreModel(n_features,
                                    layer_sizes,
                                    dropout=0.0,
                                    score_scale=5,
                                    model_dir=model_dir)
    pretrained_model.load_from_pretrained(source_model=model,
                                          model_dir=model_dir)

    pred = model.predict(np_dataset)
    pretrained_pred = pretrained_model.predict(np_dataset)

    assert len(pred) == len(
        pretrained_pred
    ), "Number of pretrained predictions do not match original predictions"
    assert np.allclose(
        pred, pretrained_pred,
        atol=1e-04), "Predictions do not match pretrained predictions"


@pytest.mark.torch
def test_scscore_dropout_respects_eval_mode():
    """ScScore.forward() must disable dropout when the module is in eval mode.

    Regression test for F.dropout() being called without
    training=self.training, which left dropout active even after
    model.eval() was called.
    """
    import torch
    from deepchem.models.torch_models.scscore import ScScore

    torch.manual_seed(0)
    model = ScScore(n_features=32, layer_sizes=[64, 64], dropout=0.5)
    x = torch.rand(4, 32)

    # Eval mode: dropout must be disabled, so repeated forward passes on
    # identical input must be exactly reproducible. This does not depend
    # on random seeding, since with dropout off the forward pass is a
    # deterministic composition of linear/relu/sigmoid ops.
    model.eval()
    eval_out1 = model(x)
    eval_out2 = model(x)
    assert torch.equal(eval_out1, eval_out2), (
        "ScScore produced different outputs for identical input while in "
        "eval() mode; dropout is not being disabled during evaluation.")

    # Train mode: dropout must remain active. Two forward passes seeded
    # differently should not produce identical output. With dropout=0.5
    # over 64-wide hidden layers, the probability of two independently
    # sampled dropout masks coinciding is 2**-64, so this is not a flaky
    # check, and both seeds are fixed for reproducibility across runs.
    model.train()
    torch.manual_seed(1)
    train_out1 = model(x)
    torch.manual_seed(2)
    train_out2 = model(x)
    assert not torch.equal(train_out1, train_out2), (
        "ScScore produced identical outputs across differently-seeded "
        "forward passes in train() mode; dropout appears inactive during "
        "training.")
