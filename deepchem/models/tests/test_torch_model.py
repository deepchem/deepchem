import os
import pytest
import deepchem as dc
import numpy as np
import math
import unittest

try:
    import torch
    import torch.nn.functional as F
    has_pytorch = True
except:
    has_pytorch = False

try:
    import wandb  # noqa: F401
    has_wandb = True
except:
    has_wandb = False


@pytest.mark.torch
def test_overfit_subclass_model():
    """Test fitting a TorchModel defined by subclassing Module."""
    n_data_points = 10
    n_features = 2
    np.random.seed(1234)
    X = np.random.rand(n_data_points, n_features)
    y = (X[:, 0] > X[:, 1]).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)

    class ExampleModel(torch.nn.Module):

        def __init__(self, layer_sizes):
            super(ExampleModel, self).__init__()
            self.layers = torch.nn.ModuleList()
            in_size = n_features
            for out_size in layer_sizes:
                self.layers.append(torch.nn.Linear(in_size, out_size))
                in_size = out_size

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = F.relu(x)
            return torch.sigmoid(x), x

    pytorch_model = ExampleModel([10, 1])
    model = dc.models.TorchModel(pytorch_model,
                                 dc.models.losses.SigmoidCrossEntropy(),
                                 output_types=['prediction', 'loss'],
                                 learning_rate=0.005)
    model.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(model.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    scores = model.evaluate(dataset, [metric])
    assert scores[metric.name] > 0.9


@pytest.mark.torch
def test_overfit_sequential_model():
    """Test fitting a TorchModel defined as a sequential model."""
    n_data_points = 10
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = (X[:, 0] > X[:, 1]).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    pytorch_model = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.ReLU(),
                                        torch.nn.Linear(10, 1),
                                        torch.nn.Sigmoid())
    model = dc.models.TorchModel(pytorch_model,
                                 dc.models.losses.BinaryCrossEntropy(),
                                 learning_rate=0.005)
    model.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(model.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    generator = model.default_generator(dataset, pad_batches=False)
    scores = model.evaluate_generator(generator, [metric])
    assert scores[metric.name] > 0.9


@pytest.mark.torch
def test_fit_use_all_losses():
    """Test fitting a TorchModel and getting a loss curve back."""
    n_data_points = 10
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = (X[:, 0] > X[:, 1]).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    pytorch_model = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.ReLU(),
                                        torch.nn.Linear(10, 1),
                                        torch.nn.Sigmoid())
    model = dc.models.TorchModel(pytorch_model,
                                 dc.models.losses.BinaryCrossEntropy(),
                                 learning_rate=0.005,
                                 log_frequency=10)
    losses = []
    model.fit(dataset, nb_epoch=1000, all_losses=losses)
    # Each epoch is a single step for this model
    assert len(losses) == 100
    assert np.count_nonzero(np.array(losses)) == 100


@pytest.mark.torch
def test_fit_on_batch():
    """Test fitting a TorchModel to individual batches."""
    n_data_points = 10
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = (X[:, 0] > X[:, 1]).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    pytorch_model = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.ReLU(),
                                        torch.nn.Linear(10, 1),
                                        torch.nn.Sigmoid())
    model = dc.models.TorchModel(pytorch_model,
                                 dc.models.losses.BinaryCrossEntropy(),
                                 learning_rate=0.005)
    i = 0
    for X, y, w, ids in dataset.iterbatches(model.batch_size, 500):
        i += 1
        model.fit_on_batch(X, y, w, checkpoint=False)
    prediction = np.squeeze(model.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    generator = model.default_generator(dataset, pad_batches=False)
    scores = model.evaluate_generator(generator, [metric])
    assert scores[metric.name] > 0.9


@pytest.mark.torch
def test_checkpointing():
    """Test loading and saving checkpoints with TorchModel."""
    # Create two models using the same model directory.

    pytorch_model1 = torch.nn.Sequential(torch.nn.Linear(5, 10))
    pytorch_model2 = torch.nn.Sequential(torch.nn.Linear(5, 10))
    model1 = dc.models.TorchModel(pytorch_model1, dc.models.losses.L2Loss())
    model2 = dc.models.TorchModel(pytorch_model2,
                                  dc.models.losses.L2Loss(),
                                  model_dir=model1.model_dir)

    # Check that they produce different results.

    X = np.random.rand(5, 5)
    y1 = model1.predict_on_batch(X)
    y2 = model2.predict_on_batch(X)
    assert not np.array_equal(y1, y2)

    # Save a checkpoint from the first model and load it into the second one,
    # and make sure they now match.

    model1.save_checkpoint()
    model2.restore()
    y3 = model1.predict_on_batch(X)
    y4 = model2.predict_on_batch(X)
    assert np.array_equal(y1, y3)
    assert np.array_equal(y1, y4)


@pytest.mark.torch
def test_fit_restore():
    """Test specifying restore=True when calling fit()."""
    n_data_points = 10
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = (X[:, 0] > X[:, 1]).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)

    # Train a model to overfit the dataset.

    pytorch_model = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.ReLU(),
                                        torch.nn.Linear(10, 1),
                                        torch.nn.Sigmoid())
    model = dc.models.TorchModel(pytorch_model,
                                 dc.models.losses.BinaryCrossEntropy(),
                                 learning_rate=0.005)
    model.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(model.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))

    # Create an identical model, do a single step of fitting with restore=True,
    # and make sure it got restored correctly.

    pytorch_model2 = torch.nn.Sequential(torch.nn.Linear(2,
                                                         10), torch.nn.ReLU(),
                                         torch.nn.Linear(10, 1),
                                         torch.nn.Sigmoid())
    model2 = dc.models.TorchModel(pytorch_model2,
                                  dc.models.losses.BinaryCrossEntropy(),
                                  model_dir=model.model_dir)
    model2.fit(dataset, nb_epoch=1, restore=True)
    prediction = np.squeeze(model2.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))


@pytest.mark.torch
def test_uncertainty():
    """Test estimating uncertainty a TorchModel."""
    n_samples = 30
    n_features = 1
    noise = 0.1
    X = np.random.rand(n_samples, n_features)
    y = (10 * X + np.random.normal(scale=noise, size=(n_samples, n_features)))
    dataset = dc.data.NumpyDataset(X, y)

    # Build a model that predicts uncertainty.

    class PyTorchUncertainty(torch.nn.Module):

        def __init__(self):
            super(PyTorchUncertainty, self).__init__()
            self.hidden = torch.nn.Linear(n_features, 200)
            self.output = torch.nn.Linear(200, n_features)
            self.log_var = torch.nn.Linear(200, n_features)

        def forward(self, inputs):
            x, use_dropout = inputs
            x = self.hidden(x)
            if use_dropout:
                x = F.dropout(x, 0.1)
            output = self.output(x)
            log_var = self.log_var(x)
            var = torch.exp(log_var)
            return (output, var, output, log_var)

    def loss(outputs, labels, weights):
        diff = labels[0] - outputs[0]
        log_var = outputs[1]
        var = torch.exp(log_var)
        return torch.mean(diff * diff / var + log_var)

    class UncertaintyModel(dc.models.TorchModel):

        def default_generator(self,
                              dataset,
                              epochs=1,
                              mode='fit',
                              deterministic=True,
                              pad_batches=True):
            for epoch in range(epochs):
                for (X_b, y_b, w_b,
                     ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                                   deterministic=deterministic,
                                                   pad_batches=pad_batches):
                    if mode == 'predict':
                        dropout = np.array(False)
                    else:
                        dropout = np.array(True)
                    yield ([X_b, dropout], [y_b], [w_b])

    pytorch_model = PyTorchUncertainty()
    model = UncertaintyModel(
        pytorch_model,
        loss,
        output_types=['prediction', 'variance', 'loss', 'loss'],
        learning_rate=0.003)

    # Fit the model and see if its predictions are correct.

    model.fit(dataset, nb_epoch=2500)
    pred, std = model.predict_uncertainty(dataset)
    assert np.mean(np.abs(y - pred)) < 1.0
    assert noise < np.mean(std) < 1.0


@pytest.mark.torch
def test_saliency_mapping():
    """Test computing a saliency map."""
    n_tasks = 3
    n_features = 5
    pytorch_model = torch.nn.Sequential(torch.nn.Linear(n_features, 20),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(20, n_tasks))
    model = dc.models.TorchModel(pytorch_model, dc.models.losses.L2Loss())
    x = np.random.random(n_features)
    s = model.compute_saliency(x)
    assert s.shape[0] == n_tasks
    assert s.shape[1] == n_features

    # Take a tiny step in the direction of s and see if the output changes by
    # the expected amount.

    delta = 0.01
    for task in range(n_tasks):
        norm = np.sqrt(np.sum(s[task]**2))
        step = 0.5 * delta / norm
        pred1 = model.predict_on_batch((x + s[task] * step).reshape(
            (1, n_features))).flatten()
        pred2 = model.predict_on_batch((x - s[task] * step).reshape(
            (1, n_features))).flatten()
        assert np.allclose(pred1[task], (pred2 + norm * delta)[task], atol=1e-6)


@pytest.mark.torch
def test_saliency_shapes():
    """Test computing saliency maps for multiple outputs with multiple dimensions."""

    class SaliencyModel(torch.nn.Module):

        def __init__(self):
            super(SaliencyModel, self).__init__()
            self.layer1 = torch.nn.Linear(6, 4)
            self.layer2 = torch.nn.Linear(6, 5)

        def forward(self, x):
            x = torch.flatten(x)
            output1 = self.layer1(x).reshape(1, 4, 1)
            output2 = self.layer2(x).reshape(1, 1, 5)
            return output1, output2

    pytorch_model = SaliencyModel()
    model = dc.models.TorchModel(pytorch_model, dc.models.losses.L2Loss())
    x = np.random.random((2, 3))
    s = model.compute_saliency(x)
    assert len(s) == 2
    assert s[0].shape == (4, 1, 2, 3)
    assert s[1].shape == (1, 5, 2, 3)


@pytest.mark.torch
def test_tensorboard():
    """Test logging to Tensorboard."""
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0.0, 1.0] for x in range(n_data_points)]
    dataset = dc.data.NumpyDataset(X, y)
    pytorch_model = torch.nn.Sequential(torch.nn.Linear(n_features, 2),
                                        torch.nn.Softmax(dim=1))
    model = dc.models.TorchModel(pytorch_model,
                                 dc.models.losses.CategoricalCrossEntropy(),
                                 tensorboard=True,
                                 log_frequency=1)
    model.fit(dataset, nb_epoch=10)
    files_in_dir = os.listdir(model.model_dir)
    event_file = list(filter(lambda x: x.startswith("events"), files_in_dir))
    assert len(event_file) > 0
    event_file = os.path.join(model.model_dir, event_file[0])
    file_size = os.stat(event_file).st_size
    assert file_size > 0


@pytest.mark.torch
@unittest.skipIf((not has_pytorch) or (not has_wandb),
                 'PyTorch and/or Wandb is not installed')
def test_wandblogger():
    """Test logging to Weights & Biases."""
    # Load dataset and Models
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP',
                                                           splitter='random')
    train_dataset, valid_dataset, test_dataset = datasets
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    wandblogger = dc.models.WandbLogger(anonymous="allow",
                                        save_run_history=True)

    pytorch_model = torch.nn.Sequential(torch.nn.Linear(1024, 1000),
                                        torch.nn.Dropout(p=0.5),
                                        torch.nn.Linear(1000, 1))
    model = dc.models.TorchModel(pytorch_model,
                                 dc.models.losses.L2Loss(),
                                 wandb_logger=wandblogger)
    vc_train = dc.models.ValidationCallback(train_dataset, 1, [metric])
    vc_valid = dc.models.ValidationCallback(valid_dataset, 1, [metric])
    model.fit(train_dataset, nb_epoch=10, callbacks=[vc_train, vc_valid])
    # call model.fit again to test multiple fit() calls
    model.fit(train_dataset, nb_epoch=10, callbacks=[vc_train, vc_valid])
    wandblogger.finish()

    run_data = wandblogger.run_history
    valid_score = model.evaluate(valid_dataset, [metric], transformers)

    assert math.isclose(valid_score["pearson_r2_score"],
                        run_data['eval/pearson_r2_score_(1)'],
                        abs_tol=0.0005)


@pytest.mark.torch
def test_fit_variables():
    """Test training a subset of the variables in a model."""

    class VarModel(torch.nn.Module):

        def __init__(self, **kwargs):
            super(VarModel, self).__init__(**kwargs)
            self.var1 = torch.nn.Parameter(torch.Tensor([0.5]))
            self.var2 = torch.nn.Parameter(torch.Tensor([0.5]))

        def forward(self, inputs):
            return [self.var1, self.var2]

    def loss(outputs, labels, weights):
        return (outputs[0] * outputs[1] - labels[0])**2

    pytorch_model = VarModel()
    model = dc.models.TorchModel(pytorch_model, loss, learning_rate=0.02)
    x = np.ones((1, 1))
    vars = model.predict_on_batch(x)
    assert np.allclose(vars[0], 0.5)
    assert np.allclose(vars[1], 0.5)
    model.fit_generator([(x, x, x)] * 300)
    vars = model.predict_on_batch(x)
    assert np.allclose(vars[0], 1.0)
    assert np.allclose(vars[1], 1.0)
    model.fit_generator([(x, 2 * x, x)] * 300, variables=[pytorch_model.var1])
    vars = model.predict_on_batch(x)
    assert np.allclose(vars[0], 2.0)
    assert np.allclose(vars[1], 1.0)
    model.fit_generator([(x, x, x)] * 300, variables=[pytorch_model.var2])
    vars = model.predict_on_batch(x)
    assert np.allclose(vars[0], 2.0)
    assert np.allclose(vars[1], 0.5)


@pytest.mark.torch
def test_fit_loss():
    """Test specifying a different loss function when calling fit()."""

    class VarModel(torch.nn.Module):

        def __init__(self):
            super(VarModel, self).__init__()
            self.var1 = torch.nn.Parameter(torch.Tensor([0.5]))
            self.var2 = torch.nn.Parameter(torch.Tensor([0.5]))

        def forward(self, inputs):
            return [self.var1, self.var2]

    def loss1(outputs, labels, weights):
        return (outputs[0] * outputs[1] - labels[0])**2

    def loss2(outputs, labels, weights):
        return (outputs[0] + outputs[1] - labels[0])**2

    pytorch_model = VarModel()
    model = dc.models.TorchModel(pytorch_model, loss1, learning_rate=0.01)
    x = np.ones((1, 1))
    vars = model.predict_on_batch(x)
    assert np.allclose(vars[0], 0.5)
    assert np.allclose(vars[1], 0.5)
    model.fit_generator([(x, x, x)] * 300)
    vars = model.predict_on_batch(x)
    assert np.allclose(vars[0], 1.0)
    assert np.allclose(vars[1], 1.0)
    model.fit_generator([(x, 3 * x, x)] * 300, loss=loss2)
    vars = model.predict_on_batch(x)
    assert np.allclose(vars[0] + vars[1], 3.0)


@pytest.mark.torch
def test_fit_generator_parameters():
    """Test confirming that fit_generator accepts variables both as a generator object and as an iteable object like list or tuple."""

    from typing import Generator

    class Model(torch.nn.Module):

        def __init__(self):
            super(Model, self).__init__()
            self.dense1 = torch.nn.Linear(100, 50)
            self.dense2 = torch.nn.Linear(50, 1)

        def forward(self, x):
            x = torch.nn.functional.relu(self.dense1(x))
            x = self.dense2(x)
            return x

    model1 = Model()
    model2 = Model()
    model2.load_state_dict(model1.state_dict())

    torch_model1 = dc.models.TorchModel(model1,
                                        dc.models.losses.L2Loss(),
                                        output_types=['prediction'])
    torch_model2 = dc.models.TorchModel(model2,
                                        dc.models.losses.L2Loss(),
                                        output_types=['prediction'])

    variables_generator = torch_model1.model.parameters()
    variables_list = list(torch_model2.model.parameters())

    assert isinstance(variables_generator, Generator)

    X = np.random.rand(10, 100)
    y = np.random.rand(10, 1)

    dataset = dc.data.NumpyDataset(X, y)

    # fit both models with the same dataset
    torch_model1.fit_generator(torch_model1.default_generator(dataset),
                               variables=variables_generator)
    torch_model2.fit_generator(torch_model2.default_generator(dataset),
                               variables=variables_list)

    # check if the models are the same
    X_ = np.random.rand(10, 100)

    prediction1 = torch_model1.predict_on_batch(X_)
    prediction2 = torch_model2.predict_on_batch(X_)

    assert np.allclose(prediction1, prediction2)


@pytest.mark.torch
def test_torch_compile():
    """Test compiling a TorchModel."""

    n_samples = 16
    n_features = 32
    n_tasks = 1
    n_classes = 2

    X = np.random.rand(n_samples, 32, n_features)
    y = np.random.randint(n_classes,
                          size=(n_samples, n_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)

    model = dc.models.torch_models.CNN(n_tasks,
                                       n_features,
                                       dims=1,
                                       kernel_size=3,
                                       mode='classification')
    model.compile(mode='default', backend='inductor')
    model.fit(dataset, nb_epoch=500)

    model_output = model.predict_on_batch(X)
    model_output = np.argmax(model_output, axis=2)

    assert np.all(model_output == y)
