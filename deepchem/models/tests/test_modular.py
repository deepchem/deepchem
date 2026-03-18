import pytest
import deepchem as dc
import numpy as np
try:
    from deepchem.models.torch_models.modular import ModularTorchModel
    import torch
    import torch.nn as nn

    class ExampleTorchModel(ModularTorchModel):
        """Example TorchModel for testing pretraining."""

        def __init__(self, input_dim, d_hidden, n_layers, d_output, **kwargs):
            self.input_dim = input_dim
            self.d_hidden = d_hidden
            self.n_layers = n_layers
            self.d_output = d_output
            self.components = self.build_components()
            self.model = self.build_model()
            super().__init__(self.model, self.components, **kwargs)

        def build_components(self):
            return {
                'encoder': self.encoder(),
                'FF1': self.FF1(),
                'FF2': self.FF2()
            }

        def loss_func(self, inputs, labels, weights):
            preds1 = self.components['FF2'](self.components['encoder'](inputs))
            labels = labels[0]
            loss1 = torch.nn.functional.mse_loss(preds1, labels)

            preds2 = self.components['FF1'](inputs)
            loss2 = torch.nn.functional.smooth_l1_loss(preds2, labels)
            total_loss = loss1 + loss2
            return (total_loss * weights[0]).mean()

        def encoder(self):
            embedding = []
            for i in range(self.n_layers):
                if i == 0:
                    embedding.append(nn.Linear(self.input_dim, self.d_hidden))
                    embedding.append(nn.ReLU())
                else:
                    embedding.append(nn.Linear(self.d_hidden, self.d_hidden))
                    embedding.append(nn.ReLU())
            return nn.Sequential(*embedding)

        def FF1(self):
            linear = nn.Linear(self.input_dim, self.d_output)
            af = nn.Sigmoid()
            return nn.Sequential(linear, af)

        def FF2(self):
            linear = nn.Linear(self.d_hidden, self.d_output)
            af = nn.ReLU()
            return nn.Sequential(linear, af)

        def build_model(self):
            return nn.Sequential(self.components['encoder'],
                                 self.components['FF2'])

    class ExamplePretrainer(ModularTorchModel):

        def __init__(self, model, pt_tasks, **kwargs):
            self.source_model = model  # the pretrainer takes the original model as input in order to modify it
            self.pt_tasks = pt_tasks
            self.components = self.build_components()
            self.model = self.build_model()
            super().__init__(self.model, self.components, **kwargs)

        def FF_pt(self):
            linear = nn.Linear(self.source_model.d_hidden, self.pt_tasks)
            af = nn.ReLU()
            return nn.Sequential(linear, af)

        def loss_func(self, inputs, labels, weights):
            inputs = inputs[0]
            labels = labels[0]
            weights = weights[0]
            preds = self.components['FF_pt'](self.components['encoder'](inputs))
            loss = torch.nn.functional.mse_loss(preds, labels)
            loss = loss * weights
            loss = loss.mean()
            return loss

        def build_components(self):
            pt_components = self.source_model.build_components()
            pt_components.update({'FF_pt': self.FF_pt()})
            return pt_components

        def build_model(self):
            return nn.Sequential(self.components['encoder'],
                                 self.components['FF_pt'])
except:
    pass


@pytest.mark.torch
def test_overfit_modular():
    """Overfit test the pretrainer to ensure it can learn a simple task."""
    np.random.seed(123)
    torch.manual_seed(10)
    n_samples = 6
    n_feat = 3
    d_hidden = 3
    n_layers = 1
    n_tasks = 6

    X = np.random.rand(n_samples, n_feat)
    y = np.zeros((n_samples, n_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)

    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)

    example_model.fit(dataset, nb_epoch=1000)
    prediction = np.round(np.squeeze(example_model.predict_on_batch(X)))
    assert np.array_equal(y, prediction)


@pytest.mark.torch
def test_fit_restore():
    """Test that the pretrainer can be restored and continue training."""
    np.random.seed(123)
    torch.manual_seed(10)
    n_samples = 6
    n_feat = 3
    d_hidden = 3
    n_layers = 1
    n_tasks = 6

    X = np.random.rand(n_samples, n_feat)
    y = np.zeros((n_samples, n_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)

    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    example_model.fit(dataset, nb_epoch=1000)

    # Create an identical model, do a single step of fitting with restore=True and make sure it got restored correctly.
    example_model2 = ExampleTorchModel(n_feat,
                                       d_hidden,
                                       n_layers,
                                       n_tasks,
                                       model_dir=example_model.model_dir)
    example_model2.fit(dataset, nb_epoch=1, restore=True)

    prediction = np.squeeze(example_model2.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))


@pytest.mark.torch
def test_load_freeze_unfreeze():
    np.random.seed(123)
    torch.manual_seed(10)
    n_samples = 60
    n_feat = 3
    d_hidden = 3
    n_layers = 1
    ft_tasks = 6
    pt_tasks = 6

    X_ft = np.random.rand(n_samples, n_feat)
    y_ft = np.random.rand(n_samples, ft_tasks).astype(np.float32)
    dataset_ft = dc.data.NumpyDataset(X_ft, y_ft)

    X_ft2 = np.random.rand(n_samples, n_feat)
    y_ft2 = np.zeros((n_samples, ft_tasks)).astype(np.float32)
    dataset_ft2 = dc.data.NumpyDataset(X_ft2, y_ft2)

    X_pt = np.random.rand(n_samples, n_feat)
    y_pt = np.random.rand(n_samples, pt_tasks).astype(np.float32)
    dataset_pt = dc.data.NumpyDataset(X_pt, y_pt)

    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers, ft_tasks)
    example_pretrainer = ExamplePretrainer(example_model, pt_tasks)

    example_pretrainer.fit(dataset_pt, nb_epoch=1000)

    example_model.load_from_pretrained(model_dir=example_pretrainer.model_dir,
                                       components=['encoder'])

    example_model.freeze_components(['encoder'])

    example_model.fit(dataset_ft, nb_epoch=100)

    # check that the first layer is still the same between the two models
    assert np.array_equal(
        example_pretrainer.components['encoder'][0].weight.data.cpu().numpy(),
        example_model.components['encoder'][0].weight.data.cpu().numpy())

    # check that the predictions are different because of the fine tuning
    assert not np.array_equal(
        np.round(np.squeeze(example_pretrainer.predict_on_batch(X_ft))),
        np.round(np.squeeze(example_model.predict_on_batch(X_ft))))

    example_model.unfreeze_components(['encoder'])

    example_model.fit(dataset_ft2, nb_epoch=100)

    # check that the first layer is different between the two models
    assert not np.array_equal(
        example_pretrainer.components['encoder'][0].weight.data.cpu().numpy(),
        example_model.components['encoder'][0].weight.data.cpu().numpy())
