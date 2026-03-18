import unittest
import pytest
import deepchem as dc
import numpy as np

try:
    import torch

    class MLP(dc.models.TorchModel):

        def __init__(self,
                     n_tasks=1,
                     feature_dim=100,
                     hidden_layer_size=64,
                     **kwargs):
            pytorch_model = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, hidden_layer_size),
                torch.nn.ReLU(), torch.nn.Linear(hidden_layer_size, n_tasks),
                torch.nn.Sigmoid())
            loss = dc.models.losses.BinaryCrossEntropy()
            super(MLP, self).__init__(model=pytorch_model, loss=loss, **kwargs)

    has_pytorch = True
except:
    has_pytorch = False


@unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
class TestPretrainedTorch(unittest.TestCase):

    @pytest.mark.torch
    def setUp(self):
        self.feature_dim = 2
        self.hidden_layer_size = 10
        data_points = 10

        X = np.random.randn(data_points, self.feature_dim)
        y = (X[:, 0] > X[:, 1]).astype(np.float32)

        self.dataset = dc.data.NumpyDataset(X, y)

    @pytest.mark.torch
    def test_load_from_pretrained(self):
        """Tests loading pretrained model."""
        source_model = MLP(hidden_layer_size=self.hidden_layer_size,
                           feature_dim=self.feature_dim,
                           batch_size=10)

        source_model.fit(self.dataset, nb_epoch=1000, checkpoint_interval=0)

        dest_model = MLP(feature_dim=self.feature_dim,
                         hidden_layer_size=self.hidden_layer_size,
                         n_tasks=10)

        assignment_map = dict()
        value_map = dict()
        source_vars = list(source_model.model.parameters())
        dest_vars = list(dest_model.model.parameters())[:-2]

        for idx, dest_var in enumerate(dest_vars):
            source_var = source_vars[idx]
            assignment_map[source_var] = dest_var
            value_map[source_var] = source_var.detach().cpu().numpy()

        dest_model.load_from_pretrained(source_model=source_model,
                                        assignment_map=assignment_map,
                                        value_map=value_map)

        for source_var, dest_var in assignment_map.items():
            source_val = source_var.detach().cpu().numpy()
            dest_val = dest_var.detach().cpu().numpy()
            np.testing.assert_array_almost_equal(source_val, dest_val)

    @pytest.mark.torch
    def test_restore_equivalency(self):
        """Test for restore based pretrained model loading."""
        source_model = MLP(feature_dim=self.feature_dim,
                           hidden_layer_size=self.hidden_layer_size,
                           learning_rate=0.003)

        source_model.fit(self.dataset, nb_epoch=1000)

        dest_model = MLP(feature_dim=self.feature_dim,
                         hidden_layer_size=self.hidden_layer_size)

        dest_model.load_from_pretrained(source_model=source_model,
                                        assignment_map=None,
                                        value_map=None,
                                        model_dir=None,
                                        include_top=True)

        predictions = np.squeeze(dest_model.predict_on_batch(self.dataset.X))
        np.testing.assert_array_almost_equal(self.dataset.y,
                                             np.round(predictions))
