import torch 
import torch.nn.functional as F
import pytest
import deepchem as dc
import numpy as np
from deepchem.models.torch_models.pretrainer import Pretrainer, PretrainableTorchModel
import torch.nn as nn
from deepchem.models.torch_models.torch_model import TorchModel


@pytest.mark.torch
def test_overfit_pretrainer():
    """Test fitting a TorchModel defined by subclassing Module."""
    np.random.seed(123)
    n_samples = 6
    n_feat = 8
    d_hidden = 6
    n_layers = 10
    n_tasks = 6
    pt_tasks = 3

    X = np.random.rand(n_samples, n_feat)
    y = np.random.randint(2, size=(n_samples, pt_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    
    class ExampleTorchModel(PretrainableTorchModel):
        """Example TorchModel for testing pretraining."""

        def __init__(self, input_dim, d_hidden, n_layers, d_output, **kwargs):
            self.input_dim = input_dim
            self.d_hidden = d_hidden
            self.n_layers = n_layers
            self.d_output = d_output
            self.loss = dc.models.losses.L2Loss()
            self._head = self.build_head()
            self._embedding = self.build_embedding()
            self._model = self.build_model(self._embedding, self._head)
            super().__init__(self._model, self.loss, **kwargs)

        @property
        def embedding(self):
            return self._embedding

        def build_embedding(self):
            embedding = []
            for i in range(self.n_layers):
                if i == 0:
                    embedding.append(nn.Linear(self.input_dim, self.d_hidden))
                    # embedding.append(nn.ReLU())
                else:
                    embedding.append(nn.Linear(self.d_hidden, self.d_hidden))
                    # embedding.append(nn.ReLU())
            return nn.Sequential(*embedding)

        def build_head(self):
            return nn.Linear(self.d_hidden, self.d_output)

        def build_model(self, embedding, head):
            
            class ExampleModel(nn.Module):

                def __init__(self, embedding, head):
                    super(ExampleModel, self).__init__()
                    self.embedding = embedding
                    self.head = head

                def forward(self, x):
                    for i, layer in enumerate(self.embedding):
                        x = layer(x)
                        if i < len(self.embedding) - 1:
                            x = F.relu(x)
                    x = self.head(x)
                    x = torch.sigmoid(x)
                    return x
            
            return ExampleModel(embedding, head)
        
    class ExamplePretrainer(Pretrainer):
        """Example Pretrainer for testing."""

        def __init__(self, model: ExampleTorchModel, pt_tasks: int, **kwargs):

            self._embedding = model.build_embedding()
            self._head = self.build_head(model.d_hidden, pt_tasks)
            self._model = model.build_model(self._embedding, self._head)
            self.loss = self.build_pretrain_loss()
            torchmodel = TorchModel(self._model, self.loss, **kwargs)
            super().__init__(torchmodel, **kwargs)

        @property
        def embedding(self): 
            return self._embedding
        
        def build_pretrain_loss(self):
            return dc.models.losses.L2Loss()

        def build_head(self, d_hidden, pt_tasks):
            linear = nn.Linear(d_hidden, pt_tasks)
            # af = nn.Sigmoid()
            # return nn.Sequential(linear, af)
            return linear

    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    pretrainer = ExamplePretrainer(example_model, pt_tasks, learning_rate=0.001)

    pretrainer.fit(dataset, nb_epoch=1000)
    prediction = np.round(np.squeeze(pretrainer.predict_on_batch(X)))
    assert np.array_equal(y, prediction)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    scores = pretrainer.evaluate(dataset, [metric])
    assert scores[metric.name] > 0.9

test_overfit_pretrainer()
# @pytest.mark.torch
# def test_freeze_embedding():
#     pass

# @pytest.mark.torch
# def test_load_from_pretrainer():
#     pass