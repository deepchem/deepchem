import torch 
import torch.nn.functional as F
import pytest
import deepchem as dc
import numpy as np
from deepchem.models.torch_models.pretrainer import Pretrainer, PretrainableTorchModel
import torch.nn as nn
from deepchem.models.torch_models.torch_model import TorchModel

class ExampleTorchModel(PretrainableTorchModel):
    """Example TorchModel for testing pretraining."""

    def __init__(self, input_dim, d_hidden, n_layers, d_output, **kwargs):
        self.input_dim = input_dim
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.d_output = d_output
        self.loss = dc.models.losses.SigmoidCrossEntropy()
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
                embedding.append(nn.ReLU())
            else:
                embedding.append(nn.Linear(self.d_hidden, self.d_hidden))
                embedding.append(nn.ReLU())
        return nn.Sequential(*embedding)

    def build_head(self):
        linear = nn.Linear(self.d_hidden, self.d_output)
        af = nn.Sigmoid()
        return nn.Sequential(linear, af)

    def build_model(self, embedding, head):
        return nn.Sequential(embedding, head)
    
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
        af = nn.Sigmoid()
        return nn.Sequential(linear, af)

@pytest.mark.torch
def test_overfit_pretrainer():
    """Test fitting a TorchModel defined by subclassing Module."""
    np.random.seed(123)
    torch.manual_seed(10)
    n_samples = 6
    n_feat = 3
    d_hidden = 6
    n_layers = 8
    n_tasks = 6
    pt_tasks = 3

    X = np.random.rand(n_samples, n_feat)
    y = np.random.randint(2, size=(n_samples, pt_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    
    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    pretrainer = ExamplePretrainer(example_model, pt_tasks)

    pretrainer.fit(dataset, nb_epoch=1000)
    prediction = np.round(np.squeeze(pretrainer.predict_on_batch(X)))
    assert np.array_equal(y, prediction)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    scores = pretrainer.evaluate(dataset, [metric])
    assert scores[metric.name] > 0.9

@pytest.mark.torch
def test_fit_restore():
    np.random.seed(123)
    torch.manual_seed(10)
    n_samples = 6
    n_feat = 3
    d_hidden = 6
    n_layers = 8
    n_tasks = 6
    pt_tasks = 3

    X = np.random.rand(n_samples, n_feat)
    y = np.random.randint(2, size=(n_samples, pt_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    
    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    pretrainer = ExamplePretrainer(example_model, pt_tasks)

    pretrainer.fit(dataset, nb_epoch=1000)
    
    # Create an identical model, do a single step of fitting with restore=True,
    # and make sure it got restored correctly.
  
    example_model2 = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    pretrainer2 = ExamplePretrainer(example_model2, pt_tasks, model_dir=pretrainer.model_dir)
    pretrainer2.fit(dataset, nb_epoch=1, restore=True)
    
    prediction = np.squeeze(pretrainer2.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))
    
# @pytest.mark.torch
# def test_load_from_pretrainer(): 
    
    
# @pytest.mark.torch
# def test_freeze_embedding():
#     pass