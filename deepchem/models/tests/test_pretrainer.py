import torch 
import torch.nn as nn
import pytest
import deepchem as dc
import numpy as np
from deepchem.models.torch_models.pretrainer import Pretrainer, PretrainableTorchModel
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
    """Overfit test the pretrainer to ensure it can learn a simple task."""
    np.random.seed(123)
    torch.manual_seed(10)
    n_samples = 6
    n_feat = 3
    d_hidden = 3
    n_layers = 1
    n_tasks = 6
    pt_tasks = 3

    X = np.random.rand(n_samples, n_feat)
    y = np.zeros((n_samples, pt_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    
    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    pretrainer = ExamplePretrainer(example_model, pt_tasks)

    pretrainer.fit(dataset, nb_epoch=1000)
    prediction = np.round(np.squeeze(pretrainer.predict_on_batch(X)))
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
    pt_tasks = 3

    X = np.random.rand(n_samples, n_feat)
    y = np.zeros((n_samples, pt_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    
    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    pretrainer = ExamplePretrainer(example_model, pt_tasks)

    pretrainer.fit(dataset, nb_epoch=1000)
    
    # Create an identical model, do a single step of fitting with restore=True and make sure it got restored correctly.
  
    example_model2 = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    pretrainer2 = ExamplePretrainer(example_model2, pt_tasks, model_dir=pretrainer.model_dir)
    pretrainer2.fit(dataset, nb_epoch=1, restore=True)
    
    prediction = np.squeeze(pretrainer2.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))
    

    
@pytest.mark.torch
def test_load_freeze_embedding():
    """Test that the pretrainer can be used to load into a PretrainableTorchModel, freeze the TorchModel embedding, and train the head."""
    
    np.random.seed(123)
    torch.manual_seed(10)
    n_samples = 6
    n_feat = 3
    d_hidden = 3
    n_layers = 1
    n_tasks = 6
    pt_tasks = 3

    X_pt = np.random.rand(n_samples, n_feat)
    y = np.zeros((n_samples, pt_tasks)).astype(np.float32)
    pt_dataset = dc.data.NumpyDataset(X_pt, y)
    
    X_ft = np.random.rand(n_samples, n_feat)
    y = np.zeros((n_samples, n_tasks)).astype(np.float32)
    ft_dataset = dc.data.NumpyDataset(X_ft, y)
    
    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    pretrainer = ExamplePretrainer(example_model, pt_tasks)

    # train pretrainer
    pretrainer.fit(pt_dataset, nb_epoch=1000)
    
    # load embedding from pretrainer
    example_model2 = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    example_model2.load_from_pretrained(pretrainer,include_top=False,model_dir=pretrainer.model_dir)
    
    # freeze embedding layers
    for param in example_model2.embedding.parameters():
        param.requires_grad = False
        
    # fine tune the second model
    example_model2.fit(ft_dataset, nb_epoch=1)
    
    # check that the first layer is still the same between the two models
    assert np.array_equal(pretrainer.embedding[0].weight.data.cpu().numpy(),example_model2.embedding[0].weight.data.cpu().numpy())
    
    # check that the predictions are different becuase of the fine tuning
    assert not np.array_equal(np.round(np.squeeze(pretrainer.predict_on_batch(X_ft))), np.round(np.squeeze(example_model2.predict_on_batch(X_ft))))

