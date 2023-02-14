import torch 
import torch.nn as nn
import pytest
import deepchem as dc
import numpy as np
from deepchem.models.torch_models.modular import ModularTorchModel

class ExampleTorchModel(ModularTorchModel): 
    """Example TorchModel for testing pretraining."""

    def __init__(self, input_dim, d_hidden, n_layers, d_output, **kwargs):
        self.input_dim = input_dim
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.d_output = d_output
        self.components = self.build_components()
        self.custom_loss = self.loss_func
        self.model = self.build_model()
        super().__init__(self.model, self.components, self.custom_loss, **kwargs)

    def build_components(self):
        return {'encoder': self.encoder(), 'FF1': self.FF1(), 'FF2': self.FF2()}

    def loss_func(self, data, components:dict):
        inputs1 = torch.from_numpy(data.X).float()
        preds1 = components['FF1'](components['encoder'](inputs1))
        labels1 = torch.tensor(data.y)
        loss1 = torch.nn.functional.mse_loss(preds1, labels1)
        
        inputs2 = torch.from_numpy(data.X).float()
        preds2 = components['FF1'](inputs2)
        labels2 = torch.tensor(data.y)
        loss2 = torch.nn.functional.smooth_l1_loss(preds2, labels2)
        
        return loss1 + loss2

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
        encoder = self.encoder()
        FF2 = self.FF2()
        return nn.Sequential(encoder, FF2)
    
class ExamplePretrainer(ModularTorchModel): 
    def __init__(self, model, pt_tasks, **kwargs):
        self.source_model = model # the pretrainer takes the original model as input in order to modify it
        self.pt_tasks = pt_tasks
        self.components = self.build_components()
        self.model = self.build_model()
        super().__init__(self.model, self.components, **kwargs)
    def FF_pt(self):
        linear = nn.Linear(self.source_model.d_hidden, self.pt_tasks)
        af = nn.ReLU()
        return nn.Sequential(linear, af).cuda()
    def loss_func(self, inputs, labels, weights):
        inputs = inputs[0]
        labels = labels[0]
        weights = weights[0]
        # inputs = torch.from_numpy(inputs).float().cuda()
        preds = self.components['FF_pt'](self.components['encoder'](inputs))
        # labels = torch.tensor(labels).cuda()
        loss = torch.nn.functional.mse_loss(preds, labels)
        loss = loss * weights
        loss = loss.mean()
        return loss
    def build_components(self):
        pt_components = self.source_model.build_components()
        pt_components.update({'FF_pt': self.FF_pt()})
        return pt_components
    def build_model(self):
        return nn.Sequential(self.components['encoder'], self.components['FF_pt']).cuda()

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
    # pt_tasks = 3

    X = np.random.rand(n_samples, n_feat)
    y = np.zeros((n_samples, n_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    
    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)

    example_model.fit(dataset)
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
    pt_tasks = 3

    X = np.random.rand(n_samples, n_feat)
    y = np.zeros((n_samples, pt_tasks)).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    
    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    pretrainer = ExampleTorchModel(example_model, pt_tasks)

    pretrainer.fit(dataset, nb_epoch=1000)
    
    # Create an identical model, do a single step of fitting with restore=True and make sure it got restored correctly.
  
    example_model2 = ExampleTorchModel(n_feat, d_hidden, n_layers, n_tasks)
    pretrainer2 = ExamplePretrainer(example_model2, pt_tasks, model_dir=pretrainer.model_dir)
    pretrainer2.fit(dataset, nb_epoch=1, restore=True)
    
    prediction = np.squeeze(pretrainer2.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))
        
def test_overfit_modular():
    np.random.seed(123)
    torch.manual_seed(10)
    n_samples = 6
    n_feat = 3
    d_hidden = 3
    n_layers = 2
    ft_tasks = 6
    pt_tasks = 3

    # X = np.random.rand(n_samples, n_feat)
    # y_ft = np.zeros((n_samples, ft_tasks)).astype(np.float32)
    # # w_ft = np.ones((n_samples, ft_tasks)).astype(np.float32)
    # dataset_ft = dc.data.NumpyDataset(X, y_ft)
    
    X = np.random.rand(n_samples, n_feat)
    y_pt = np.zeros((n_samples, pt_tasks)).astype(np.float32)
    w_pt = np.ones((n_samples, pt_tasks)).astype(np.float32)
    dataset_pt = dc.data.NumpyDataset(X, y_pt,w_pt)
    
    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers,  ft_tasks, model_dir='./example_model') 
    example_pretrainer = ExamplePretrainer(example_model, pt_tasks, model_dir='./example_pretrainer')

    example_pretrainer.fit(dataset_pt, nb_epoch=1000)

    prediction = np.round(np.squeeze(example_pretrainer.predict_on_batch(X)))
    assert np.array_equal(y_pt, prediction)
    
def test_load_freeze():
    np.random.seed(123)
    torch.manual_seed(10)
    n_samples = 6
    n_feat = 3
    d_hidden = 3
    n_layers = 2
    ft_tasks = 6
    pt_tasks = 3
    
    X_ft = np.random.rand(n_samples, n_feat)
    y_ft = np.zeros((n_samples, ft_tasks)).astype(np.float32)
    # w_ft = np.ones((n_samples, ft_tasks)).astype(np.float32)
    dataset_ft = dc.data.NumpyDataset(X_ft, y_ft)
    
    X_pt = np.random.rand(n_samples, n_feat)
    y_pt = np.zeros((n_samples, pt_tasks)).astype(np.float32)
    # w_pt = np.ones((n_samples, pt_tasks)).astype(np.float32)
    dataset_pt = dc.data.NumpyDataset(X_pt, y_pt)
    
    example_model = ExampleTorchModel(n_feat, d_hidden, n_layers,  ft_tasks, model_dir='./example_model') 
    example_pretrainer = ExamplePretrainer(example_model, pt_tasks, model_dir='./example_pretrainer')

    example_pretrainer.fit(dataset_pt, nb_epoch=1000)
    
    example_model.load_from_pretrained(source_model = example_pretrainer, components=['encoder'])
    
    for param in example_model.components['encoder'].parameters():
        param.requires_grad = False
    example_model.model = example_model.build_model()
    
    example_model.fit(dataset_ft, nb_epoch=1)
    
    # check that the first layer is still the same between the two models
    assert np.array_equal(example_pretrainer.components['encoder'][0].weight.data.cpu().numpy(),example_model.components['encoder'][0].weight.data.cpu().numpy())

    # check that the predictions are different becuase of the fine tuning
    assert not np.array_equal(np.round(np.squeeze(example_pretrainer.predict_on_batch(X_ft))), np.round(np.squeeze(example_model.predict_on_batch(X_ft))))

test_fit_restore()
test_load_freeze()
test_overfit_modular()