import pytest
import numpy as np
import deepchem as dc
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
from deepchem.data import NumpyDataset
from deepchem.models.torch_models.infograph import Infograph
from deepchem.molnet import load_bace_classification, load_delaney
try:
    import torch
except:
    pass


@pytest.mark.torch
def get_dataset(mode='classification', num_tasks=1):
    data_points = 20
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    
    if mode == 'classification':
        tasks, all_dataset, transformers = load_bace_classification(featurizer)
    else:
        tasks, all_dataset, transformers = load_delaney(featurizer)

    train, valid, test = all_dataset
    w = torch.ones(size=(data_points, num_tasks)).float()

    if mode == 'classification':
        y = torch.randint(0, 2, size=(data_points, num_tasks)).float()
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score,
                                   np.mean,
                                   mode="classification")
    else:
        y = np.random.normal(size=(data_points, num_tasks))
        metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                                   mode="regression")

    ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

    return tasks, ds, transformers, metric


@pytest.mark.torch
def test_infograph_regression():

    tasks, dataset, transformers, metric = get_dataset('regression')
    num_feat = max([dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max([dataset.X[i].num_edge_features for i in range(len(dataset))])
    dim = 32
    
    model = Infograph(num_feat, edge_dim, dim, use_unsup_loss=False, separate_encoder=False, batch_size = 10)
    
    model.fit(dataset, nb_epoch=500)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean_absolute_error'] < 0.1
    
@pytest.mark.torch
def test_infograph_classification():
    
    tasks, dataset, transformers, metric = get_dataset('classification')
    
    num_feat = max([dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max([dataset.X[i].num_edge_features for i in range(len(dataset))])
    dim = 32
    
    model = Infograph(num_feat, edge_dim, dim, use_unsup_loss=False, separate_encoder=False, batch_size = 10)
    
    model.fit(dataset, nb_epoch=500)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9

@pytest.mark.torch
def test_fit_restore():
    # n_samples = 20
    # n_feat = 3
    # n_tasks = 3
    # X = np.random.rand(n_samples, n_feat)
    # # inputs = BatchGraphData(inputs[0])
    # # inputs.edge_features = torch.from_numpy(inputs.edge_features).float().to(self.device)
    # # inputs.edge_index = torch.from_numpy(inputs.edge_index).long().to(self.device)
    # # inputs.node_features = torch.from_numpy(inputs.node_features).float().to(self.device)
    # # inputs.graph_index = torch.from_numpy(inputs.graph_index).long().to(self.device)
    # y = np.zeros((n_samples, n_tasks)).astype(np.float32)
    # dataset = dc.data.NumpyDataset(X, y)
    tasks, dataset, transformers, metric = get_dataset('classification')
    
    num_feat = max([dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max([dataset.X[i].num_edge_features for i in range(len(dataset))])
    dim = 64
    
    model = Infograph(num_feat, edge_dim, dim, use_unsup_loss=False, separate_encoder=False, batch_size = 10)
    
    model.fit(dataset, nb_epoch=1000)
    
    model2 = Infograph(num_feat, edge_dim, dim, use_unsup_loss=False, separate_encoder=False, model_dir=model.model_dir)
    model2.fit(dataset, nb_epoch=1, restore=True)
    prediction = model2.predict_on_batch(dataset.X).reshape(-1, 1)
    assert np.allclose(dataset.y, np.round(prediction))

test_fit_restore()
# featurizer = MolGraphConvFeaturizer(use_edges=True)
# targets, dataset, transforms = dc.molnet.load_zinc15(featurizer=featurizer, splitter='index')
# train_dc, valid_dc, test_dc = dataset

# target = 1
# use_unsup_loss = True
# separate_encoder = True

# # mean = train_dc.y[:, target].mean().item() # just train dc
# # std = train_dc.y[:, target].std().item()

# num_feat_pt = 30 # max([train_dc.X[i].num_node_features for i in range(len(train_dc))])
# edge_dim_pt = 11 # max([train_dc.X[i].num_edge_features for i in range(len(train_dc))])
# dim = 64
# # # num_feat = 20
# # # dim = 30
# weight_decay = 0
# epochs_pt = 10

# batch_size = 200

# x = train_dc.X
# y = train_dc.y[:, target]
# w = train_dc.w[:, target]
# ids = train_dc.ids
# train_zinc = DiskDataset.from_numpy(x, y, w, ids)

# # train_dc_py = train_dc.make_pytorch_dataset(batch_size=batch_size)

# Infograph_model_pt = Infograph(num_feat_pt, edge_dim_pt, dim, use_unsup_loss, separate_encoder, model_dir='infograph_model', tensorboard=True, log_frequency=10, batch_size=batch_size)
# loss_pt = Infograph_model_pt.fit(train_zinc, nb_epoch=epochs_pt)

# ##%
# target = 1
# use_unsup_loss = True
# separate_encoder = True

# featurizer = MolGraphConvFeaturizer(use_edges=True)
# targets, dataset, transforms = dc.molnet.load_bbbp(featurizer=featurizer, splitter='index')
# train_dc, valid_dc, test_dc = dataset
# x = train_dc.X
# y = train_dc.y
# w = train_dc.w
# ids = train_dc.ids
# train_bbbp = DiskDataset.from_numpy(x, y, w, ids)

# # num_feat = 30 
# # edge_dim = 11 
# num_feat = max([train_bbbp.X[i].num_node_features for i in range(len(train_dc))])
# edge_dim = max([train_bbbp.X[i].num_edge_features for i in range(len(train_dc))])
# dim = 64
# epochs_ft = 50
# batch_size = 200

# Infograph_model_ft = Infograph(num_feat, edge_dim, dim, use_unsup_loss, separate_encoder, model_dir='infograph_model_ft2',tensorboard=True, log_frequency=10, batch_size=batch_size)
# # Infograph_model_ft.load_from_modular(model_dir='infograph_model')

# loss_ft = Infograph_model_ft.fit(train_bbbp, nb_epoch=epochs_ft)
