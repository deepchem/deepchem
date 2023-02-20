import deepchem as dc
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
from deepchem.data import DiskDataset
from deepchem.models.torch_models.infograph import Infograph
import pytest


@pytest.mark.torch
def get_dataset(mode='classification', featurizer='GraphConv', num_tasks=2):
    data_points = 20
    if mode == 'classification':
        tasks, all_dataset, transformers = load_bace_classification(featurizer)
    else:
        tasks, all_dataset, transformers = load_delaney(featurizer)

    train, valid, test = all_dataset
    for i in range(1, num_tasks):
        tasks.append("random_task")
    w = np.ones(shape=(data_points, len(tasks)))

    if mode == 'classification':
        y = np.random.randint(0, 2, size=(data_points, len(tasks)))
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score,
                                   np.mean,
                                   mode="classification")
    else:
        y = np.random.normal(size=(data_points, len(tasks)))
        metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                                   mode="regression")

    ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

    return tasks, ds, transformers, metric


@pytest.mark.torch
def test_infograph_regression():
    pass

def test_infograph_classification():
    pass


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

##%
target = 1
use_unsup_loss = True
separate_encoder = True

featurizer = MolGraphConvFeaturizer(use_edges=True)
targets, dataset, transforms = dc.molnet.load_bbbp(featurizer=featurizer, splitter='index')
train_dc, valid_dc, test_dc = dataset
x = train_dc.X
y = train_dc.y
w = train_dc.w
ids = train_dc.ids
train_bbbp = DiskDataset.from_numpy(x, y, w, ids)

# num_feat = 30 
# edge_dim = 11 
num_feat = max([train_bbbp.X[i].num_node_features for i in range(len(train_dc))])
edge_dim = max([train_bbbp.X[i].num_edge_features for i in range(len(train_dc))])
dim = 64
epochs_ft = 50
batch_size = 200

Infograph_model_ft = Infograph(num_feat, edge_dim, dim, use_unsup_loss, separate_encoder, model_dir='infograph_model_ft2',tensorboard=True, log_frequency=10, batch_size=batch_size)
# Infograph_model_ft.load_from_modular(model_dir='infograph_model')

loss_ft = Infograph_model_ft.fit(train_bbbp, nb_epoch=epochs_ft)
