import deepchem as dc
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
from deepchem.data import DiskDataset
# import torch
# from deepchem.feat.graph_data import BatchGraphData
from deepchem.models.torch_models.infograph import Infograph
# import torch.nn.functional as F
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

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

# plt.plot(loss_ft)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()






# if use_unsup_loss:
#     unsup_train_dataset = train_dc_py
    

    
# def test(loader):
#     model.eval()
#     error = 0

#     for batch in loader:
#         X, y = dc_to_pyg(batch)
        
#         error += (model(X) * std - y * std).abs().sum().item()  # MAE
#     return error / len(loader.disk_dataset)

# def train(epoch, use_unsup_loss):
#     model.train()
#     loss_all = 0
#     sup_loss_all = 0
#     unsup_loss_all = 0
#     unsup_sup_loss_all = 0

#     if use_unsup_loss:
#         for data, data2 in zip(train_dc_py, unsup_train_dataset):
#             data, y = dc_to_pyg(data)
#             data2, y2 = dc_to_pyg(data2)
#             # data = data.to(device)
#             # data2 = data2.to(device)
#             optimizer.zero_grad()

#             sup_loss = F.mse_loss(model(data), y)
#             unsup_loss = model.unsup_loss(data2)
#             if separate_encoder:
#                 unsup_sup_loss = model.unsup_sup_loss(data2)
#                 loss = sup_loss + unsup_loss + unsup_sup_loss * llama
#             else:
#                 loss = sup_loss + unsup_loss * llama

#             loss.backward()

#             sup_loss_all += sup_loss.item()
#             unsup_loss_all += unsup_loss.item()
#             if separate_encoder:
#                 unsup_sup_loss_all += unsup_sup_loss.item()
#             loss_all += loss.item() * batch_size #data.num_graphs

#             optimizer.step()

#         if separate_encoder:
#             print(sup_loss_all, unsup_loss_all, unsup_sup_loss_all)
#         else:
#             print(sup_loss_all, unsup_loss_all)
#         return loss_all / len(train_dc_py.disk_dataset)
#     else:
#         for data in train_dc_py:
#             # data = data.to(device)
#             optimizer.zero_grad()

#             sup_loss = F.mse_loss(model(data), y)
#             loss = sup_loss

#             loss.backward()
#             loss_all += loss.item() * batch_size # data.num_graphs
#             optimizer.step()

#         return loss_all / len(train_dc_py.disk_dataset)

# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# model = InfoGraph(num_feat, dim, use_unsup_loss, separate_encoder).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

# val_error = test(valid_dc_py)
# test_error = test(test_dc_py)
# print('Epoch: {:03d}, Validation MAE: {:.7f}, Test MAE: {:.7f},'.format(0, val_error, test_error))

# best_val_error = None
# for epoch in range(1, epochs):
#     lr = scheduler.optimizer.param_groups[0]['lr']
#     loss = train(epoch, use_unsup_loss)
#     val_error = test(valid_dc_py)
#     scheduler.step(val_error)

#     if best_val_error is None or val_error <= best_val_error:
#         test_error = test(test_dc_py)
#         best_val_error = val_error