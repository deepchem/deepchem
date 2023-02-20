from deepchem.models.torch_models.modular import ModularTorchModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from deepchem.feat.graph_data import BatchGraphData
import math
from torch_geometric.nn import NNConv
from torch_geometric.nn.aggr import Set2Set


class Encoder(torch.nn.Module):
    def __init__(self, num_features, edge_features, dim):
        super(Encoder, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(edge_features, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)

    def forward(self, data):
        out = F.relu(self.lin0(data.node_features))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_features))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.graph_index)
        return out
    
    
class FF(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class Infograph(ModularTorchModel):
    def __init__(self, num_features, edge_features, dim, use_unsup_loss=False, separate_encoder=False, **kwargs):
        self.embedding_dim = dim
        self.edge_features = edge_features
        self.separate_encoder = separate_encoder
        self.local = True
        self.num_features = num_features
        self.use_unsup_loss = use_unsup_loss
        
        self.components = self.build_components()
        self.model = self.build_model()
        super().__init__(self.model, self.components, **kwargs)
        
                     
    def build_components(self):
        return {'encoder': Encoder(self.num_features, self.edge_features, self.embedding_dim),
                'unsup_encoder': Encoder(self.num_features, self.edge_features, self.embedding_dim),
                'ff1': FF(2*self.embedding_dim, self.embedding_dim),
                'ff2': FF(2*self.embedding_dim, self.embedding_dim),
                'fc1': torch.nn.Linear(2 * self.embedding_dim, self.embedding_dim),
                'fc2': torch.nn.Linear(self.embedding_dim, 1),
                'local_d': FF(self.embedding_dim, self.embedding_dim),
                'global_d': FF(2*self.embedding_dim, self.embedding_dim)
                }
    
    def build_model(self):
        return InfoGraph_module(**self.components)
            
    def loss_func(self, inputs, labels, weights):
        if self.use_unsup_loss:
            sup_loss = F.mse_loss(self.model(inputs), labels)
            unsup_loss = self.unsup_loss(inputs)
            if self.separate_encoder:
                unsup_sup_loss = self.unsup_sup_loss(inputs, labels, weights)
                loss = sup_loss + unsup_loss + unsup_sup_loss * self.learning_rate
            else:
                loss = sup_loss + unsup_loss * self.learning_rate
            return (loss * weights).mean()
        else:
            sup_loss = F.mse_loss(self.model(inputs), labels)
            return (sup_loss * weights).mean()

    def unsup_loss(self, inputs):
        if self.separate_encoder:
            y, M = self.components['unsup_encoder'](inputs)
        else:
            y, M = self.components['encoder'](inputs)
        g_enc = self.components['global_d'](y)
        l_enc = self.components['local_d'](M)
    
        measure = 'JSD'
        if self.local:
            loss = self.local_global_loss_(l_enc, g_enc, inputs.edge_index,
                                      inputs.graph_index, measure)
        return loss
    
    def unsup_sup_loss(self, inputs, labels, weights):
        y, M = self.components['encoder'](inputs)
        y_, M_ = self.components['unsup_encoder'](inputs)
    
        g_enc = self.components['ff1'](y)
        g_enc1 = self.components['ff2'](y_)
    
        measure = 'JSD'
        loss = self.global_global_loss_(g_enc, g_enc1, inputs.edge_index,
                                   inputs.graph_index, measure)
        # loss = (loss * weights).mean()
        return loss
    
    def _prepare_batch(self, batch):
        inputs, labels, weights = batch
        inputs = BatchGraphData(inputs[0])
        inputs.edge_features = torch.from_numpy(inputs.edge_features).float().to(self.device)
        inputs.edge_index = torch.from_numpy(inputs.edge_index).long().to(self.device)
        inputs.node_features = torch.from_numpy(inputs.node_features).float().to(self.device)
        inputs.graph_index = torch.from_numpy(inputs.graph_index).long().to(self.device)
    
        _, labels, weights = super(Infograph, self)._prepare_batch(
            ([], labels, weights))
        
        if (len(labels) != 0) and (len(weights) != 0):
            labels = labels[0]
            weights = weights[0]
        
        return inputs, labels, weights
    
    def local_global_loss_(self,l_enc, g_enc, edge_index, batch, measure):
        '''
        Args:
            l: Local feature map.
            g: Global features.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
        Returns:
            torch.Tensor: Loss.
        '''
        num_graphs = g_enc.shape[0]
        num_nodes = l_enc.shape[0]
    
        pos_mask = torch.zeros((num_nodes, num_graphs)).to(self.device)  
        neg_mask = torch.ones((num_nodes, num_graphs)).to(self.device)
        for nodeidx, graphidx in enumerate(batch):
            pos_mask[nodeidx][graphidx] = 1.
            neg_mask[nodeidx][graphidx] = 0.
    
        res = torch.mm(l_enc, g_enc.t())
    
        E_pos = get_positive_expectation(res * pos_mask, measure, average=False)
        E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
        E_neg = get_negative_expectation(res * neg_mask, measure, average=False)
        E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()
    
        return E_neg - E_pos
    
    def global_global_loss_(self, g_enc, g_enc1, edge_index, batch, measure):
        '''
        Args:
            g: Global features
            g1: Global features.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
        Returns:
            torch.Tensor: Loss.
        '''
        num_graphs = g_enc.shape[0]
    
        pos_mask = torch.eye(num_graphs).to(self.device)
        neg_mask = 1 - pos_mask
    
        res = torch.mm(g_enc, g_enc1.t())
    
        E_pos = get_positive_expectation(res * pos_mask, measure, average=False)
        E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
        E_neg = get_negative_expectation(res * neg_mask, measure, average=False)
        E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()
    
        return E_neg - E_pos
    
    
class InfoGraph_module(torch.nn.Module):
        
    def __init__(self, encoder, unsup_encoder, ff1, ff2, fc1, fc2, local_d, global_d): 
        super(InfoGraph_module, self).__init__()
        self.encoder = encoder
        self.unsup_encoder = unsup_encoder
        self.ff1 = ff1
        self.ff2 = ff2
        self.fc1 = fc1
        self.fc2 = fc2
        self.local_d = local_d
        self.global_d = global_d
        self.init_emb()

    def init_emb(self):
        # initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        out = self.encoder(data)
        out = F.relu(self.fc1(out))
        pred = self.fc2(out)
        return pred


def log_sum_exp(x, axis=None):
    """Log sum exp function                 

    Args:
        x: Input.
        axis: Axis over which to perform sum.

    Returns:
        torch.Tensor: log sum exp

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def random_permute(X):
    """Randomly permutes a tensor.

    Args:
        X: Input tensor.

    Returns:
        torch.Tensor

    """
    X = X.transpose(1, 2)
    b = torch.rand((X.size(0), X.size(1))).cuda()
    idx = b.sort(0)[1]
    adx = torch.range(0, X.size(1) - 1).long()
    X = X[idx, adx[None, :]].transpose(1, 2)
    return X


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = -F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples**2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples**2) + 1.)**2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq







