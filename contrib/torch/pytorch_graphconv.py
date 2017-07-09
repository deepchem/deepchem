import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import roc_auc_score
import scipy


def symmetric_normalize_adj(adj):
    """
    Implements symmetric normalization of graphs
    trick described here:
    https://tkipf.github.io/graph-convolutional-networks/]

    adj: NxN graph adjacency matrix (2d square numpy array)
    """
    n_atoms = np.where(np.max(adj, axis=1)==0)[0][0]
    if n_atoms == 0:
        return(adj)
    orig_shape = adj.shape
    adj = adj[:n_atoms, :n_atoms]
    degree = np.sum(adj, axis=1)
    D = np.diag(degree)
    D_sqrt = scipy.linalg.sqrtm(D)
    D_sqrt_inv = scipy.linalg.inv(D_sqrt)
    sym_norm = D_sqrt_inv.dot(adj)
    sym_norm = sym_norm.dot(D_sqrt_inv)
    new_adj = np.zeros(orig_shape)
    new_adj[:n_atoms, :n_atoms] = sym_norm
    return(new_adj)


class GraphConvolution(nn.Module):
    """
    Differentiable function that performs a graph convolution 
    given adjacency matrix G and feature matrix X
    """
    def __init__(self, n_conv_layers=1,
                 max_n_atoms=200,
                 n_atom_types=75,
                 conv_layer_dims=[64,128,256],
                 n_fc_layers=2,
                 fc_layer_dims=[64, 10],
                 dropout=0.,
                 return_sigmoid=True):

        """
        Defines the operations available in this module.

        n_conv_layers: int, number of graph convolution layers 
        max_n_atoms: int, N, n_rows (n_cols) of adjacency matrix 
        n_atom_types: int, number of features describing each atom in 
            input 
        conv_layer_dims: list of ints, output n_features for each 
            graph conv layer 
        n_fc_layers: int, number of fully connected layers
        fc_layer_dims: list of ints, output n_features for each 
            fully connected layer 
        dropout: float, probability of zeroing out a given output neuron 
        return_sigmoid: boolean, determines if forward pass 
            returns sigmoid activation on the final layer
        """

        super(GraphConvolution, self).__init__()
        
        self.n_conv_layers = n_conv_layers
        self.max_n_atoms = max_n_atoms
        self.n_atom_types = n_atom_types
        self.fc_layer_dims = fc_layer_dims
        self.n_fc_layers = n_fc_layers
        self.return_sigmoid = return_sigmoid

        self.conv_layer_dims = [n_atom_types] + conv_layer_dims
        self.dropout = dropout

        self.conv_ops = nn.ModuleList()
        for layer_idx in range(self.n_conv_layers):
            p_in = self.conv_layer_dims[layer_idx]
            p_out = self.conv_layer_dims[layer_idx+1]
            op = nn.Sequential(
                    nn.Linear(p_in, p_out),
                    nn.Dropout(p=self.dropout),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(p_out))
            self.conv_ops.append(op)

        self.fc_ops = nn.ModuleList()

        self.fc_layer_dims = [self.conv_layer_dims[self.n_conv_layers]] + self.fc_layer_dims


        for layer_idx in range(self.n_fc_layers):
            p_in = self.fc_layer_dims[layer_idx]
            p_out = self.fc_layer_dims[layer_idx+1]
            op = nn.Sequential(
                    nn.Linear(p_in, p_out),
                    nn.Dropout(p=self.dropout),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(p_out))
            self.fc_ops.append(op)


        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif m.__class__.__name__.find("BatchNorm") != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)



    def forward(self, G, x):
        """
        Performs a series of graph convolutions 
        followed by a summation and 
        fully connected layers.

        G: (batch_size, max_n_atoms, max_n_atoms) batch of adjacency matrices 
        x: (batch_size, max_n_atoms, p) batch of feature matrices for each 
            molecule
        """
        h = x
        for layer_idx in range(self.n_conv_layers):
            h = torch.bmm(G, h)
            h = h.view(-1, h.size()[-1])

            op = self.conv_ops[layer_idx]
            h = op(h)
            h = h.view(-1, self.max_n_atoms, self.conv_layer_dims[layer_idx+1])

        h = torch.squeeze(torch.sum(h, dim=1), dim=1)

        for layer_idx in range(self.n_fc_layers):
            op = self.fc_ops[layer_idx]
            h = op(h)

        if self.return_sigmoid:
            h = nn.Sigmoid()(h)

        return(h)

class SingleTaskGraphConvolution(object):
    """
    Convenience class for training a single task graph convolutional model. 
    """
    def __init__(self, net, lr, weight_decay):
        """
        net: an instance of class GraphConvolution
        lr: float, learning rate 
        weight_decay: float
        """
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.input_x = torch.FloatTensor(-1, self.net.max_n_atoms, self.net.n_atom_types)
        self.input_g = torch.FloatTensor(-1, self.net.max_n_atoms, self.net.max_n_atoms)
        self.label = torch.FloatTensor(-1)
        
        self.net.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.cuda()
        
        self.input_x, self.input_g, self.label = self.input_x.cuda(), self.input_g.cuda(), self.label.cuda()

        self.lr = lr
        self.weight_decay = weight_decay
        # setup optimizer
        self.optimizer = optim.Adam(self.net.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)

    def train_epoch(self, train_features, y_train, batch_size=32,
                    shuffle_train_inds=True):
        """
        train_features: list of dictionaries. each dictionary represents one sample feature. 
            key "x" maps to max_n_atoms x p feature matrix. key "g" maps to square adjacency matrix 
        y_train: numpy array of labels 
        """

        train_inds = range(0, len(train_features))
        if shuffle_train_inds:
            random.shuffle(train_inds)

        for b in range(0, len(train_inds)/batch_size):
            batch_inds = [train_inds[idx] for idx in range(b*batch_size, (b+1)*batch_size)]
            
            train_x_batch = np.concatenate([np.expand_dims(train_features[idx]["x"], 0) for idx in batch_inds], axis=0)
            train_g_batch = np.concatenate([np.expand_dims(train_features[idx]["g"], 0) for idx in batch_inds], axis=0)

            xb = torch.from_numpy(train_x_batch.astype(np.float32)).cuda()
            gb = torch.from_numpy(train_g_batch.astype(np.float32)).cuda()
            yb = torch.from_numpy(y_train[batch_inds].astype(np.float32)).cuda()

            self.net.train()
            self.net.zero_grad()
            
            self.input_x.resize_as_(xb).copy_(xb)
            self.input_g.resize_as_(gb).copy_(gb)
            self.label.resize_as_(yb).copy_(yb)
            
            input_xv = Variable(self.input_x)
            input_gv = Variable(self.input_g)
            label_v = Variable(self.label)

            output = self.net(input_gv, input_xv)
            
            err = self.criterion(output, label_v)
            err.backward()
            
            self.optimizer.step()

    def evaluate(self, train_features,
                       test_features,
                       y_train,
                       y_test, 
                       transformer,
                       batch_size=32):
        
        self.net.eval()
        print("TRAIN:")
        
        o = []
        l = []

        train_inds = range(0, len(train_features))

        for b in range(0, len(train_features)/batch_size):
            batch_inds = [train_inds[idx] for idx in range(b*batch_size, (b+1)*batch_size)]
            
            train_x_batch = np.concatenate([np.expand_dims(train_features[idx]["x"], 0) for idx in batch_inds], axis=0)
            train_g_batch = np.concatenate([np.expand_dims(train_features[idx]["g"], 0) for idx in batch_inds], axis=0)

            xb = torch.from_numpy(train_x_batch.astype(np.float32)).cuda()
            gb = torch.from_numpy(train_g_batch.astype(np.float32)).cuda()
            
            self.input_x.resize_as_(xb).copy_(xb)
            self.input_g.resize_as_(gb).copy_(gb)
            
            input_xv = Variable(self.input_x)
            input_gv = Variable(self.input_g)

            output = self.net(input_gv, input_xv)
            
            if transformer is not None:
                o.append(transformer.inverse_transform(output.data.cpu().numpy().reshape((-1,1))).flatten())
                l.append(transformer.inverse_transform(y_train[batch_inds].reshape((-1,1))).flatten())
            else:
                o.append(output.data.cpu().numpy().reshape((-1,1)).flatten())
                l.append(y_train[batch_inds].reshape((-1,1)).flatten())

        o = np.concatenate(o)
        l = np.concatenate(l)
        print("RMSE:")
        print(np.sqrt(np.mean(np.square(l-o))))
        print("ROC AUC:")
        print(roc_auc_score(l, o))
        
        o = []
        l = []

        print("TEST:")
        test_inds = range(0, len(test_features))

        for b in range(0, len(test_features)/batch_size):
            batch_inds = [test_inds[idx] for idx in range(b*batch_size, (b+1)*batch_size)]
            
            test_x_batch = np.concatenate([np.expand_dims(test_features[idx]["x"], 0) for idx in batch_inds], axis=0)
            test_g_batch = np.concatenate([np.expand_dims(test_features[idx]["g"], 0) for idx in batch_inds], axis=0)

            xb = torch.from_numpy(test_x_batch.astype(np.float32)).cuda()
            gb = torch.from_numpy(test_g_batch.astype(np.float32)).cuda()
            
            self.input_x.resize_as_(xb).copy_(xb)
            self.input_g.resize_as_(gb).copy_(gb)
            
            input_xv = Variable(self.input_x)
            input_gv = Variable(self.input_g)

            output = self.net(input_gv, input_xv)
            
            if transformer is not None:
                o.append(transformer.inverse_transform(output.data.cpu().numpy().reshape((-1,1))).flatten())
                l.append(transformer.inverse_transform(y_test[batch_inds].reshape((-1,1))).flatten())
            else:
                o.append(output.data.cpu().numpy().reshape((-1,1)).flatten())
                l.append(y_test[batch_inds].reshape((-1,1)).flatten())

        o = np.concatenate(o)
        l = np.concatenate(l)
        print("RMSE:")
        print(np.sqrt(np.mean(np.square(l-o))))
        print("ROC AUC:")
        print(roc_auc_score(l, o))



class MultiTaskGraphConvolution(object):
    """
    Convenience Class for training and evaluating multitask graph convolutional network 
    """

    def __init__(self, net, lr, weight_decay, n_tasks):
        """
        net: an instance of class GraphConvolution
        lr: float, learning rate 
        weight_decay: float
        n_tasks: int, number of tasks
        """
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.input_x = torch.FloatTensor(-1, self.net.max_n_atoms, self.net.n_atom_types)
        self.input_g = torch.FloatTensor(-1, self.net.max_n_atoms, self.net.max_n_atoms)
        self.label = torch.FloatTensor(-1)
        
        self.net.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.criterion.cuda()
        
        self.input_x, self.input_g, self.label = self.input_x.cuda(), self.input_g.cuda(), self.label.cuda()

        self.lr = lr
        self.weight_decay = weight_decay
        # setup optimizer
        self.optimizer = optim.Adam(self.net.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)

        self.n_tasks = n_tasks

    def multitask_loss(self, output, label_v):
        losses = []
        
        for task in range(self.n_tasks):
            #print("tasK: %d" %task)
            scores = output[:,task].contiguous().view((-1,1))
            #cores = torch.cat([scores, 1.-scores], dim=1)
            #print("scores")
            #print(scores.size())
            task_label = label_v[:,task]#.long()

            #print("task_label")
            #print(task_label.size())
            #task_loss =  self.criterion(scores, task_label)
            task_loss = -(task_label * torch.log(scores)) + (1. - task_label) * torch.log(1. - task_label)
            task_loss = task_loss.mean()
            losses.append(task_loss)
            #print("task_loss")
            #print(task_loss.size())
        loss = torch.cat(losses).mean()
        return(loss)


    def train_epoch(self, train_features, y_train, batch_size=32,
                    shuffle_train_inds=True):
        train_inds = range(0, len(train_features))
        if shuffle_train_inds:
            random.shuffle(train_inds)

        for b in range(0, len(train_inds)/batch_size):
            batch_inds = [train_inds[idx] for idx in range(b*batch_size, (b+1)*batch_size)]
            
            train_x_batch = np.concatenate([np.expand_dims(train_features[idx]["x"], 0) for idx in batch_inds], axis=0)
            train_g_batch = np.concatenate([np.expand_dims(train_features[idx]["g"], 0) for idx in batch_inds], axis=0)

            xb = torch.from_numpy(train_x_batch.astype(np.float32)).cuda()
            gb = torch.from_numpy(train_g_batch.astype(np.float32)).cuda()
            yb = torch.from_numpy(y_train[batch_inds].astype(np.float32)).cuda()

            self.net.train()
            self.net.zero_grad()
            
            self.input_x.resize_as_(xb).copy_(xb)
            self.input_g.resize_as_(gb).copy_(gb)
            self.label.resize_as_(yb).copy_(yb)
            
            input_xv = Variable(self.input_x)
            input_gv = Variable(self.input_g)
            label_v = Variable(self.label)

            output = self.net(input_gv, input_xv)
            
            err = self.multitask_loss(output, label_v)
            err.backward()
            
            self.optimizer.step()

    def evaluate(self, train_features,
                       test_features,
                       y_train,
                       y_test, 
                       transformer,
                       batch_size=32):
        
        self.net.eval()
        print("TRAIN:")
        
        o = []
        l = []

        train_inds = range(0, len(train_features))

        for b in range(0, len(train_features)/batch_size):
            batch_inds = [train_inds[idx] for idx in range(b*batch_size, (b+1)*batch_size)]
            
            train_x_batch = np.concatenate([np.expand_dims(train_features[idx]["x"], 0) for idx in batch_inds], axis=0)
            train_g_batch = np.concatenate([np.expand_dims(train_features[idx]["g"], 0) for idx in batch_inds], axis=0)

            xb = torch.from_numpy(train_x_batch.astype(np.float32)).cuda()
            gb = torch.from_numpy(train_g_batch.astype(np.float32)).cuda()
            
            self.input_x.resize_as_(xb).copy_(xb)
            self.input_g.resize_as_(gb).copy_(gb)
            
            input_xv = Variable(self.input_x)
            input_gv = Variable(self.input_g)

            output = self.net(input_gv, input_xv)
            
            if transformer is not None:
                o.append(transformer.inverse_transform(output.data.cpu().numpy().reshape((-1,1))).flatten())
                l.append(transformer.inverse_transform(y_train[batch_inds].reshape((-1,1))).flatten())
            else:
                o.append(output.data.cpu().numpy().reshape((-1,1)).flatten())
                l.append(y_train[batch_inds].reshape((-1,1)).flatten())

        o = np.concatenate(o)
        l = np.concatenate(l)
        print("RMSE:")
        print(np.sqrt(np.mean(np.square(l-o))))
        print("ROC AUC:")
        print(roc_auc_score(l, o))
        
        o = []
        l = []

        print("TEST:")
        test_inds = range(0, len(test_features))

        for b in range(0, len(test_features)/batch_size):
            batch_inds = [test_inds[idx] for idx in range(b*batch_size, (b+1)*batch_size)]
            
            test_x_batch = np.concatenate([np.expand_dims(test_features[idx]["x"], 0) for idx in batch_inds], axis=0)
            test_g_batch = np.concatenate([np.expand_dims(test_features[idx]["g"], 0) for idx in batch_inds], axis=0)

            xb = torch.from_numpy(test_x_batch.astype(np.float32)).cuda()
            gb = torch.from_numpy(test_g_batch.astype(np.float32)).cuda()
            
            self.input_x.resize_as_(xb).copy_(xb)
            self.input_g.resize_as_(gb).copy_(gb)
            
            input_xv = Variable(self.input_x)
            input_gv = Variable(self.input_g)

            output = self.net(input_gv, input_xv)
            
            if transformer is not None:
                o.append(transformer.inverse_transform(output.data.cpu().numpy().reshape((-1,1))).flatten())
                l.append(transformer.inverse_transform(y_test[batch_inds].reshape((-1,1))).flatten())
            else:
                o.append(output.data.cpu().numpy().reshape((-1,1)).flatten())
                l.append(y_test[batch_inds].reshape((-1,1)).flatten())

        o = np.concatenate(o)
        l = np.concatenate(l)
        print("RMSE:")
        print(np.sqrt(np.mean(np.square(l-o))))
        print("ROC AUC:")
        print(roc_auc_score(l, o))
