# 2017 DeepCrystal Technologies - Patrick Hop
#
# Message Passing Neural Network SELU [MPNN-S] for Chemical Multigraphs
#
# MIT License - have fun!!
# ===========================================================

import math

import deepchem as dc
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import numpy as np

import random
from collections import OrderedDict
from scipy.stats import pearsonr

import donkey

random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

DATASET = 'az_ppb.csv'
print(DATASET)

T = 3
BATCH_SIZE = 48
MAXITER = 40000
LIMIT = 0
LR = 5e-4

R = nn.Linear(150, 128)
U = {0: nn.Linear(156, 75), 1: nn.Linear(156, 75), 2: nn.Linear(156, 75)}
V = {0: nn.Linear(75, 75), 1: nn.Linear(75, 75), 2: nn.Linear(75, 75)}
E = nn.Linear(6, 6)

def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by .8 every 5 epochs"""
  lr = LR * (0.9 ** (epoch // 10))
  print('new lr [%.5f]' % lr)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def load_dataset():
  train_features, train_labels, val_features, val_labels = donkey.load_dataset(DATASET)

  scaler = preprocessing.StandardScaler().fit(train_labels)
  train_labels = scaler.transform(train_labels)
  val_labels = scaler.transform(val_labels)

  train_labels = Variable(torch.FloatTensor(train_labels), requires_grad=False)
  val_labels = Variable(torch.FloatTensor(val_labels), requires_grad=False)
  
  return train_features, train_labels, val_features, val_labels

def readout(h, h2):
  catted_reads = map(lambda x: torch.cat([h[x[0]], h2[x[1]]], 1), zip(h2.keys(), h.keys()))
  activated_reads = map(lambda x: F.selu( R(x) ), catted_reads)
  readout = Variable(torch.zeros(1, 128))
  for read in activated_reads:
    readout = readout + read
  return F.tanh( readout )

def message_pass(g, h, k):
  for v in g.keys():
    neighbors = g[v]
    for neighbor in neighbors:
      e_vw = neighbor[0] # feature variable
      w = neighbor[1]
      
      m_w = V[k](h[w])
      m_e_vw = E(e_vw)
      reshaped = torch.cat( (h[v], m_w, m_e_vw), 1)
      h[v] = F.selu(U[k](reshaped))

def construct_multigraph(smile):
  g = OrderedDict({})
  h = OrderedDict({})

  molecule = Chem.MolFromSmiles(smile)
  for i in xrange(0, molecule.GetNumAtoms()):
    atom_i = molecule.GetAtomWithIdx(i)
    h[i] = Variable(torch.FloatTensor(dc.feat.graph_features.atom_features(atom_i))).view(1, 75)
    for j in xrange(0, molecule.GetNumAtoms()):
      e_ij = molecule.GetBondBetweenAtoms(i, j)
      if e_ij != None:
        e_ij =  map(lambda x: 1 if x == True else 0, dc.feat.graph_features.bond_features(e_ij)) # ADDED edge feat
        e_ij = Variable(torch.FloatTensor(e_ij).view(1, 6))
        atom_j = molecule.GetAtomWithIdx(j)
        if i not in g:
          g[i] = []
          g[i].append( (e_ij, j) )

  return g, h

train_smiles, train_labels, val_smiles, val_labels = load_dataset()

linear = nn.Linear(128, 1)
params = [{'params': R.parameters()},
         {'params': U[0].parameters()},
         {'params': U[1].parameters()},
         {'params': U[2].parameters()},
         {'params': E.parameters()},
         {'params': V[0].parameters()},
         {'params': V[1].parameters()},
         {'params': V[2].parameters()},
         {'params': linear.parameters()}]

num_epoch = 0
optimizer = optim.Adam(params, lr=LR, weight_decay=1e-4)
for i in xrange(0, MAXITER):
  optimizer.zero_grad()
  train_loss = Variable(torch.zeros(1, 1))
  y_hats_train = []
  for j in xrange(0, BATCH_SIZE):
    sample_index = random.randint(0, len(train_smiles) - 2)
    smile = train_smiles[sample_index]
    g, h = construct_multigraph(smile) # TODO: cache this

    g2, h2 = construct_multigraph(smile)
    
    for k in xrange(0, T):
      message_pass(g, h, k)

    x = readout(h, h2)
    #x = F.selu( fc(x) )
    y_hat = linear(x)
    y = train_labels[sample_index]

    y_hats_train.append(y_hat)

    error = (y_hat - y)*(y_hat - y) / Variable(torch.FloatTensor([BATCH_SIZE])).view(1, 1)
    train_loss = train_loss + error

  train_loss.backward()
  optimizer.step()

  if i % int(len(train_smiles) / BATCH_SIZE) == 0:
    val_loss = Variable(torch.zeros(1, 1), requires_grad=False)
    y_hats_val = []
    for j in xrange(0, len(val_smiles)):
      g, h = construct_multigraph(val_smiles[j])
      g2, h2 = construct_multigraph(val_smiles[j])

      for k in xrange(0, T):
        message_pass(g, h, k)

      x = readout(h, h2)
      #x = F.selu( fc(x) )
      y_hat = linear(x)
      y = val_labels[j]

      y_hats_val.append(y_hat)

      error = (y_hat - y)*(y_hat - y) / Variable(torch.FloatTensor([len(val_smiles)])).view(1, 1)
      val_loss = val_loss + error

    y_hats_val = np.array(map(lambda x: x.data.numpy(), y_hats_val))
    y_val = np.array(map(lambda x: x.data.numpy(), val_labels))
    y_hats_val = y_hats_val.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    
    r2_val_old = r2_score(y_val, y_hats_val)
    r2_val_new = pearsonr(y_val, y_hats_val)[0]**2
  
    train_loss_ = train_loss.data.numpy()[0]
    val_loss_ = val_loss.data.numpy()[0]
    print 'epoch [%i/%i] train_loss [%f] val_loss [%f] r2_val_old [%.4f], r2_val_new [%.4f]' \
                  % (num_epoch, 100, train_loss_, val_loss_, r2_val_old, r2_val_new)
    num_epoch += 1
