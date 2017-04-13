# 2017 DeepCrystal Technologies - Patrick Hop
#
# Message Passing Neural Network for Chemical Multigraphs
# 
# MIT License - have fun!!
# ===========================================================

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
import numpy as np

import random
from collections import OrderedDict

random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

T = 4
BATCH_SIZE = 64
MAXITER = 2000

#A = {}
# valid_bonds = {'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'}
#for valid_bond in valid_bonds:
#  A[valid_bond] = nn.Linear(75, 75)

R = nn.Linear(75, 128)
#GRU = nn.GRU(150, 75, 1)
U = nn.Linear(150, 75)

def load_dataset():
  f = open('delaney-processed.csv', 'r')
  features = []
  labels = []
  tracer = 0
  for line in f:
    if tracer == 0:
      tracer += 1
      continue
    splits =  line[:-1].split(',')
    features.append(splits[-1])
    labels.append(float(splits[-2]))

  train_features = np.array(features[:900])
  train_labels = np.array(labels[:900])
  val_features = np.array(features[900:1100])
  val_labels = np.array(labels[900:1100])

  train_labels = Variable(torch.FloatTensor(train_labels), requires_grad=False)
  val_labels = Variable(torch.FloatTensor(val_labels), requires_grad=False)

  return train_features, train_labels, val_features, val_labels

def readout(h):
  reads = map(lambda x: F.relu(R(h[x])), h.keys())
  readout = Variable(torch.zeros(1, 128))
  for read in reads:
    readout = readout + read
  return readout

def message_pass(g, h, k):
  #flow_delta = Variable(torch.zeros(1, 1))
  #h_t = Variable(torch.zeros(1, 1, 75))
  for v in g.keys():
    neighbors = g[v]
    for neighbor in neighbors:
      e_vw = neighbor[0]
      w = neighbor[1]
      #bond_type = e_vw.GetBondType()
      #A_vw = A[str(e_vw.GetBondType())]

      m_v = h[w]
      catted = torch.cat([h[v], m_v], 1)
      #gru_act, h_t = GRU(catted.view(1, 1, 150), h_t)
      
      # measure convergence
      #pdist = nn.PairwiseDistance(2)
      #flow_delta = flow_delta + torch.sum(pdist(gru_act.view(1, 75), h[v]))
      
      #h[v] = gru_act.view(1, 75)
      h[v] = U(catted)

  #print '    flow delta [%i] [%f]' % (k, flow_delta.data.numpy()[0])

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
        atom_j = molecule.GetAtomWithIdx(j)
        if i not in g:
          g[i] = []
          g[i].append( (e_ij, j) )

  return g, h

train_smiles, train_labels, val_smiles, val_labels = load_dataset()

# training loop
linear = nn.Linear(128, 1)
params = [#{'params': A['SINGLE'].parameters()},
         #{'params': A['DOUBLE'].parameters()},
         #{'params': A['TRIPLE'].parameters()},
         #{'params': A['AROMATIC'].parameters()},
         {'params': R.parameters()},
         #{'params': GRU.parameters()},
         {'params': U.parameters()},
         {'params': linear.parameters()}]

optimizer = optim.SGD(params, lr=1e-5, momentum=0.9)
for i in xrange(0, MAXITER):
  optimizer.zero_grad()
  train_loss = Variable(torch.zeros(1, 1))
  y_hats_train = []
  for j in xrange(0, BATCH_SIZE):
    sample_index = random.randint(0, 799) # TODO: sampling without replacement
    smile = train_smiles[sample_index]
    g, h = construct_multigraph(smile) # TODO: cache this

    for k in xrange(0, T):
      message_pass(g, h, k)
    
    x = readout(h)
    y_hat = linear(x)
    y = train_labels[sample_index]

    y_hats_train.append(y_hat)

    error = (y_hat - y)*(y_hat - y)
    train_loss = train_loss + error

  train_loss.backward()
  optimizer.step()

  if i % 12 == 0:
    val_loss = Variable(torch.zeros(1, 1), requires_grad=False)
    y_hats_val = []
    for j in xrange(0, len(val_smiles)):
      g, h = construct_multigraph(val_smiles[j])

      for k in xrange(0, T):
        message_pass(g, h, k)

      x = readout(h)
      y_hat = linear(x)
      y = val_labels[j]

      y_hats_val.append(y_hat)
    
      error = (y_hat - y)*(y_hat - y)
      val_loss = val_loss + error

    y_hats_val = map(lambda x: x.data.numpy()[0], y_hats_val)
    y_val = map(lambda x: x.data.numpy()[0], val_labels)
    r2_val = r2_score(y_val, y_hats_val)
  
    train_loss_ = train_loss.data.numpy()[0]
    val_loss_ = val_loss.data.numpy()[0]
    print 'epoch [%i/%i] train_loss [%f] val_loss [%f] r2_val [%s]' \
                  % ((i + 1) / 12, maxiter_train / 12, train_loss_, val_loss_, r2_val)

'''
train_labels = train_labels.data.numpy()
val_labels = val_labels.data.numpy()
  
train_mols = map(lambda x: Chem.MolFromSmiles(x), train_smiles)
train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in train_mols]
val_mols = map(lambda x: Chem.MolFromSmiles(x), val_smiles)
val_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in val_mols]

np_fps_train = []
for fp in train_fps:
  arr = np.zeros((1,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_fps_train.append(arr)

np_fps_val = []
for fp in val_fps:
  arr = np.zeros((1,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_fps_val.append(arr)

rf = RandomForestRegressor(n_estimators=100, random_state=2)
#rf.fit(np_fps_train, train_labels)
#labels = rf.predict(val_fps)

ave = np.ones( (300,) )*(np.sum(val_labels) / 300.0)

print ave.shape
print val_labels.shape
r2 =  r2_score(ave, val_labels)
print 'rf r2 is:'
print r2
'''
