import deepchem as dc
import numpy as np

N = 10
n_feat = 5
n_classes = 3
n_tasks = 1
X = np.random.rand(N, n_feat)
y = np.random.randint(3, size=(N, n_tasks))
dataset = dc.data.NumpyDataset(X, y)


