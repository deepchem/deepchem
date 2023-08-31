import numpy as np
import deepchem

dataset = dc.data.NumpyDataset(np.random.rand(500, 5))

print(dataset)