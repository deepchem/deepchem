import numpy as np
import deepchem as dc

dataset = dc.data.NumpyDataset(np.random.rand(500, 5))
print(dataset)
